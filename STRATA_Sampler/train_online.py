"""STRATA online active sampling with ST-Net predictor (single-pass)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_pool import build_candidate_pool                           # noqa: E402
from predictor import STNetPredictor                                 # noqa: E402
from policy import (                                                 # noqa: E402
    PointerDecoderPolicy,
    FrozenStaticProjector,
    sample_action,
)
from cell_head import CellHead, nearest_centroid                     # noqa: E402
from state_builder import (                                          # noqa: E402
    build_frozen_block,
    init_dynamic_state,
    reveal,
    frozen_block_dim,
    dynamic_block_dim,
)

ST_NET_DIR = os.environ.get("ST_NET_DIR")
if ST_NET_DIR:
    sys.path.insert(0, ST_NET_DIR)
from cross_fold import get_folds                                     # noqa: E402
from config import DATA_ROOT, DATASET_CONFIG, N_TARGET_GENES          # noqa: E402
from eval_metric import calculate_pcc                                 # noqa: E402


# 
# logging
# 
def setup_logger(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "train.log", mode="w"),
        ],
    )


# 
# policy rollout helpers
# 
def masked_softmax_sample(
    policy: DeepSetsPolicy,
    state: torch.Tensor,               # (N, feat_dim) on device
    selected_mask_bool: torch.Tensor,  # (N,) bool on device
    candidate_mask_bool: torch.Tensor, # (N,) bool on device  (not-yet-picked pool)
    k: int,
    deterministic: bool,
) -> Tuple[List[int], List[torch.Tensor], List[torch.Tensor]]:
    """Sequentially pick k spots without replacement.

    After each pick, we mask that spot out and re-score. The policy's
    `selected_mask` input reflects the *accumulated* picks for context.
    """
    acts: List[int] = []
    logps: List[torch.Tensor] = []
    ents: List[torch.Tensor] = []
    cur_sel = selected_mask_bool.clone()
    cur_cand = candidate_mask_bool.clone()
    N = cur_sel.shape[0]
    for _ in range(k):
        if not cur_cand.any():
            break
        logits, _value = policy(state, cur_sel, cur_cand)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            a = int(torch.argmax(probs).item())
        else:
            a = int(torch.multinomial(probs, num_samples=1).item())
        logp = torch.log(probs[a].clamp_min(1e-12))
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
        acts.append(a)
        logps.append(logp)
        ents.append(ent)
        # out-of-place mask update: the policy's masked_fill / bool-indexing
        # references these tensors in the autograd graph, so mutating them
        # in-place would break backward via the version counter.
        one_hot = torch.zeros(N, dtype=torch.bool, device=cur_sel.device)
        one_hot[a] = True
        cur_sel = cur_sel | one_hot
        cur_cand = cur_cand & ~one_hot
    return acts, logps, ents


# 
# 3D cross-slice redundancy tracker
# 
def build_pool_cross_neighbors(pool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each pool spot, list its cross-slice neighbors as pool indices.

    Returns three arrays:
      nb_ptr  : (N+1,)  int64 - CSR offsets into nb_idx / nb_w
      nb_idx  : (E,)    int64 - pool indices of cross-slice neighbors
      nb_w    : (E,)    float32 - corresponding cross weights
    Only neighbors that are themselves in the pool are kept.
    """
    N = pool.n_pool
    # inverse map: (patient, in_patient_idx) -> pool_global_idx
    inv_map: dict = {}
    for patient, pool_arr in pool.pool_idx_in_patient.items():
        ip_arr = pool.in_patient_global[patient]
        for pg, ipg in zip(pool_arr, ip_arr):
            inv_map[(patient, int(ipg))] = int(pg)

    # collect per spot
    per_spot_idx: List[List[int]] = [[] for _ in range(N)]
    per_spot_w: List[List[float]] = [[] for _ in range(N)]
    for patient, pool_arr in pool.pool_idx_in_patient.items():
        ip_arr = pool.in_patient_global[patient]
        cross_idx, cross_w = pool.cross[patient]  # (N_p, 25), (N_p, 25)
        for pg, ipg in zip(pool_arr, ip_arr):
            nbs = cross_idx[int(ipg)]
            wts = cross_w[int(ipg)]
            for nb, w in zip(nbs, wts):
                key = (patient, int(nb))
                if key in inv_map:
                    per_spot_idx[int(pg)].append(inv_map[key])
                    per_spot_w[int(pg)].append(float(w))

    # pack CSR
    nb_ptr = np.zeros(N + 1, dtype=np.int64)
    for i in range(N):
        nb_ptr[i + 1] = nb_ptr[i] + len(per_spot_idx[i])
    E = int(nb_ptr[-1])
    nb_idx = np.zeros(E, dtype=np.int64)
    nb_w = np.zeros(E, dtype=np.float32)
    for i in range(N):
        s = int(nb_ptr[i]); e = int(nb_ptr[i + 1])
        if e > s:
            nb_idx[s:e] = per_spot_idx[i]
            nb_w[s:e] = per_spot_w[i]
    return nb_ptr, nb_idx, nb_w


def apply_cross_redundancy_update(
    cross_red: np.ndarray,       # (N,) float32
    nb_ptr: np.ndarray, nb_idx: np.ndarray, nb_w: np.ndarray,
    new_pick: int,
):
    """When `new_pick` joins the selected set, add its cross-edge weights to
    every one of its neighbors' running redundancy score.
    """
    s = int(nb_ptr[new_pick]); e = int(nb_ptr[new_pick + 1])
    if e > s:
        cross_red[nb_idx[s:e]] += nb_w[s:e]


# 
# dyn tensor builder (small, per-step, CPU->device)
# 
def build_dyn_tensor(
    dyn_state: dict,
    cross_red: np.ndarray,
    budget_left: int,
    budget_total: int,
    N: int,
    device: str,
) -> torch.Tensor:
    """Assemble the live dynamic block as a (N, dyn_dim) float32 tensor.

    Layout (matches state_builder.dynamic_block_dim ordering):
      cross_red_norm(1) | is_sampled(1) | budget(1) | own_cell_type(C)
      | own_mol_embed(512) | observed_flag(1)
    Note: `cell_prior` is passed SEPARATELY to the policy, not concatenated
    here, because it's a live tensor produced by CellHead on device.
    """
    cr_max = float(cross_red.max()) if cross_red.size > 0 else 0.0
    cr_scale = max(cr_max, 1e-3)
    cr_norm = (cross_red / cr_scale).astype(np.float32)[:, None]
    budget_vec = np.full((N, 1), budget_left / max(1, budget_total), dtype=np.float32)
    dyn_np = np.concatenate(
        [
            cr_norm,
            dyn_state["is_sampled"][:, None],
            budget_vec,
            dyn_state["own_cell_type"],
            dyn_state["own_mol_embed"],
            dyn_state["observed_flag"][:, None],
        ],
        axis=1,
    )
    return torch.from_numpy(dyn_np).to(device)


# 
# one sampling round: no-grad rollout (memory efficient) + replay-based HER
# 
@torch.no_grad()
def rollout_no_grad(
    policy: PointerDecoderPolicy,
    h_static: torch.Tensor,           # (N, h_static_dim)  frozen cache
    cell_prior: torch.Tensor,         # (N, K)             live from CellHead
    dyn: torch.Tensor,                # (N, dyn_dim)       built from dyn_state
    candidate_mask: torch.Tensor,     # (N,) bool
    k: int,
    *,
    patient_of_spot: torch.Tensor = None,
    patient_quota_remaining: dict = None,
    patient_spots: dict = None,
) -> List[int]:
    """Sequentially pick k spots. Updates internal h_ctx but does NOT persist it.

    The trajectory memory `h_ctx` is reconstructed during HER replay.
    """
    acts: List[int] = []
    cur_cand = candidate_mask.clone()
    quota_rem = dict(patient_quota_remaining) if patient_quota_remaining is not None else None
    h_ctx = torch.zeros(policy.h_ctx_dim, device=h_static.device)
    for t in range(k):
        if not cur_cand.any():
            break
        logits = policy.score(h_static, cell_prior, dyn, h_ctx)
        a, _lp, _ent = sample_action(logits, cur_cand, deterministic=False)
        acts.append(a)
        # update h_ctx as running mean of picked spots' h_static
        h_ctx = h_ctx + (h_static[a] - h_ctx) / (t + 1)
        cur_cand[a] = False
        if quota_rem is not None and patient_of_spot is not None:
            p = int(patient_of_spot[a].item())
            quota_rem[p] = quota_rem.get(p, 0) - 1
            if quota_rem[p] <= 0 and patient_spots is not None and p in patient_spots:
                idx_p = patient_spots[p].to(cur_cand.device)
                cur_cand[idx_p] = False
    return acts


def update_policy_her_replay(
    *,
    policy: PointerDecoderPolicy,
    policy_opt: torch.optim.Optimizer,
    pool,
    h_static: torch.Tensor,            # (N, h_static_dim)  frozen cache
    cell_prior_at_round: torch.Tensor, # (N, K)  frozen snapshot from CellHead
    dyn_start: dict,
    cross_red_start: np.ndarray,
    actions: List[int],
    r_true: float,
    dense_per_step: List[float],
    target_spots: int,
    budget_left_at_start: int,
    ema_baseline: dict,
    entropy_coef: float,
    device: str,
    chunk_size: int = 32,
    replay_epochs: int = 5,
    nb_ptr: np.ndarray = None,
    nb_idx: np.ndarray = None,
    nb_w: np.ndarray = None,
) -> dict:
    """Multi-epoch replay-based HER REINFORCE update on the pointer decoder.

    For each `replay_epoch`:
      - walks the saved action sequence in chunks of `chunk_size`;
      - each chunk recomputes logp(a_t) with the current policy, rebuilds the
        per-step dyn / h_ctx, and runs one backward + step.
    EMA baseline is updated ONCE per round (at the start of the first epoch),
    so multi-epoch replay reuses the same advantage normalisation.
    """
    if len(actions) == 0:
        return {"policy_loss": 0.0, "entropy": 0.0, "r_true": 0.0, "dense_mean": 0.0, "adv_mean": 0.0, "n_chunks": 0, "n_updates": 0}

    rewards = [float(r_true) + float(d) for d in dense_per_step]
    mean_r = float(np.mean(rewards))
    ema = ema_baseline.get("value", None)
    ema_sq = ema_baseline.get("value_sq", None)
    if ema is None:
        ema = mean_r; ema_sq = mean_r * mean_r
    else:
        ema = 0.9 * ema + 0.1 * mean_r
        ema_sq = 0.9 * ema_sq + 0.1 * (mean_r * mean_r)
    ema_baseline["value"] = ema
    ema_baseline["value_sq"] = ema_sq
    var = max(ema_sq - ema * ema, 1e-8)
    std = var ** 0.5
    advs_all = [float(np.clip((r - ema) / std, -5.0, 5.0)) for r in rewards]

    N = pool.n_pool
    loss_vals: List[float] = []
    ent_vals: List[float] = []
    n_updates = 0

    for epoch_pass in range(replay_epochs):
        # reset replay state at the start of every epoch pass
        dyn_rep = {k_: v.copy() for k_, v in dyn_start.items()}
        cr_rep = cross_red_start.copy()
        h_ctx = torch.zeros(policy.h_ctx_dim, device=device)

        for chunk_start in range(0, len(actions), chunk_size):
            chunk_end = min(len(actions), chunk_start + chunk_size)
            chunk_acts = actions[chunk_start:chunk_end]
            chunk_advs = advs_all[chunk_start:chunk_end]

            logp_list: List[torch.Tensor] = []
            ent_list: List[torch.Tensor] = []
            for t_in_chunk, a in enumerate(chunk_acts):
                t_global = chunk_start + t_in_chunk
                bl = max(0, budget_left_at_start - t_global)
                dyn_t = build_dyn_tensor(dyn_rep, cr_rep, bl, target_spots, N, device)
                cand_mask = torch.from_numpy(dyn_rep["is_sampled"] == 0).bool().to(device)

                logits = policy.score(h_static, cell_prior_at_round, dyn_t, h_ctx.detach())
                logits = logits.masked_fill(~cand_mask, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                log_probs_safe = torch.log(probs.clamp_min(1e-12))
                logp_list.append(log_probs_safe[a])
                ent_list.append(-(probs * log_probs_safe).sum())

                dyn_rep["is_sampled"][a] = 1.0
                dyn_rep["observed_flag"][a] = 1.0
                dyn_rep["own_cell_type"][a] = pool.cell_type[a]
                dyn_rep["own_mol_embed"][a] = pool.mol_embed[a]
                if nb_ptr is not None:
                    s = int(nb_ptr[a]); e = int(nb_ptr[a + 1])
                    if e > s:
                        cr_rep[nb_idx[s:e]] += nb_w[s:e]
                with torch.no_grad():
                    h_ctx = h_ctx + (h_static[a] - h_ctx) / (t_global + 1)

            logp_stack = torch.stack(logp_list)
            adv_stack = torch.tensor(chunk_advs, dtype=logp_stack.dtype, device=logp_stack.device)
            ent_mean = torch.stack(ent_list).mean()
            loss = -(logp_stack * adv_stack).mean() - entropy_coef * ent_mean

            policy_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            policy_opt.step()
            loss_vals.append(float(loss.item()))
            ent_vals.append(float(ent_mean.item()))
            n_updates += 1

            del logp_list, ent_list, logp_stack, adv_stack, ent_mean, loss
        torch.cuda.empty_cache()

    return {
        "policy_loss": float(np.mean(loss_vals)) if loss_vals else 0.0,
        "entropy": float(np.mean(ent_vals)) if ent_vals else 0.0,
        "r_true": float(r_true),
        "dense_mean": float(np.mean(dense_per_step)) if dense_per_step else 0.0,
        "adv_mean": float(np.mean(advs_all)),
        "n_chunks": len(loss_vals),
        "n_updates": n_updates,
    }


# 
# Per-slide eval.
# 
@torch.no_grad()
def eval_per_slide(model: nn.Module, val_loaders: dict, epoch: int,
                   save_gt_path: str, save_pred_path: str) -> List[dict]:
    model.eval()
    rows = []
    for slide_name, loader in val_loaders.items():
        preds_list, tgts_list = [], []
        for imgs, tgt in loader:
            imgs = imgs.cuda(non_blocking=True)
            pred = model(imgs).cpu()
            preds_list.append(pred)
            tgts_list.append(tgt)
        preds = torch.cat(preds_list).numpy()
        targets = torch.cat(tgts_list).numpy()
        mse = float(np.mean((preds - targets) ** 2))
        mae = float(np.mean(np.abs(preds - targets)))
        pcc = float(calculate_pcc(torch.tensor(preds), torch.tensor(targets)).item())
        epdir = os.path.join(save_pred_path, f"epoch_{epoch}")
        os.makedirs(epdir, exist_ok=True)
        np.save(os.path.join(epdir, f"{slide_name}_pred.npy"), preds)
        np.save(os.path.join(save_gt_path, f"{slide_name}_gt.npy"), targets)
        rows.append({"slide": slide_name, "epoch": epoch, "mse": mse, "pcc": pcc, "mae": mae})
        logging.info(f"    [Eval ep {epoch} | {slide_name}] MSE={mse:.4f}  PCC={pcc:.4f}  MAE={mae:.4f}")
    model.train()
    return rows


# 
# per-fold pipeline
# 
def run_fold(args, fold_idx: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = f"{args.out_root}/{args.dataset}/ours_{args.total_ratio}"
    out_dir = Path(base) / f"fold{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_weight_path = out_dir / "checkpoint_epoch"
    save_gt_path = out_dir / "gt"
    save_pred_path = out_dir / "pred"
    save_metrics_path = out_dir / "slidewise_metrics"
    for p in [save_weight_path, save_gt_path, save_pred_path, save_metrics_path]:
        p.mkdir(parents=True, exist_ok=True)
    setup_logger(out_dir)

    logging.info(f"{'='*70}")
    logging.info(f"3D-ACK-Online  |  Dataset={args.dataset}  fold={fold_idx}")
    logging.info(f"  total_ratio={args.total_ratio}  target_genes={N_TARGET_GENES}")
    logging.info(f"  warmup={args.warmup_epochs}  interval={args.sample_interval}  cutoff={args.sampling_cutoff}")
    logging.info(
        f"  gamma_redundancy={args.gamma_redundancy}  "
        f"entropy_coef={args.entropy_coef_start}->{args.entropy_coef}  "
        f"patient_cap_x{args.patient_cap_multiplier}"
    )
    logging.info(f"{'='*70}")

    folds = get_folds(args.dataset)
    split = folds[fold_idx]
    label_root = os.path.join(DATA_ROOT, args.dataset)
    train_files = [os.path.join(label_root, f) for f in split["train"]]
    val_files = [os.path.join(label_root, f) for f in split["val"]]

    #  build candidate pool (= indexing of predictor.train_dataset)
    pool = build_candidate_pool(args.dataset, train_files, val_files)
    logging.info(f"  candidate pool: N={pool.n_pool}  cell_type_dim={pool.cell_type.shape[1]}")

    #  build predictor (owns train_dataset, val_loaders, model)
    predictor = STNetPredictor(
        dataset=args.dataset,
        train_files=train_files,
        val_files=val_files,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    assert pool.n_pool == len(predictor.train_dataset), (
        f"pool ({pool.n_pool}) vs train_dataset ({len(predictor.train_dataset)}) mismatch"
    )
    predictor.setup_rl_val_subset(n_spots=args.rl_val_subsample, seed=args.seed)
    _, output_dim = predictor.output_dim, predictor.output_dim  # (n_target,)
    logging.info(f"  model: Image -> {output_dim} target genes")

    #  pool tensor caches (frozen block, patient int map, etc.)
    frozen_block = build_frozen_block(pool)
    pool_tensor_cache = {"frozen": frozen_block}
    cell_type_dim = pool.cell_type.shape[1]
    K_sc = pool.n_sc_clusters
    dyn_state = init_dynamic_state(pool.n_pool, cell_type_dim)

    #  v3: frozen static projector + cached h_static (per fold)
    static_proj = FrozenStaticProjector(
        in_dim=frozen_block_dim(), hidden=args.policy_hidden
    ).to(device)
    frozen_block_t = torch.from_numpy(frozen_block).to(device)
    with torch.no_grad():
        h_static = static_proj(frozen_block_t)   # (N, policy_hidden)
    logging.info(
        f"  h_static: {tuple(h_static.shape)}  (frozen projection of 3D block)"
    )

    #  v3: on-device tensors for CellHead training + SC novelty
    conch_t = torch.from_numpy(pool.vision).to(device)              # (N, 512)
    mol_embed_t = torch.from_numpy(pool.mol_embed).to(device)       # (N, 512)
    if K_sc > 0:
        sc_centroids_t = torch.from_numpy(pool.sc_centroids).to(device)  # (K, 512)
        sc_covered = np.zeros(K_sc, dtype=np.int64)
    else:
        sc_centroids_t = None
        sc_covered = np.zeros(0, dtype=np.int64)
    logging.info(f"  SC prototypes: K={K_sc}")

    #  v3: CellHead (online co-trained 512->256->K)
    cell_head = CellHead(
        in_dim=512, hidden=256, n_clusters=max(K_sc, 1),
        dropout=0.1,
    ).to(device)
    # initial cell_prior: uniform until first training step
    cell_prior_np = np.ones((pool.n_pool, max(K_sc, 1)), dtype=np.float32) / max(K_sc, 1)
    cell_prior_t = torch.from_numpy(cell_prior_np).to(device)
    n_cell_head_params = sum(p.numel() for p in cell_head.parameters())
    logging.info(f"  CellHead params = {n_cell_head_params/1e3:.1f}k")

    #  3D cross-slice neighbors (CSR) + running redundancy tracker
    nb_ptr, nb_idx, nb_w = build_pool_cross_neighbors(pool)
    logging.info(
        f"  3D cross-slice graph: {pool.n_pool} spots, "
        f"avg cross-degree={len(nb_idx) / max(1, pool.n_pool):.1f}"
    )
    cross_red = np.zeros(pool.n_pool, dtype=np.float32)
    test_prox = pool.test_proximity.astype(np.float32)       # (N,)
    # test_proximity is non-zero only for anchor-slide (sibling) spots
    n_anchor_spots = int((test_prox > 0).sum())
    logging.info(f"  anchor-slide (sibling-patient) spots in pool: {n_anchor_spots}")
    # normalize test_prox to [0, 1] for stable shaping magnitudes
    tp_max = float(test_prox.max()) if test_prox.max() > 0 else 1.0
    test_prox_norm = test_prox / tp_max

    #  per-patient quota cap (structural anti-collapse guard)
    # rationale: without a cap the policy can collapse the whole round onto
    # one patient's anchor slide and overfit its exact morphology, hurting
    # adjacent val slices. we allow up to `patient_cap_multiplier x (patient's
    # proportional share)` of the total budget per patient.
    patient_str_to_int = {
        p: i for i, p in enumerate(sorted(set(pool.patient_id)))
    }
    patient_of_spot_np = np.asarray(
        [patient_str_to_int[p] for p in pool.patient_id], dtype=np.int64
    )
    patient_of_spot_t = torch.from_numpy(patient_of_spot_np).to(device)
    patient_spots_t: dict = {}
    pool_share: dict = {}
    for p_str, p_int in patient_str_to_int.items():
        idx = np.where(patient_of_spot_np == p_int)[0].astype(np.int64)
        patient_spots_t[p_int] = torch.from_numpy(idx)
        pool_share[p_int] = len(idx) / pool.n_pool

    # Sampling schedule.
    trigger_epochs = [
        e for e in range(args.max_epochs)
        if e == 0
        or (e >= args.warmup_epochs
            and e + 1 <= args.sampling_cutoff
            and (e - args.warmup_epochs) % args.sample_interval == 0)
    ]
    num_rounds = max(1, len(trigger_epochs))
    target_spots = max(1, int(args.total_ratio * pool.n_pool))
    per_round_k = max(1, target_spots // num_rounds)
    # last round absorbs the remainder so total commits == target_spots exactly
    last_round_k = target_spots - per_round_k * (num_rounds - 1)
    logging.info(
        f"  sampling: total_ratio={args.total_ratio}  "
        f"target_spots={target_spots}  rounds={num_rounds}  "
        f"per_round_k={per_round_k}  last_round_k={last_round_k}"
    )

    # per-patient quota: cap at (share x target x multiplier), at least the patient's share
    patient_cap: dict = {}
    for p_int, share in pool_share.items():
        cap = int(np.ceil(share * target_spots * args.patient_cap_multiplier))
        cap = min(cap, int(patient_spots_t[p_int].numel()))
        patient_cap[p_int] = cap
    total_cap = sum(patient_cap.values())
    logging.info(
        f"  patient quota caps (x{args.patient_cap_multiplier}): "
        f"sum={total_cap} (target={target_spots})  "
        + "  ".join(f"p{k}={v}" for k, v in sorted(patient_cap.items()))
    )
    if total_cap < target_spots:
        raise RuntimeError(
            f"patient cap sum ({total_cap}) < target_spots ({target_spots}); "
            f"increase --patient_cap_multiplier"
        )
    patient_selected: dict = {p: 0 for p in patient_cap}

    #  v3 PointerDecoder policy
    dyn_dim = dynamic_block_dim(cell_type_dim, K_sc) - K_sc  # dyn excludes cell_prior
    policy = PointerDecoderPolicy(
        h_static_dim=args.policy_hidden,
        cell_prior_dim=max(K_sc, 1),
        dyn_dim=dyn_dim,
        h_ctx_dim=args.policy_hidden,
        hidden=args.policy_hidden,
    ).to(device)
    n_policy_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(
        f"  policy(PointerDecoder): h_static={args.policy_hidden} "
        f"cell_prior={K_sc} dyn={dyn_dim} h_ctx={args.policy_hidden}  "
        f"trainable={n_policy_params/1e3:.1f}k"
    )
    policy_opt = optim.Adam(policy.parameters(), lr=args.policy_lr)

    # Main predictor optimizer.
    predictor_model = predictor.model
    for p in predictor_model.parameters():
        p.requires_grad = True
    predictor_opt = optim.SGD(
        predictor_model.parameters(),
        lr=args.base_lr, momentum=0.9, weight_decay=1e-4,
    )
    loss_fn = nn.MSELoss()

    #  bookkeeping
    selected_indices: List[int] = []
    selected_set: set = set()
    iter_num = 0
    ema_baseline = {}
    round_log = []
    history_rows = []

    #  round staging: we collect (logps, ents, picks, dense) at commit time,
    # then close out the round with the true DeltaL_val after the next training
    # segment finishes.
    pending_round: dict | None = None
    L_val_prev = None   # scalar val loss *before* this round's contribution

    #  CV reward: split sampled spots into pseudo_train/pseudo_val
    #    using deterministic hash so each spot always stays in the same group
    CV_SEED = 42
    CV_VAL_MOD = 5  # ~20% pseudo_val

    def _get_pseudo_val_indices() -> list:
        """Return indices of sampled spots assigned to pseudo_val."""
        return [i for i in selected_indices if hash((i, CV_SEED)) % CV_VAL_MOD == 0]

    def _compute_val_loss() -> float:
        """Compute loss on pseudo_val subset of sampled data (CV reward).
        Falls back to real val_loss if too few pseudo_val spots."""
        pseudo_val = _get_pseudo_val_indices()
        if len(pseudo_val) < 10:
            return float(predictor.val_loss())
        return float(predictor.eval_on_subset(pseudo_val))

    def _close_pending(r_log_tag: str):
        """Compute DeltaL_val and run replay-based HER update on the pending round."""
        nonlocal pending_round, L_val_prev
        if pending_round is None:
            return
        L_val_curr = _compute_val_loss()
        r_true = float(L_val_prev - L_val_curr)
        # entropy anneal: high early (explore), low late (exploit)
        n_done = len(round_log)
        frac = min(1.0, n_done / max(1, num_rounds - 1))
        ent_coef_round = args.entropy_coef_start * (1 - frac) + args.entropy_coef * frac
        stats = update_policy_her_replay(
            policy=policy,
            policy_opt=policy_opt,
            pool=pool,
            h_static=h_static,
            cell_prior_at_round=pending_round["cell_prior_at_round"],
            dyn_start=pending_round["dyn_start"],
            cross_red_start=pending_round["cross_red_start"],
            actions=pending_round["actions"],
            r_true=r_true,
            dense_per_step=pending_round["dense"],
            target_spots=target_spots,
            budget_left_at_start=pending_round["budget_left_at_start"],
            ema_baseline=ema_baseline,
            entropy_coef=ent_coef_round,
            device=device,
            chunk_size=args.her_chunk_size,
            replay_epochs=args.her_replay_epochs,
            nb_ptr=nb_ptr, nb_idx=nb_idx, nb_w=nb_w,
        )
        stats["ent_coef"] = ent_coef_round
        stats["round"] = len(round_log)
        stats["L_prev"] = float(L_val_prev)
        stats["L_curr"] = float(L_val_curr)
        round_log.append(stats)
        logging.info(
            f"    [HER {r_log_tag} r={len(round_log)-1}] DeltaL_val={r_true:+.4e}  "
            f"dense_mean={stats['dense_mean']:+.3f}  adv_mean={stats['adv_mean']:+.3f}  "
            f"pi_loss={stats['policy_loss']:+.3f}  H={stats['entropy']:.3f}  "
            f"updates={stats.get('n_updates', stats['n_chunks'])}"
        )
        L_val_prev = L_val_curr
        pending_round = None

    start_time = time.time()
    for epoch in range(args.max_epochs):
        if epoch in trigger_epochs:
            #  close out the PREVIOUS round (if any): compute DeltaL_val + HER replay
            _close_pending("close")

            #  sample a new round
            budget_so_far = len(selected_indices)
            is_last_round = (len(round_log) + 1) == num_rounds
            k_this_round = last_round_k if is_last_round else per_round_k
            k_this_round = min(k_this_round, target_spots - budget_so_far)
            logging.info(f"  [Round @ ep {epoch+1}] sampling k={k_this_round}")

            # snapshot dyn_state + cross_red + cell_prior for HER replay
            dyn_snapshot = {k_: v.copy() for k_, v in dyn_state.items()}
            cross_red_snapshot = cross_red.copy()
            cell_prior_at_round = cell_prior_t.detach().clone()   # frozen for HER
            budget_left_start = max(0, target_spots - budget_so_far)

            # build live dyn tensor + candidate mask
            dyn_t = build_dyn_tensor(
                dyn_state, cross_red, budget_left_start, target_spots,
                pool.n_pool, device,
            )
            cand_mask = torch.from_numpy(dyn_state["is_sampled"] == 0).bool().to(device)

            # per-patient quota remaining at round start
            quota_rem_now = {
                p: max(0, patient_cap[p] - patient_selected[p])
                for p in patient_cap
            }
            acts = rollout_no_grad(
                policy, h_static, cell_prior_t, dyn_t, cand_mask, k=k_this_round,
                patient_of_spot=patient_of_spot_t,
                patient_quota_remaining=quota_rem_now,
                patient_spots=patient_spots_t,
            )
            del dyn_t, cand_mask
            torch.cuda.empty_cache()

            # dense shaping(-g*cross_red + eta_sc*SC novelty) + real commit
            dense_per_step: List[float] = []
            newly_committed_idx: List[int] = []
            for a in acts:
                # 3D cross-slice redundancy penalty (potential-based diversity)
                d = -args.gamma_redundancy * float(cross_red[a])
                # SC-atlas TF-IDF novelty using post-sampling mol_embed
                if K_sc > 0:
                    emb = pool.mol_embed[a]
                    dists = np.linalg.norm(pool.sc_centroids - emb[None, :], axis=1)
                    k_star = int(dists.argmin())
                    # log(N / (1 + covered)) ranges from log(N) to ~0
                    sc_bonus = np.log(target_spots / (1.0 + sc_covered[k_star]))
                    d += args.eta_sc_novelty * float(sc_bonus)
                    sc_covered[k_star] += 1
                dense_per_step.append(d)
                if a not in selected_set:
                    selected_indices.append(int(a))
                    selected_set.add(int(a))
                    newly_committed_idx.append(int(a))
                    reveal(dyn_state, pool, int(a))
                    apply_cross_redundancy_update(cross_red, nb_ptr, nb_idx, nb_w, int(a))
                    p_int = int(patient_of_spot_np[a])
                    patient_selected[p_int] = patient_selected.get(p_int, 0) + 1

            if L_val_prev is None:
                L_val_prev = _compute_val_loss()

            pending_round = {
                "dyn_start": dyn_snapshot,
                "cross_red_start": cross_red_snapshot,
                "cell_prior_at_round": cell_prior_at_round,
                "actions": acts,
                "dense": dense_per_step,
                "budget_left_at_start": budget_left_start,
            }
            anchor_frac = float(np.mean([pool.same_patient_flag[a] for a in acts]))
            logging.info(
                f"    committed {len(acts)} spots  total={len(selected_indices)}/{target_spots}  "
                f"test_prox_mean={np.mean([float(test_prox_norm[a]) for a in acts]):.3f}  "
                f"anchor_frac={anchor_frac:.3f}  "
                f"sc_covered={int((sc_covered > 0).sum())}/{K_sc}"
            )

            #  v3: CellHead co-train on newly committed (CONCH, nearest SC cluster)
            if K_sc > 0 and len(newly_committed_idx) > 0:
                new_idx_t = torch.as_tensor(newly_committed_idx, dtype=torch.long, device=device)
                X_new = conch_t[new_idx_t]                   # (m, 512)
                sc_new = mol_embed_t[new_idx_t]              # (m, 512)
                with torch.no_grad():
                    y_new = nearest_centroid(sc_new, sc_centroids_t)
                stats_head = cell_head.fit_incremental(
                    X_new, y_new, steps=args.cell_head_steps, bs=128, lr=1e-3,
                )
                # re-forward whole pool to refresh cell_prior for next round's state
                with torch.no_grad():
                    cell_prior_np = cell_head.predict_all(conch_t, batch_size=4096)
                cell_prior_t = torch.from_numpy(cell_prior_np).to(device)
                logging.info(
                    f"    CellHead: buf={stats_head['n_buffer']}  "
                    f"loss={stats_head['loss']:.4f}  "
                    f"prior_entropy={(-(cell_prior_np * np.log(cell_prior_np + 1e-12)).sum(1)).mean():.3f}"
                )

        if len(selected_indices) == 0:
            continue

        #  real ST-Net training step on current selected subset
        train_subset = Subset(predictor.train_dataset, selected_indices)
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        predictor_model.train()
        for imgs, tgt in train_loader:
            lr = args.base_lr * (1 - iter_num / args.max_iterations) ** args.lr_decay
            for pg in predictor_opt.param_groups:
                pg["lr"] = lr
            imgs = imgs.cuda(non_blocking=True)
            tgt = tgt.cuda(non_blocking=True)
            pred = predictor_model(imgs)
            loss = loss_fn(pred, tgt)
            predictor_opt.zero_grad()
            loss.backward()
            predictor_opt.step()
            iter_num += 1
            if iter_num >= args.max_iterations:
                break

        #  eval every 10 epochs (matches baseline)
        if (epoch + 1) % 10 == 0 or epoch == args.max_epochs - 1:
            rows = eval_per_slide(
                predictor_model,
                predictor.val_loaders,
                epoch + 1,
                str(save_gt_path),
                str(save_pred_path),
            )
            df = pd.DataFrame(rows)
            df.to_csv(save_metrics_path / f"epoch_{epoch+1}.csv", index=False)
            history_rows.extend(rows)
            mean_pcc = float(np.mean([r["pcc"] for r in rows]))
            mean_mse = float(np.mean([r["mse"] for r in rows]))
            logging.info(
                f"  [epoch {epoch+1}/{args.max_epochs}] "
                f"selected={len(selected_indices)}  val PCC={mean_pcc:.4f}  MSE={mean_mse:.4f}"
            )

        if iter_num >= args.max_iterations:
            break

    #  flush any pending round (HER close-out on final val loss)
    _close_pending("final")

    #  BUDGET ASSERT: selected total must equal target_spots
    assert len(selected_indices) == target_spots, (
        f"budget violation: selected={len(selected_indices)} vs target={target_spots}"
    )

    elapsed = (time.time() - start_time) / 60
    logging.info(f"Fold {fold_idx} finished in {elapsed:.1f} min")

    #  save artifacts
    torch.save(predictor_model.state_dict(), save_weight_path / "final.pth")
    np.save(out_dir / "selected_indices.npy", np.asarray(sorted(selected_indices), dtype=np.int64))
    slide_counts = Counter(pool.sample_id[i] for i in selected_indices)
    with (out_dir / "slide_counts.json").open("w") as fh:
        json.dump(slide_counts, fh, indent=2)
    with (out_dir / "round_log.json").open("w") as fh:
        json.dump(round_log, fh, indent=2)

    #  anchor-slide selection fraction: how many selected spots are on
    # same-patient (sibling) slides? This is the natural metric for whether
    # the policy learned to prefer the anchor slide without being told.
    sel_arr = np.asarray(selected_indices, dtype=np.int64)
    n_on_anchor = int(pool.same_patient_flag[sel_arr].sum())
    frac_anchor = n_on_anchor / max(1, len(selected_indices))
    pool_frac_anchor = float(pool.same_patient_flag.mean())
    logging.info(
        f"ANCHOR  selected on sibling slides = {n_on_anchor}/{len(selected_indices)} "
        f"({frac_anchor:.3f})  vs pool baseline {pool_frac_anchor:.3f}  "
        f"(ratio = {frac_anchor / max(pool_frac_anchor, 1e-6):.2f}x)"
    )

    # final per-slide metrics in same format as old pipeline
    final_rows = eval_per_slide(
        predictor_model,
        predictor.val_loaders,
        args.max_epochs,
        str(save_gt_path),
        str(save_pred_path),
    )
    per_slide = {
        r["slide"]: {"mse": r["mse"], "mae": r["mae"], "pcc": r["pcc"]}
        for r in final_rows
    }
    with (out_dir / "final_metrics.json").open("w") as fh:
        json.dump(per_slide, fh, indent=2)
    macro_pcc = float(np.mean([m["pcc"] for m in per_slide.values()]))
    macro_mse = float(np.mean([m["mse"] for m in per_slide.values()]))
    logging.info(f"FINAL  macro PCC={macro_pcc:.4f}  macro MSE={macro_mse:.4f}")


# 
# Cross-fold metric merging.
# 
def merge_epoch_metrics_to_excel(root_dir, save_dir):
    """Merge slidewise metrics from all folds into per-epoch Excel files."""
    from glob import glob as _glob
    all_folds = sorted(_glob(os.path.join(root_dir, "fold*")))
    if not all_folds:
        logging.warning(f"No fold directories found in {root_dir}")
        return

    epoch_dict = {}
    for fold_dir in all_folds:
        fold_idx = int(os.path.basename(fold_dir).replace("fold", ""))
        metrics_dir = os.path.join(fold_dir, "slidewise_metrics")
        if not os.path.isdir(metrics_dir):
            continue
        for csv_file in _glob(os.path.join(metrics_dir, "epoch_*.csv")):
            epoch_name = os.path.splitext(os.path.basename(csv_file))[0]
            if epoch_name not in epoch_dict:
                epoch_dict[epoch_name] = []
            df = pd.read_csv(csv_file)
            df["fold"] = fold_idx
            epoch_dict[epoch_name].append(df)

    if not epoch_dict:
        logging.warning(f"No epoch CSVs found in {root_dir}")
        return

    os.makedirs(save_dir, exist_ok=True)
    for epoch_name, df_list in epoch_dict.items():
        df_epoch = pd.concat(df_list, ignore_index=True)
        epoch_num = epoch_name.split("_")[1]
        xlsx_path = os.path.join(save_dir, f"epoch_{epoch_num}.xlsx")
        df_epoch.to_excel(xlsx_path, index=False)

    # avg_summary.xlsx - one row per (epoch, top_k)
    parts = []
    for epoch_name, df_list in epoch_dict.items():
        df_all = pd.concat(df_list, ignore_index=True)
        epoch_num = int(epoch_name.split("_")[1])
        if "top_k" in df_all.columns:
            g = df_all.groupby("top_k", sort=True).agg(
                mse=("mse", "mean"), mae=("mae", "mean"),
                pcc=("pcc", "mean"), n_rows=("slide", "size"),
            ).reset_index()
        else:
            g = pd.DataFrame([{
                "top_k": 0,
                "mse": df_all["mse"].mean(),
                "mae": df_all["mae"].mean(),
                "pcc": df_all["pcc"].mean(),
                "n_rows": len(df_all),
            }])
        g.insert(0, "epoch", epoch_num)
        parts.append(g)
    avg_df = pd.concat(parts, ignore_index=True).sort_values(
        ["epoch", "top_k"]).reset_index(drop=True)
    avg_df.to_excel(os.path.join(save_dir, "avg_summary.xlsx"), index=False)
    logging.info(f"Merged metrics ({len(epoch_dict)} epochs) -> {save_dir}")


# 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HER2", choices=["HER2"])
    p.add_argument("--fold", type=int, default=-1, help="run one fold (-1 = all)")
    p.add_argument("--total_ratio", type=float, default=0.3,
                   help="fraction of the candidate pool to eventually label")
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--sample_interval", type=int, default=5)
    p.add_argument("--sampling_cutoff", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=196)
    p.add_argument("--max_epochs", type=int, default=200,
                   help="enough epochs for the predictor to converge at high sampling ratios")
    p.add_argument("--max_iterations", type=int, default=100_000_000)
    p.add_argument("--base_lr", type=float, default=5e-5)
    p.add_argument("--lr_decay", type=float, default=0.9)
    p.add_argument("--num_workers", type=int, default=4)
    # online RL hparams
    p.add_argument("--rl_val_subsample", type=int, default=256,
                   help="fixed val subset for fast L_val eval during RL "
                        "(used only as the HER closing signal, not as head-fit)")
    p.add_argument("--policy_hidden", type=int, default=128)
    p.add_argument("--policy_lr", type=float, default=3e-4)
    p.add_argument("--entropy_coef", type=float, default=0.01,
                   help="entropy bonus coef at the LAST round (after anneal)")
    p.add_argument("--entropy_coef_start", type=float, default=0.1,
                   help="entropy bonus coef at the FIRST round; linearly "
                        "annealed to --entropy_coef by the final round")
    p.add_argument("--patient_cap_multiplier", type=float, default=1.3,
                   help="per-patient quota = share x target x multiplier; "
                        "1.0 = strict proportional, 1.3 = allow 30% over-"
                        "pick per patient, inf = unbounded")
    p.add_argument("--gamma_redundancy", type=float, default=0.05,
                   help="weight on 3D cross-slice redundancy penalty in the "
                        "dense shaping term (potential-based diversity only)")
    p.add_argument("--her_chunk_size", type=int, default=16,
                   help="chunk size for replay-based HER update (bounds memory)")
    p.add_argument("--her_replay_epochs", type=int, default=10,
                   help="number of replay passes per round in HER update; "
                        "total updates per fold = rounds x ceil(k_round/chunk) "
                        "x replay_epochs (default: 5x41x10 ≈ 2050/fold)")
    p.add_argument("--eta_sc_novelty", type=float, default=0.03,
                   help="weight on SC-atlas TF-IDF novelty bonus in dense reward")
    p.add_argument("--cell_head_steps", type=int, default=5,
                   help="SGD steps per round for online CellHead co-training")
    # infra
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out_root", default="./weights_st")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    folds_all = list(range(len(get_folds(args.dataset))))
    folds = [args.fold] if args.fold >= 0 else folds_all
    for fi in folds:
        run_fold(args, fi)

    # merge results across folds (only when running all folds)
    if args.fold < 0:
        root_dir = f"{args.out_root}/{args.dataset}/ours_{args.total_ratio}"
        merge_epoch_metrics_to_excel(
            root_dir=root_dir,
            save_dir=os.path.join(root_dir, "merged_metrics"),
        )
        logging.info("All experiments completed!")


if __name__ == "__main__":
    main()
