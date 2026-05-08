"""Triple-Branch predictor with active-sampling, fusion training, and per-slide eval."""
from __future__ import annotations

import os
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from models import TripleBranchModel, build_gene_embed_database
from models.fusion_variants import (
    build_fusion_variant,
    FUSION_VARIANT_NAMES,
)
from data_loader import (
    GeneExpressionDataset3D,
    build_neighbor_lookup,
    load_samples_with_info,
    get_transforms,
)
from config import get_model_dims, DATASET_CONFIG
from eval_metric import compute_metrics
from cross_fold import parse_file_path


class TripleBranchPredictor:
    """
    Predictor for active learning + joint triple-branch training.

    Indices passed to sampler/training are global pool indices into
    `self.train_dataset`.
    """

    def __init__(
        self,
        dataset: str,
        train_files: List[str],
        val_files: List[str],
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 8,
        mask_ratio: float = 300,
        # Model hyperparameters
        temperature: float = 0.07,
        top_k: int = 25,
        projection_dim: int = 256,
        tau_3d: float = 0.1,
        # Training schedule
        ret_epochs: int = 100,
        lambda_ret: float = 0.1,
        lambda_con: float = 0.1,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mask_ratio = mask_ratio

        self.ret_epochs = ret_epochs
        self.lambda_ret = lambda_ret
        self.lambda_con = lambda_con

        #  Build 3D neighbor lookup (all potential neighbors), query CONCH map 
        self.neighbor_data, self.pool_index_map, self.query_conch_map = \
            build_neighbor_lookup(dataset, train_files)
        self.pool_idx_to_key = {v: k for k, v in self.pool_index_map.items()}

        #  Order train files to match pool_index_map walking order 
        ordered_files = self._order_files(train_files, dataset)

        train_samples, train_info = load_samples_with_info(ordered_files, dataset)
        self.train_dataset = GeneExpressionDataset3D(
            samples=train_samples, case_slice_info=train_info,
            dataset_name=dataset, mask_ratio=mask_ratio,
            neighbor_data=self.neighbor_data,
            query_conch_map=self.query_conch_map,
            transform=get_transforms(is_train=True),
            sampled_set=None,
        )
        self.train_samples = train_samples

        #  Val loaders (per-slide) 
        self.val_loaders: Dict[str, DataLoader] = {}
        all_val_samples, all_val_info = [], []
        for f in val_files:
            case_id, slice_name = parse_file_path(f)
            slide_key = f"{case_id}/{slice_name}"
            samples, info = load_samples_with_info([f], dataset)
            all_val_samples.extend(samples)
            all_val_info.extend(info)
            ds = GeneExpressionDataset3D(
                samples=samples, case_slice_info=info,
                dataset_name=dataset, mask_ratio=mask_ratio,
                neighbor_data=self.neighbor_data,
                query_conch_map=self.query_conch_map,
                transform=get_transforms(is_train=False),
                sampled_set=None,
            )
            self.val_loaders[slide_key] = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )

        self._all_val_samples = all_val_samples
        self._all_val_info    = all_val_info
        self._rl_val_loader: Optional[DataLoader] = None

        #  Model 
        _, output_dim = get_model_dims(dataset, mask_ratio)
        self.output_dim = output_dim
        self.model = TripleBranchModel(
            n_target_genes=output_dim,
            projection_dim=projection_dim,
            temperature=temperature,
            top_k=top_k,
            tau_3d=tau_3d,
            pretrained=True,
        ).to(device)

        #  Sampler state 
        self._sampled_set: Optional[set] = None
        self._epoch: int = 0

        # Retrieval DB caches (built on demand for eval/fusion)
        self._gene_embeds_db = None
        self._all_target_genes = None

        # Fusion state (populated after train_fusion_all_variants)
        self.fusion_variants: Dict[str, Dict[int, nn.Module]] = {}  # name -> top_k -> head
        self._fusion_trained = False

    # File ordering to match data_pool traversal
    def _order_files(self, train_files, dataset):
        from config import GRAPH_ROOT
        by_patient: Dict[str, List[str]] = {}
        for f in train_files:
            case_id, _ = parse_file_path(f)
            by_patient.setdefault(case_id, []).append(f)
        ordered = []
        for patient in sorted(by_patient.keys()):
            npz_path = os.path.join(GRAPH_ROOT, dataset, f"{patient}.npz")
            if os.path.exists(npz_path):
                npz = np.load(npz_path, allow_pickle=True)
                slice_order = [str(s) for s in npz["slice_names"]]
                by_patient[patient].sort(
                    key=lambda f: slice_order.index(parse_file_path(f)[1])
                    if parse_file_path(f)[1] in slice_order else 10**9
                )
            ordered.extend(by_patient[patient])
        return ordered

    # Sampling state management (called by RL agent)
    def reveal_spots(self, pool_indices: List[int]):
        """Mark spots as sampled - their expression becomes available as 3D neighbors."""
        if self._sampled_set is None:
            self._sampled_set = set()
        self._sampled_set.update(int(i) for i in pool_indices)
        self.train_dataset.set_sampled(self._sampled_set)
        for loader in self.val_loaders.values():
            loader.dataset.set_sampled(self._sampled_set)
        if self._rl_val_loader is not None:
            self._rl_val_loader.dataset.set_sampled(self._sampled_set)
        # invalidate retrieval DB cache (selected set changed)
        self._gene_embeds_db = None

    def set_standalone_mode(self):
        """Switch to standalone mode: all neighbor expressions available."""
        self._sampled_set = None
        self.train_dataset.set_sampled(None)
        for loader in self.val_loaders.values():
            loader.dataset.set_sampled(None)
        if self._rl_val_loader is not None:
            self._rl_val_loader.dataset.set_sampled(None)

    # RL val subset
    def setup_rl_val_subset(self, n_spots: int = 256, seed: int = 0):
        if n_spots <= 0 or n_spots >= len(self._all_val_samples):
            self._rl_val_loader = None
            return
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self._all_val_samples), size=n_spots, replace=False)
        subset_samples = [self._all_val_samples[i] for i in idx]
        subset_info    = [self._all_val_info[i]    for i in idx]
        ds = GeneExpressionDataset3D(
            samples=subset_samples, case_slice_info=subset_info,
            dataset_name=self.dataset, mask_ratio=self.mask_ratio,
            neighbor_data=self.neighbor_data,
            query_conch_map=self.query_conch_map,
            transform=get_transforms(is_train=False),
            sampled_set=self._sampled_set,
        )
        self._rl_val_loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    # Snapshot / Restore
    def snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def restore(self, snap: dict):
        self.model.load_state_dict({k: v.to(self.device) for k, v in snap.items()})

    # Optimizer (called by sampler to build outer-loop optimizer)
    def get_optimizer(self, args):
        """
        Two param-groups AdamW: Ret path at lr_ret, Reg path at lr_reg.

        Reads from args (with fallback defaults):
            lr_ret       = getattr(args, 'lr_ret',  1e-3)
            lr_reg       = getattr(args, 'lr_reg',  5e-5)
            weight_decay = getattr(args, 'weight_decay', 1e-4)
        """
        lr_ret = getattr(args, 'lr_ret', 1e-3)
        lr_reg = getattr(args, 'lr_reg', 5e-5)
        wd     = getattr(args, 'weight_decay', 1e-4)
        return optim.AdamW([
            {'params': self.model.get_ret_params(), 'lr': lr_ret},
            {'params': self.model.get_reg_params(), 'lr': lr_reg},
        ], weight_decay=wd)

    # Epoch tracking + training step
    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def train_step(self, batch, device) -> torch.Tensor:
        """
        Joint forward + three-loss backward (single loss returned).

        L = L_reg + lambda_ret*L_ret (if epoch < ret_epochs) + lambda_con*L_con
        """
        # Move tensors to device (strings are passed through untouched by collate)
        batch_on_dev = {k: (v.to(device, non_blocking=True)
                            if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()}

        self.model.train()

        out = self.model.forward_joint(
            batch_on_dev,
            compute_ret=(self._epoch < self.ret_epochs),
            compute_3d_teacher=(self.lambda_con > 0),
        )

        losses = self.model.compute_joint_loss(
            out, batch_on_dev,
            epoch=self._epoch,
            ret_epochs=self.ret_epochs,
            lambda_ret=self.lambda_ret,
            lambda_con=self.lambda_con,
        )
        return losses['total']

    # Val loss for RL reward
    @torch.no_grad()
    def val_loss(self) -> float:
        """pred_reg MSE on RL val subset (or full val if subset not set up)."""
        self.model.eval()
        if self._rl_val_loader is not None:
            loaders = [self._rl_val_loader]
        else:
            loaders = list(self.val_loaders.values())

        total, n = 0.0, 0
        for loader in loaders:
            for batch in loader:
                imgs = batch['image'].to(self.device, non_blocking=True)
                tgt  = batch['target_genes'].to(self.device, non_blocking=True)
                _, pred_reg = self.model.forward_reg(imgs)
                total += F.mse_loss(pred_reg, tgt).item() * imgs.size(0)
                n += imgs.size(0)
        self.model.train()
        return total / max(1, n)

    @torch.no_grad()
    def eval_on_subset(self, indices: List[int]) -> float:
        """pred_reg MSE on a subset of training (sampled) spots."""
        if len(indices) == 0:
            return float('nan')
        self.model.eval()
        subset = Subset(self.train_dataset, indices)
        loader = DataLoader(
            subset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        total, n = 0.0, 0
        for batch in loader:
            imgs = batch['image'].to(self.device, non_blocking=True)
            tgt  = batch['target_genes'].to(self.device, non_blocking=True)
            _, pred_reg = self.model.forward_reg(imgs)
            total += F.mse_loss(pred_reg, tgt).item() * imgs.size(0)
            n += imgs.size(0)
        self.model.train()
        return total / max(1, n)

    # Retrieval DB builder
    def _build_retrieval_db(self, sample_indices: List[int]):
        """Build gene embedding DB from selected training samples."""
        target_indices = self.train_dataset.target_indices
        if sample_indices is None or len(sample_indices) == 0:
            sample_indices = list(range(len(self.train_samples)))
        samples = [self.train_samples[i] for i in sample_indices]

        all_target_genes = torch.stack([
            torch.tensor(s['expression_norm'][target_indices], dtype=torch.float32)
            for s in samples
        ])
        gene_embeds_db = build_gene_embed_database(
            self.model, all_target_genes, batch_size=256, device=self.device,
        ).to(self.device)

        self._all_target_genes = all_target_genes.to(self.device)
        self._gene_embeds_db   = gene_embeds_db

    # Precompute three-branch predictions for given indices
    @torch.no_grad()
    def _precompute_branch_preds(self, indices: List[int], top_ks: List[int]) -> Dict:
        """
        Run each sampled spot through all three branches; cache predictions.

        Returns a dict with tensors (all on CPU to save GPU memory):
            pred_ret:  {top_k: (N, G)}
            pred_reg:           (N, G)
            pred_3d:            (N, G)
            confidence:         (N,)
            target:             (N, G)
        """
        self.model.eval()

        subset = Subset(self.train_dataset, indices)
        loader = DataLoader(
            subset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

        # Ensure DB is current
        if self._gene_embeds_db is None:
            self._build_retrieval_db(indices)

        all_pred_reg, all_pred_3d, all_conf, all_target = [], [], [], []
        all_pred_ret = {k: [] for k in top_ks}
        all_img_emb = []  # for possible reuse

        for batch in loader:
            imgs    = batch['image'].to(self.device, non_blocking=True)
            target  = batch['target_genes']

            # Ret
            img_emb = self.model.get_bleep_image_embed(imgs)
            for k in top_ks:
                pred_ret_k, _ = self.model.retrieve_topk(
                    img_emb, self._gene_embeds_db, self._all_target_genes, top_k=k
                )
                all_pred_ret[k].append(pred_ret_k.cpu())

            # Reg
            _, pred_reg = self.model.forward_reg(imgs)
            all_pred_reg.append(pred_reg.cpu())

            # 3D teacher (no params)
            pred_3d, confidence, _ = self.model.forward_3d_teacher(
                batch['query_conch'].to(self.device, non_blocking=True),
                batch['neighbor_vis'].to(self.device, non_blocking=True),
                batch['neighbor_expr'].to(self.device, non_blocking=True),
                batch['neighbor_mask'].to(self.device, non_blocking=True),
                batch['neighbor_weights'].to(self.device, non_blocking=True),
                tau=self.model.tau_3d,
            )
            all_pred_3d.append(pred_3d.cpu())
            all_conf.append(confidence.cpu())
            all_target.append(target)

        return {
            'pred_ret':   {k: torch.cat(v, dim=0) for k, v in all_pred_ret.items()},
            'pred_reg':   torch.cat(all_pred_reg, dim=0),
            'pred_3d':    torch.cat(all_pred_3d,  dim=0),
            'confidence': torch.cat(all_conf,     dim=0),
            'target':     torch.cat(all_target,   dim=0),
        }

    # Fusion training (all variants x all top_ks)
    def train_fusion_all_variants(
        self,
        n_epochs: int = 30,
        lr: float = 5e-4,
        top_ks: List[int] = (25,),
        selected_indices: Optional[List[int]] = None,
        batch_size: int = 256,
        variant_names: Optional[List[str]] = None,
    ):
        """
        Train each fusion variant at each top_k using predictions on
        sampled training spots.

        Freezes backbones (they never receive gradient here because predictions
        are pre-computed under no_grad and cached).
        """
        if selected_indices is None or len(selected_indices) == 0:
            selected_indices = list(range(len(self.train_samples)))
        top_ks = list(top_ks)

        print(f"[Fusion] Building retrieval DB on {len(selected_indices)} spots...")
        self._build_retrieval_db(selected_indices)

        print(f"[Fusion] Precomputing three-branch predictions...")
        cache = self._precompute_branch_preds(selected_indices, top_ks)

        target = cache['target']                                # (N, G) on CPU
        pred_reg_all = cache['pred_reg']
        pred_3d_all  = cache['pred_3d']
        conf_all     = cache['confidence']

        names = list(variant_names) if variant_names is not None else list(FUSION_VARIANT_NAMES)
        self.fusion_variants = {name: {} for name in names}

        n = target.size(0)
        device = self.device

        for variant_name in names:
            for top_k in top_ks:
                pred_ret_all = cache['pred_ret'][top_k]
                fusion = build_fusion_variant(variant_name, n_genes=self.output_dim).to(device)
                opt = optim.AdamW(fusion.parameters(), lr=lr, weight_decay=0.0)
                for epoch in range(n_epochs):
                    perm = torch.randperm(n)
                    for start in range(0, n, batch_size):
                        idx = perm[start:start + batch_size]
                        pr = pred_ret_all[idx].to(device)
                        pg = pred_reg_all[idx].to(device)
                        p3 = pred_3d_all[idx].to(device)
                        cf = conf_all[idx].to(device)
                        tg = target[idx].to(device)
                        pred_final = fusion(pr, pg, p3, cf)
                        loss = F.mse_loss(pred_final, tg)
                        opt.zero_grad(); loss.backward(); opt.step()

                fusion.eval()
                self.fusion_variants[variant_name][top_k] = fusion
                print(f"[Fusion] trained {variant_name} @ top_k={top_k}")

        self._fusion_trained = True

    # Final per-slide eval: 3D branch and the trained fusion variants only.
    @torch.no_grad()
    def eval_per_slide_all_variants(
        self,
        top_ks: List[int] = (25,),
        save_pred_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """Per-slide metrics for 3d_only and the trained fusion variants."""
        self.model.eval()
        top_ks = list(top_ks)

        if self._gene_embeds_db is None:
            sel = (sorted(self._sampled_set) if self._sampled_set is not None
                   else list(range(len(self.train_samples))))
            self._build_retrieval_db(sel)

        rows = []

        for slide_key, loader in self.val_loaders.items():
            all_pred_reg, all_pred_3d, all_conf, all_target = [], [], [], []
            all_pred_ret: Dict[int, List[torch.Tensor]] = {k: [] for k in top_ks}

            for batch in loader:
                imgs = batch['image'].to(self.device, non_blocking=True)
                img_emb = self.model.get_bleep_image_embed(imgs)
                for k in top_ks:
                    pr, _ = self.model.retrieve_topk(
                        img_emb, self._gene_embeds_db, self._all_target_genes, top_k=k
                    )
                    all_pred_ret[k].append(pr.cpu())

                _, pred_reg = self.model.forward_reg(imgs)
                all_pred_reg.append(pred_reg.cpu())

                pred_3d, conf, _ = self.model.forward_3d_teacher(
                    batch['query_conch'].to(self.device, non_blocking=True),
                    batch['neighbor_vis'].to(self.device, non_blocking=True),
                    batch['neighbor_expr'].to(self.device, non_blocking=True),
                    batch['neighbor_mask'].to(self.device, non_blocking=True),
                    batch['neighbor_weights'].to(self.device, non_blocking=True),
                    tau=self.model.tau_3d,
                )
                all_pred_3d.append(pred_3d.cpu())
                all_conf.append(conf.cpu())
                all_target.append(batch['target_genes'])

            target = torch.cat(all_target, dim=0)
            pred_reg = torch.cat(all_pred_reg, dim=0)
            pred_3d  = torch.cat(all_pred_3d, dim=0)
            conf     = torch.cat(all_conf, dim=0)
            pred_rets = {k: torch.cat(v, dim=0) for k, v in all_pred_ret.items()}

            # 3D branch (top_k-independent).
            m = compute_metrics(pred_3d, target)
            rows.append({'slide': slide_key, 'method': '3d_only', 'top_k': '-', **m})

            # Trained fusion variants (V1, V3, V4) per top_k.
            if self._fusion_trained:
                for vname, by_k in self.fusion_variants.items():
                    for k, fusion in by_k.items():
                        pr = pred_rets[k].to(self.device)
                        pg = pred_reg.to(self.device)
                        p3 = pred_3d.to(self.device)
                        cf = conf.to(self.device)
                        pred_final = fusion(pr, pg, p3, cf).cpu()
                        m = compute_metrics(pred_final, target)
                        rows.append({'slide': slide_key, 'method': vname, 'top_k': k, **m})

                        if save_pred_dir is not None:
                            os.makedirs(save_pred_dir, exist_ok=True)
                            fname = f"{slide_key.replace('/', '_')}_{vname}_top{k}.npy"
                            np.save(os.path.join(save_pred_dir, fname), pred_final.numpy())

            if save_pred_dir is not None:
                os.makedirs(save_pred_dir, exist_ok=True)
                np.save(os.path.join(save_pred_dir,
                        f"{slide_key.replace('/', '_')}_target.npy"), target.numpy())
                np.save(os.path.join(save_pred_dir,
                        f"{slide_key.replace('/', '_')}_pred_3d.npy"), pred_3d.numpy())

        return pd.DataFrame(rows)

    # Intermediate per-slide eval during training: 3d_only.
    @torch.no_grad()
    def eval_per_slide_intermediate(self, epoch: int, save_dir: str) -> List[dict]:
        """Per-slide 3D-branch metrics, written to {save_dir}/epoch_{epoch}.csv."""
        self.model.eval()

        if self._sampled_set is not None and len(self._sampled_set) == 0:
            print(f"[eval_intermediate ep{epoch}] sampled_set is empty; skipping.")
            self.model.train()
            return []

        rows = []
        for slide_key, loader in self.val_loaders.items():
            pred_3d_list, tgt_list = [], []
            for batch in loader:
                pred_3d, _, _ = self.model.forward_3d_teacher(
                    batch['query_conch'].to(self.device, non_blocking=True),
                    batch['neighbor_vis'].to(self.device, non_blocking=True),
                    batch['neighbor_expr'].to(self.device, non_blocking=True),
                    batch['neighbor_mask'].to(self.device, non_blocking=True),
                    batch['neighbor_weights'].to(self.device, non_blocking=True),
                    tau=self.model.tau_3d,
                )
                pred_3d_list.append(pred_3d.cpu())
                tgt_list.append(batch['target_genes'])

            tgt = torch.cat(tgt_list, dim=0)
            p_3d = torch.cat(pred_3d_list, dim=0)
            m = compute_metrics(p_3d, tgt)
            rows.append({
                'slide': slide_key, 'epoch': epoch, 'method': '3d_only',
                'top_k': '-',
                'mse': float(m['mse']), 'pcc': float(m['pcc']), 'mae': float(m['mae']),
            })

        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(save_dir, f"epoch_{epoch}.csv"), index=False,
        )
        self.model.train()
        return rows

    # Legacy compat wrappers (some sampler codepaths expect these)
    def fit_short(self, *args, **kwargs):
        raise RuntimeError(
            "fit_short is not used by TripleBranchPredictor - "
            "the sampler should call train_step per-batch in its main loop."
        )

    def fit_full(self, *args, **kwargs):
        raise RuntimeError(
            "fit_full is not used by TripleBranchPredictor - "
            "the sampler should drive training via train_step."
        )
