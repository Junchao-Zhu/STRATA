"""
Standalone training entry for the Triple-Branch model (no sampler).
Trains on the full training pool, then runs fusion training and eval.
"""
import os
import sys
import time
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import DATA_ROOT
from cross_fold import get_folds
from predictor import TripleBranchPredictor


def setup_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_fold(args, fold_idx: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = f"{args.weight_dir}/{args.dataset}/mask_{int(args.mask_ratio)}"
    out_dir = os.path.join(base, f"fold{fold_idx}")
    os.makedirs(out_dir, exist_ok=True)
    setup_logger(out_dir)

    logging.info(f"{'='*70}")
    logging.info(f"Standalone Triple-Branch | {args.dataset} | mask={args.mask_ratio} | fold={fold_idx}")
    logging.info(f"  max_epochs={args.max_epochs}  ret_epochs={args.ret_epochs}")
    logging.info(f"  lr_ret={args.lr_ret}  lr_reg={args.lr_reg}  batch={args.batch_size}")
    logging.info(f"  lambda_ret={args.lambda_ret}  lambda_con={args.lambda_con}  tau_3d={args.tau_3d}")
    logging.info(f"{'='*70}")

    folds = get_folds(args.dataset)
    split = folds[fold_idx]
    label_root = os.path.join(DATA_ROOT, args.dataset)
    train_files = [os.path.join(label_root, f) for f in split["train"]]
    val_files   = [os.path.join(label_root, f) for f in split["val"]]

    predictor = TripleBranchPredictor(
        dataset=args.dataset,
        train_files=train_files,
        val_files=val_files,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mask_ratio=args.mask_ratio,
        temperature=args.contrastive_temp,
        top_k=args.top_k_default,
        tau_3d=args.tau_3d,
        ret_epochs=args.ret_epochs,
        lambda_ret=args.lambda_ret,
        lambda_con=args.lambda_con,
    )
    predictor.model.print_parameter_summary(log_fn=logging.info)

    # Full-pool training (no sampler, no sampled_set filtering)
    predictor.set_standalone_mode()
    all_indices = list(range(len(predictor.train_dataset)))
    logging.info(f"  pool size: {len(all_indices)}  val slides: {len(predictor.val_loaders)}")

    # Optimizer + scheduler
    opt = predictor.get_optimizer(args)
    sched = CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=1e-6)

    loader = DataLoader(
        predictor.train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    start = time.time()
    for epoch in range(args.max_epochs):
        predictor.set_epoch(epoch)
        predictor.model.train()
        loss_sum, n_batches = 0.0, 0
        for batch in loader:
            loss = predictor.train_step(batch, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n_batches += 1
        sched.step()

        if (epoch + 1) % 5 == 0 or epoch == args.max_epochs - 1:
            ret_active = "on" if epoch < args.ret_epochs else "off"
            logging.info(
                f"  [ep {epoch+1}/{args.max_epochs}] loss={loss_sum/max(1,n_batches):.4f}  "
                f"ret={ret_active}"
            )

        if (epoch + 1) % 10 == 0 or epoch == args.max_epochs - 1:
            rows = predictor.eval_per_slide_intermediate(
                epoch + 1,
                save_dir=os.path.join(out_dir, "intermediate_metrics"),
            )
            if rows:
                df_ep = pd.DataFrame(rows)
                summary = (df_ep.groupby('method')[['pcc', 'mse']]
                                 .mean().reset_index())
                msg = "  [eval ep {}]".format(epoch + 1)
                for _, r in summary.iterrows():
                    msg += f"  {r['method']}: PCC={r['pcc']:.4f}/MSE={r['mse']:.4f}"
                logging.info(msg)

    elapsed = (time.time() - start) / 60
    logging.info(f"Main training done in {elapsed:.1f} min")

    # Save checkpoint
    torch.save(predictor.model.state_dict(), os.path.join(out_dir, "final.pth"))

    # Fusion training + final eval
    top_ks = [int(x) for x in args.fusion_top_ks.split(",")]
    logging.info(f"{'='*60}")
    logging.info(f"FUSION: training all variants x top_ks={top_ks}")
    logging.info(f"{'='*60}")
    predictor.train_fusion_all_variants(
        n_epochs=args.fusion_epochs,
        lr=args.fusion_lr,
        top_ks=top_ks,
        selected_indices=all_indices,
    )

    df_final = predictor.eval_per_slide_all_variants(
        top_ks=top_ks,
        save_pred_dir=os.path.join(out_dir, "final_preds"),
    )
    df_final["fold"] = fold_idx
    df_final.to_csv(os.path.join(out_dir, "final_metrics_all_variants.csv"), index=False)

    # Macro summary
    macro_rows = []
    for (method, top_k), sub in df_final.groupby(['method', 'top_k']):
        macro_rows.append({
            'method': method, 'top_k': top_k,
            'pcc': float(sub['pcc'].mean()),
            'mse': float(sub['mse'].mean()),
            'mae': float(sub['mae'].mean()),
            'n_slides': int(len(sub)),
        })
    df_macro = pd.DataFrame(macro_rows).sort_values(['method', 'top_k'])
    df_macro.to_csv(os.path.join(out_dir, "final_macro_all_variants.csv"), index=False)
    logging.info("FINAL macro metrics:")
    for _, r in df_macro.iterrows():
        logging.info(f"  {r['method']:<20s}  top_k={str(r['top_k']):<4s}  "
                     f"PCC={r['pcc']:.4f}  MSE={r['mse']:.4f}")


def main():
    p = argparse.ArgumentParser()
    # Dataset / fold
    p.add_argument("--dataset", choices=["HER2"], default="HER2")
    p.add_argument("--mask_ratio", type=float, required=True)
    p.add_argument("--fold", type=int, default=None,
                   help="if None, run all folds")

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--ret_epochs", type=int, default=100)
    p.add_argument("--lr_ret", type=float, default=1e-3)
    p.add_argument("--lr_reg", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_ret", type=float, default=0.1)
    p.add_argument("--lambda_con", type=float, default=0.1)
    p.add_argument("--tau_3d", type=float, default=0.1)
    p.add_argument("--contrastive_temp", type=float, default=0.07)
    p.add_argument("--top_k_default", type=int, default=25)

    # Fusion
    p.add_argument("--fusion_epochs", type=int, default=30)
    p.add_argument("--fusion_lr", type=float, default=5e-4)
    p.add_argument("--fusion_top_ks", type=str, default="25")

    # Infra
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_dir", type=str, default="./weights")

    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    fold_list = [args.fold] if args.fold is not None else list(range(len(get_folds(args.dataset))))
    for fold_idx in fold_list:
        run_fold(args, fold_idx)


if __name__ == "__main__":
    main()
