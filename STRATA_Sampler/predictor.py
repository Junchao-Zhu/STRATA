"""ST-Net predictor wrapper used by the active-sampling agent."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models

# Optional override: point to an external ST-Net directory if needed.
ST_NET_DIR = os.environ.get("ST_NET_DIR")
if ST_NET_DIR:
    sys.path.insert(0, ST_NET_DIR)
from data_loader import GeneImputationDataset, load_samples_from_npy  # noqa: E402
from config import get_model_dims  # noqa: E402
from eval_metric import calculate_pcc  # noqa: E402


def _make_transform():
    """Single augmented transform used for both train and val."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _make_model(output_dim: int, hidden_dim: int = 1024) -> nn.Module:
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, output_dim),
    )
    return model


class STNetPredictor:
    """Predictor wrapper for active learning.

    The candidate pool defines the order of training samples. fit_* methods
    accept *global pool indices* (ints in [0, N_pool)) and use a Subset.
    """

    def __init__(
        self,
        dataset: str,
        train_files: List[str],
        val_files: List[str],
        device: str = "cuda",
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        # All training samples in the *same order* as data_pool walks them.
        # data_pool walks patients alphabetically and slides in slice_names
        # order; here we replicate exactly that order so global indices match.
        from data_pool import THREEDMAP_ROOT  # local import to avoid cycle
        train_by_patient: Dict[str, List[str]] = {}
        for f in train_files:
            patient = Path(f).parent.name
            train_by_patient.setdefault(patient, []).append(f)
        ordered_files: List[str] = []
        for patient in sorted(train_by_patient.keys()):
            npz = np.load(os.path.join(THREEDMAP_ROOT, dataset, f"{patient}.npz"), allow_pickle=True)
            slice_names = [str(s) for s in npz["slice_names"]]
            slides_in_train = sorted(
                [Path(f).stem for f in train_by_patient[patient]],
                key=lambda s: slice_names.index(s) if s in slice_names else 1e9,
            )
            for slide in slides_in_train:
                ordered_files.append(next(f for f in train_by_patient[patient] if Path(f).stem == slide))

        train_samples: List[dict] = []
        for f in ordered_files:
            train_samples.extend(load_samples_from_npy(f))

        self.train_dataset = GeneImputationDataset(
            samples=train_samples,
            dataset_name=dataset,
            transform=_make_transform(),
        )

        self.val_loaders: Dict[str, DataLoader] = {}
        all_val_samples: List[dict] = []
        for f in val_files:
            slide_name = Path(f).stem
            samples = load_samples_from_npy(f)
            all_val_samples.extend(samples)
            ds = GeneImputationDataset(
                samples=samples,
                dataset_name=dataset,
                transform=_make_transform(),
            )
            self.val_loaders[slide_name] = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )

        # Fast val subset for RL training reward signal: a fixed random
        # subsample of `rl_val_subsample` spots from the union of val slides.
        # The full per-slide loaders above are still used for `eval_per_slide`
        # so the final paper number is unchanged.
        self._rl_val_loader: DataLoader | None = None

        # build the RL-fast val subset once
        self._all_val_samples = all_val_samples

        _, output_dim = get_model_dims(dataset)
        self.output_dim = output_dim
        self.model: nn.Module = _make_model(output_dim).to(device)
        self.loss_fn = nn.MSELoss()

    # 
    # fast val subset for RL reward signal
    # 
    def setup_rl_val_subset(self, n_spots: int = 256, seed: int = 0):
        """Build a fixed small subset loader of val spots for fast val_loss."""
        if n_spots <= 0 or n_spots >= len(self._all_val_samples):
            self._rl_val_loader = None  # use the full per-slide loaders
            return
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self._all_val_samples), size=n_spots, replace=False)
        subset_samples = [self._all_val_samples[i] for i in idx]
        ds = GeneImputationDataset(
            samples=subset_samples,
            dataset_name=self.dataset,
            transform=_make_transform(),
        )
        self._rl_val_loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    # 
    # parameter management
    # 
    def snapshot(self) -> dict:
        """Return a CPU state_dict snapshot (for episode-level rollback)."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def restore(self, snap: dict):
        self.model.load_state_dict({k: v.to(self.device) for k, v in snap.items()})

    # 
    # short retrain - head-only, used for dense reward signal
    # 
    def fit_short(self, sample_indices: List[int], epochs: int = 2, lr: float = 1e-3):
        if len(sample_indices) == 0:
            return
        # freeze backbone (DenseNet features), train classifier only
        for p in self.model.features.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        subset = Subset(self.train_dataset, sample_indices)
        loader = DataLoader(
            subset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )
        self.model.train()
        for _ in range(epochs):
            for imgs, tgt in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)
                pred = self.model(imgs)
                loss = self.loss_fn(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

    # 
    # full retrain - used once per fold for final paper numbers
    # 
    def fit_full(self, sample_indices: List[int], epochs: int = 140, base_lr: float = 5e-5,
                 max_iterations: int = 100_000_000, lr_decay: float = 0.9):
        """SGD with poly LR; lr = base_lr * (1 - it/max_it)^0.9."""
        if len(sample_indices) == 0:
            return
        for p in self.model.parameters():
            p.requires_grad = True
        opt = optim.SGD(self.model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
        subset = Subset(self.train_dataset, sample_indices)
        loader = DataLoader(
            subset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )
        cur = 0
        self.model.train()
        for ep in range(epochs):
            for imgs, tgt in loader:
                lr = base_lr * (1 - cur / max_iterations) ** lr_decay
                for pg in opt.param_groups:
                    pg["lr"] = lr
                imgs = imgs.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)
                pred = self.model(imgs)
                loss = self.loss_fn(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                cur += 1
                if cur >= max_iterations:
                    return

    # 
    # evaluation
    # 
    @torch.no_grad()
    def val_loss(self) -> float:
        """Return scalar val loss. If a fast RL subset has been set up,
        evaluate only on that subset; otherwise fall back to the full
        per-slide val loaders.
        """
        self.model.eval()
        total = 0.0
        n = 0
        if self._rl_val_loader is not None:
            loaders = [self._rl_val_loader]
        else:
            loaders = list(self.val_loaders.values())
        for loader in loaders:
            for imgs, tgt in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)
                pred = self.model(imgs)
                total += self.loss_fn(pred, tgt).item() * imgs.size(0)
                n += imgs.size(0)
        self.model.train()
        return total / max(1, n)

    @torch.no_grad()
    def eval_on_subset(self, sample_indices: List[int]) -> float:
        """Evaluate model loss on a subset of the training pool (for CV reward).
        Args:
            sample_indices: global pool indices to evaluate on.
        Returns:
            scalar MSE loss on this subset.
        """
        if len(sample_indices) == 0:
            return 0.0
        self.model.eval()
        subset = Subset(self.train_dataset, sample_indices)
        loader = DataLoader(
            subset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )
        total = 0.0
        n = 0
        for imgs, tgt in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)
            pred = self.model(imgs)
            total += self.loss_fn(pred, tgt).item() * imgs.size(0)
            n += imgs.size(0)
        self.model.train()
        return total / max(1, n)

    # 
    # backbone feature caching (for online mini-episode training)
    # 
    @torch.no_grad()
    def _backbone_forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """DenseNet-121 pre-classifier features: (B, in_features=1024)."""
        feats = self.model.features(imgs)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        return feats

    @torch.no_grad()
    def cache_pool_features(self) -> tuple:
        """Run the current backbone over the full train_dataset once.

        Returns (feats, targets) where feats is (N_pool, 1024) and targets
        is (N_pool, N_TARGET_GENES), both on self.device.
        """
        self.model.eval()
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        feats_list = []
        tgts_list = []
        for imgs, tgt in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            f = self._backbone_forward(imgs)
            feats_list.append(f)
            tgts_list.append(tgt.to(self.device, non_blocking=True))
        self.model.train()
        return torch.cat(feats_list, dim=0), torch.cat(tgts_list, dim=0)

    @torch.no_grad()
    def cache_fast_val_features(self) -> tuple:
        """Cache backbone features + targets for the RL-fast val subset."""
        if self._rl_val_loader is None:
            raise RuntimeError("Call setup_rl_val_subset() first.")
        self.model.eval()
        feats_list = []
        tgts_list = []
        for imgs, tgt in self._rl_val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            f = self._backbone_forward(imgs)
            feats_list.append(f)
            tgts_list.append(tgt.to(self.device, non_blocking=True))
        self.model.train()
        return torch.cat(feats_list, dim=0), torch.cat(tgts_list, dim=0)

    # 
    # classifier-only snapshot / restore (fast, CPU-free)
    # 
    def classifier_snapshot(self) -> dict:
        return {k: v.detach().clone() for k, v in self.model.classifier.state_dict().items()}

    def classifier_restore(self, snap: dict):
        self.model.classifier.load_state_dict(snap)

    # 
    # head-only train / eval on pre-cached features
    # 
    def head_only_fit_on_features(
        self,
        feats: torch.Tensor,          # (k, 1024)
        targets: torch.Tensor,        # (k, n_target)
        epochs: int = 2,
        lr: float = 1e-3,
        batch_size: int = 196,
    ):
        if feats.shape[0] == 0:
            return
        opt = optim.Adam(self.model.classifier.parameters(), lr=lr)
        self.model.classifier.train()
        n = feats.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n, device=feats.device)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x = feats[idx]
                y = targets[idx]
                pred = self.model.classifier(x)
                loss = self.loss_fn(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

    @torch.no_grad()
    def head_only_val_loss(
        self,
        feats: torch.Tensor,      # (n_val, 1024)
        targets: torch.Tensor,    # (n_val, n_target)
    ) -> float:
        self.model.classifier.eval()
        preds = self.model.classifier(feats)
        loss = self.loss_fn(preds, targets).item()
        self.model.classifier.train()
        return float(loss)

    @torch.no_grad()
    def eval_per_slide(self) -> Dict[str, dict]:
        self.model.eval()
        out: Dict[str, dict] = {}
        for slide, loader in self.val_loaders.items():
            preds, tgts = [], []
            for imgs, tgt in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                pred = self.model(imgs).cpu()
                preds.append(pred)
                tgts.append(tgt)
            preds = torch.cat(preds)
            tgts = torch.cat(tgts)
            mse = float(((preds - tgts) ** 2).mean())
            mae = float((preds - tgts).abs().mean())
            pcc = float(calculate_pcc(preds, tgts).item())
            out[slide] = {"mse": mse, "mae": mae, "pcc": pcc, "n": int(preds.shape[0])}
        self.model.train()
        return out
