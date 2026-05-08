"""Evaluation metrics for gene expression prediction."""
import torch
import numpy as np


def calculate_pcc(pred, target):
    target = target.float()
    pred = pred.float()
    x = pred - pred.mean(dim=0, keepdim=True)
    y = target - target.mean(dim=0, keepdim=True)
    covariance = (x * y).sum(dim=0)
    variance_x = (x ** 2).sum(dim=0)
    variance_y = (y ** 2).sum(dim=0)
    pcc = covariance / torch.sqrt(variance_x * variance_y + 1e-8)
    return pcc.mean()


def compute_metrics(pred, target):
    if isinstance(pred, torch.Tensor):
        pred_np = pred.numpy()
        target_np = target.numpy()
        pred_t = pred
        target_t = target
    else:
        pred_np = pred
        target_np = target
        pred_t = torch.tensor(pred)
        target_t = torch.tensor(target)

    mse = np.mean((pred_np - target_np) ** 2)
    mae = np.mean(np.abs(pred_np - target_np))
    pcc = calculate_pcc(pred_t, target_t).item()

    return {'mse': mse, 'mae': mae, 'pcc': pcc}
