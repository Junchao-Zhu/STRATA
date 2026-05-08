"""State builder: per-spot frozen, dynamic, and post-sampling feature blocks."""
from __future__ import annotations

import numpy as np
import torch

from data_pool import CandidatePool


def frozen_block_dim() -> int:
    return 512 + 512 + 512 + 4 + 1 + 1  # 1542


def dynamic_block_dim(cell_type_dim: int, n_sc_clusters: int) -> int:
    # cell_prior(K) + is_sampled + budget + cross_red + own_cell(C) + own_mol(512) + observed
    return n_sc_clusters + 1 + 1 + 1 + cell_type_dim + 512 + 1


def state_feature_dim(cell_type_dim: int, n_sc_clusters: int) -> int:
    return frozen_block_dim() + dynamic_block_dim(cell_type_dim, n_sc_clusters)


def build_frozen_block(pool: CandidatePool) -> np.ndarray:
    """Concatenate the never-changing, pre-sampling part of the state."""
    return np.concatenate(
        [
            pool.vision,                            # (N, 512)
            pool.intra_agg,                         # (N, 512)
            pool.cross_agg,                         # (N, 512)
            pool.xyz,                               # (N, 4)
            pool.test_proximity[:, None],           # (N, 1)
            pool.same_patient_flag[:, None],        # (N, 1)
        ],
        axis=1,
    ).astype(np.float32)


def build_state_tensor(
    frozen: np.ndarray,             # (N, 1542)
    is_sampled: np.ndarray,         # (N,) float32
    observed_flag: np.ndarray,      # (N,) float32
    own_cell_type: np.ndarray,      # (N, C) float32
    own_mol_embed: np.ndarray,      # (N, 512) float32
    budget_left: int,
    budget_total: int,
    cross_red: np.ndarray,          # (N,) float32
    cell_prior: np.ndarray,         # (N, K) float32, softmax output of CellHead
) -> torch.Tensor:
    N = frozen.shape[0]
    budget_vec = np.full((N, 1), budget_left / max(1, budget_total), dtype=np.float32)
    cr_max = float(cross_red.max()) if cross_red.size > 0 else 0.0
    cr_scale = max(cr_max, 1e-3)
    cr_norm = (cross_red / cr_scale).astype(np.float32)[:, None]
    feat = np.concatenate(
        [
            frozen,
            cell_prior.astype(np.float32),          # (N, K)
            is_sampled[:, None],
            budget_vec,
            cr_norm,
            own_cell_type,                          # (N, C)
            own_mol_embed,                          # (N, 512)
            observed_flag[:, None],
        ],
        axis=1,
    )
    return torch.from_numpy(feat)


def init_dynamic_state(n_pool: int, cell_type_dim: int):
    return {
        "is_sampled": np.zeros(n_pool, dtype=np.float32),
        "observed_flag": np.zeros(n_pool, dtype=np.float32),
        "own_cell_type": np.zeros((n_pool, cell_type_dim), dtype=np.float32),
        "own_mol_embed": np.zeros((n_pool, 512), dtype=np.float32),
    }


def reveal(dyn: dict, pool: CandidatePool, action: int):
    dyn["is_sampled"][action] = 1.0
    dyn["observed_flag"][action] = 1.0
    dyn["own_cell_type"][action] = pool.cell_type[action]
    dyn["own_mol_embed"][action] = pool.mol_embed[action]
