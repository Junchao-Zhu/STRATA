"""Episode environment for active sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from data_pool import CandidatePool
from predictor import STNetPredictor
from sc_bank import SCBank


# 
# State feature layout (per candidate spot):
#   [ vision(512)              0:512
#     xyz(4)                   512:516
#     test_proximity(1)        516
#     same_patient_flag(1)     517
#     is_sampled(1)            518
#     budget_left(1)           519
#     own_cell_type(C) | mask  520:520+C
#     own_mol_embed(512)|mask  520+C:520+C+512
#     observed_flag(1)         end
#   ]
# 


def state_feature_dim(cell_type_dim: int) -> int:
    return 512 + 4 + 1 + 1 + 1 + 1 + cell_type_dim + 512 + 1


@dataclass
class EnvConfig:
    budget: int                     # B spots per episode
    retrain_every: int = 4          # k: trigger predictor.fit_short every k steps
    fit_short_epochs: int = 2
    alpha_sc: float = 0.1           # weight of SC coverage shaping
    beta_redun: float = 0.1         # weight of cross-redundancy penalty
    gamma_patient: float = 0.05     # weight of patient-repetition penalty
    device: str = "cuda"


class ActiveSamplingEnv:
    def __init__(self, pool: CandidatePool, predictor: STNetPredictor,
                 sc_bank: Optional[SCBank], cfg: EnvConfig):
        self.pool = pool
        self.predictor = predictor
        self.sc_bank = sc_bank
        self.cfg = cfg

        self.N = pool.n_pool
        self.cell_type_dim = pool.cell_type.shape[1]
        self.feat_dim = state_feature_dim(self.cell_type_dim)

        # snapshot the predictor weights so we can rollback at episode reset
        self._predictor_snapshot = predictor.snapshot()

        # frozen feature block (vision + xyz + test_prox + same_patient_flag)
        self._frozen = np.concatenate(
            [
                pool.vision,                              # (N, 512)
                pool.xyz,                                 # (N, 4)
                pool.test_proximity[:, None],             # (N, 1)
                pool.same_patient_flag[:, None],          # (N, 1)
            ],
            axis=1,
        ).astype(np.float32)

        self.reset()

    # 
    def reset(self) -> torch.Tensor:
        self.S: list[int] = []
        self.S_set: set[int] = set()
        self.is_sampled = np.zeros(self.N, dtype=np.float32)
        self.observed_flag = np.zeros(self.N, dtype=np.float32)
        self.own_cell_type = np.zeros((self.N, self.cell_type_dim), dtype=np.float32)
        self.own_mol_embed = np.zeros((self.N, 512), dtype=np.float32)

        self.predictor.restore(self._predictor_snapshot)
        if self.sc_bank is not None:
            self.sc_bank.reset()

        # patient-pick counts for diversity penalty
        self.patient_counts: dict[str, int] = {}

        # initial val loss (after warm-start) - needed for first window's R_true
        self._L_old = self.predictor.val_loss()

        # window buffer: per-step features for reward-model supervision
        self._window_feats: list[np.ndarray] = []

        return self._build_state_tensor()

    # 
    def _build_state_tensor(self) -> torch.Tensor:
        N = self.N
        budget_left = max(0, self.cfg.budget - len(self.S))
        budget_vec = np.full((N, 1), budget_left / self.cfg.budget, dtype=np.float32)
        is_sampled = self.is_sampled[:, None]
        obs_flag = self.observed_flag[:, None]
        feat = np.concatenate(
            [
                self._frozen,            # (N, 518)
                is_sampled,              # (N, 1)
                budget_vec,              # (N, 1)
                self.own_cell_type,      # (N, C)
                self.own_mol_embed,      # (N, 512)
                obs_flag,                # (N, 1)
            ],
            axis=1,
        )
        return torch.from_numpy(feat)

    # 
    def candidate_mask(self) -> torch.Tensor:
        return torch.from_numpy(self.is_sampled == 0).bool()

    def selected_mask(self) -> torch.Tensor:
        return torch.from_numpy(self.is_sampled == 1).bool()

    # 
    def _redundancy_penalty(self, action: int) -> float:
        """Sum of cross_weights to already-selected spots within the same patient."""
        patient = self.pool.patient_id[action]
        if patient not in self.pool.cross:
            return 0.0
        cross_idx, cross_w = self.pool.cross[patient]
        # convert pool global -> in-patient global
        pool_idx_arr = self.pool.pool_idx_in_patient[patient]
        in_patient_arr = self.pool.in_patient_global[patient]
        # build a map from pool global to in-patient global on the fly (small per-patient)
        pool_to_inpatient = dict(zip(pool_idx_arr.tolist(), in_patient_arr.tolist()))
        if action not in pool_to_inpatient:
            return 0.0
        ip = pool_to_inpatient[action]
        # set of already-sampled in-patient ids of the same patient
        same_patient_sampled_ip = set()
        for s in self.S:
            if self.pool.patient_id[s] == patient and s in pool_to_inpatient and s != action:
                same_patient_sampled_ip.add(pool_to_inpatient[s])
        if not same_patient_sampled_ip:
            return 0.0
        row_idx = cross_idx[ip]
        row_w = cross_w[ip]
        mask = np.array([j in same_patient_sampled_ip for j in row_idx])
        return float(row_w[mask].sum()) if mask.any() else 0.0

    # 
    def step(self, action: int):
        """Perform one pick. Returns (next_state, dense_reward_components, info).

        dense_reward_components is a dict so the caller can blend them with
        the learned reward-model output. The env does NOT compute the
        learned reward; that's the caller's job.
        """
        assert self.is_sampled[action] == 0, f"action {action} already sampled"

        self.S.append(action)
        self.S_set.add(action)
        self.is_sampled[action] = 1.0
        self.observed_flag[action] = 1.0
        self.own_cell_type[action] = self.pool.cell_type[action]
        self.own_mol_embed[action] = self.pool.mol_embed[action]

        patient = self.pool.patient_id[action]
        self.patient_counts[patient] = self.patient_counts.get(patient, 0) + 1

        # shaping rewards (cheap, every step)
        r_redun = self._redundancy_penalty(action)
        r_sc = self.sc_bank.update(self.pool.mol_embed[action]) if self.sc_bank is not None else 0.0
        r_patient = float(np.log1p(self.patient_counts[patient] - 1))  # 0 on first pick

        info = {
            "r_redun": r_redun,
            "r_sc": r_sc,
            "r_patient": r_patient,
            "true_reward": None,
            "step": len(self.S),
        }

        # trigger short retrain -> true reward
        if len(self.S) % self.cfg.retrain_every == 0:
            self.predictor.fit_short(self.S, epochs=self.cfg.fit_short_epochs)
            L_new = self.predictor.val_loss()
            R_true = self._L_old - L_new
            self._L_old = L_new
            info["true_reward"] = R_true

        next_state = self._build_state_tensor()
        done = len(self.S) >= self.cfg.budget
        info["done"] = done
        return next_state, info

    # 
    def shaping_scalar(self, info: dict) -> float:
        c = self.cfg
        return c.alpha_sc * info["r_sc"] - c.beta_redun * info["r_redun"] - c.gamma_patient * info["r_patient"]
