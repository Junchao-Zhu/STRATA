"""Build the candidate-pool data structure for active sampling."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple  # noqa: F401

import numpy as np


THREEDMAP_ROOT = os.environ.get("GRAPH_ROOT", "./data/3d_map")
CONCH_ROOT = os.environ.get(
    "CONCH_ROOT", os.path.join(os.environ.get("DATA_ROOT", "./data/data_infor"), "conch_features")
)
SC_PROTO_ROOT = os.environ.get("SC_PROTO_ROOT", "./sc_prototypes")


@dataclass
class CandidatePool:
    # frozen, always visible
    vision: np.ndarray           # (N, 512)  CONCH H&E feature (pre-sampling)
    xyz: np.ndarray              # (N, 4)    [x, y, slice_idx_in_patient, patient_id_int]
    test_proximity: np.ndarray   # (N,)      3D-graph weight to val spots
    same_patient_flag: np.ndarray  # (N,)    1 if the spot's patient has val slides

    # pre-sampling 3D neighborhood aggregation (v3): static summaries of each
    # spot's 3D neighbors' CONCH features, built once per fold.
    intra_agg: np.ndarray        # (N, 512)  mean(CONCH of intra-slice neighbors in pool)
    cross_agg: np.ndarray        # (N, 512)  mean(CONCH of cross-slice neighbors in pool)

    # revealable (filled from the npy fields, but only used after sampling)
    cell_type: np.ndarray        # (N, C_max) float32 (zero-padded across datasets)
    mol_embed: np.ndarray        # (N, 512) float32   (== sc_embed, post-sampling)
    expression: np.ndarray       # (N, n_genes_total) float32, used by predictor

    # external SC atlas (v3): k-means centroids from /scgpt/sc_embedding
    sc_centroids: np.ndarray     # (K, 512)  SC prototype centers
    sc_freq: np.ndarray          # (K,)      marginal frequency in SC bank

    # bookkeeping
    sample_id: List[str]         # length N, slide name per spot
    patch_id: List[str]          # length N, patch name per spot
    patient_id: List[str]        # length N, patient name per spot
    pool_idx_in_patient: Dict[str, np.ndarray]  # patient -> (n_pool_spots_of_patient,) array of pool global idx
    in_patient_global: Dict[str, np.ndarray]    # patient -> (same length) the in-patient global idx in 3d_map

    # 3d adjacency, per patient, in-patient indices
    cross: Dict[str, Tuple[np.ndarray, np.ndarray]]  # patient -> (cross_idx (Np,25), cross_w (Np,25))
    intra: Dict[str, Tuple[np.ndarray, np.ndarray]]  # patient -> (intra_idx (Np,18), intra_w (Np,18))

    # candidate "is in candidate pool" mask in the patient global frame
    pool_mask_in_patient: Dict[str, np.ndarray]  # patient -> (Np,) bool

    @property
    def n_pool(self) -> int:
        return self.vision.shape[0]

    @property
    def n_sc_clusters(self) -> int:
        return int(self.sc_centroids.shape[0])


def _load_slide_npy(path: Path) -> List[dict]:
    return list(np.load(path, allow_pickle=True))


def _detect_max_celltype_dim(samples: List[dict]) -> int:
    for s in samples:
        if "cell_type_counts" in s and s["cell_type_counts"] is not None:
            return len(s["cell_type_counts"])
    return 0


def build_candidate_pool(
    dataset: str,
    train_files: List[str],   # absolute paths from cross_fold (already include anchor middles)
    val_files: List[str],     # absolute paths to the held-out slides (used for test_proximity only)
    data_root: str = os.environ.get("DATA_ROOT", "./data/data_infor"),
) -> CandidatePool:
    """Build the per-fold candidate pool.

    `train_files` is the agent's action universe (every spot it may pick).
    `val_files` is *not* in the pool but its spot positions are used to
    precompute `test_proximity` for every candidate spot.
    """
    #  1. Group train files by patient, in slice_names order matching 3d_map
    train_by_patient: Dict[str, List[str]] = {}
    val_by_patient: Dict[str, List[str]] = {}
    for f in train_files:
        patient = Path(f).parent.name
        train_by_patient.setdefault(patient, []).append(f)
    for f in val_files:
        patient = Path(f).parent.name
        val_by_patient.setdefault(patient, []).append(f)

    #  2. Load 3d_map for each patient that contributes any spot
    npz_cache: Dict[str, dict] = {}
    for patient in set(list(train_by_patient.keys()) + list(val_by_patient.keys())):
        npz_path = os.path.join(THREEDMAP_ROOT, dataset, f"{patient}.npz")
        with np.load(npz_path, allow_pickle=True) as z:
            npz_cache[patient] = {
                "slice_names": [str(s) for s in z["slice_names"]],
                "slice_offsets": z["slice_offsets"].astype(np.int64),
                "cross_indices": z["cross_indices"].astype(np.int64),
                "cross_weights": z["cross_weights"].astype(np.float32),
                "intra_indices": z["intra_indices"].astype(np.int64),
                "intra_weights": z["intra_weights"].astype(np.float32),
            }

    # helper: given patient + slide name -> (start, end) in-patient global indices
    def _slide_range(patient: str, slide: str) -> Tuple[int, int]:
        meta = npz_cache[patient]
        names = meta["slice_names"]
        offs = list(meta["slice_offsets"])
        if slide not in names:
            raise KeyError(f"slide {slide} not in 3d_map for patient {patient}")
        i = names.index(slide)
        start = int(offs[i])
        end = int(offs[i + 1]) if i + 1 < len(offs) else int(meta["cross_indices"].shape[0])
        return start, end

    #  3. Walk train files in patient x slice_names order and collect spots
    pool_records: List[dict] = []
    pool_idx_in_patient: Dict[str, List[int]] = {}      # pool global idx
    in_patient_global: Dict[str, List[int]] = {}        # in-patient global idx
    val_in_patient_global: Dict[str, List[int]] = {}    # for test_proximity

    # collect val in-patient global indices first (for proximity)
    for patient, files in val_by_patient.items():
        if patient not in npz_cache:
            continue
        gids: List[int] = []
        for f in files:
            slide = Path(f).stem
            s, e = _slide_range(patient, slide)
            gids.extend(range(s, e))
        val_in_patient_global[patient] = sorted(gids)

    # walk train files
    for patient in sorted(train_by_patient.keys()):
        files = sorted(train_by_patient[patient])
        meta = npz_cache[patient]
        names = meta["slice_names"]
        # process slides in slice_names order to preserve in-patient global indexing
        slides_of_patient_in_train = sorted(
            [Path(f).stem for f in files], key=lambda s: names.index(s) if s in names else 1e9
        )
        for slide in slides_of_patient_in_train:
            slide_path = next(f for f in files if Path(f).stem == slide)
            samples = _load_slide_npy(Path(slide_path))
            s_start, s_end = _slide_range(patient, slide)
            assert s_end - s_start == len(samples), (
                f"3d_map vs npy length mismatch: {patient}/{slide} "
                f"npy={len(samples)} 3d_map={s_end - s_start}"
            )
            for local_i, sample in enumerate(samples):
                pool_global = len(pool_records)
                pool_records.append({
                    "patient": patient,
                    "slide": slide,
                    "local_i": local_i,
                    "in_patient_global": s_start + local_i,
                    "sample": sample,
                })
                pool_idx_in_patient.setdefault(patient, []).append(pool_global)
                in_patient_global.setdefault(patient, []).append(s_start + local_i)

    N = len(pool_records)
    if N == 0:
        raise RuntimeError("Candidate pool is empty")

    #  4. Build feature matrices
    # detect max cell-type dim across all samples (for stable concat)
    max_ct = max(
        (len(r["sample"]["cell_type_counts"])
         for r in pool_records
         if "cell_type_counts" in r["sample"] and r["sample"]["cell_type_counts"] is not None),
        default=0,
    )
    # n_genes from first sample
    n_genes = len(pool_records[0]["sample"]["expression_norm"])

    vision = np.zeros((N, 512), dtype=np.float32)
    cell_type = np.zeros((N, max(max_ct, 1)), dtype=np.float32)
    mol_embed = np.zeros((N, 512), dtype=np.float32)
    expression = np.zeros((N, n_genes), dtype=np.float32)
    xyz = np.zeros((N, 4), dtype=np.float32)
    sample_id_list: List[str] = []
    patch_id_list: List[str] = []
    patient_id_list: List[str] = []

    patient_to_int = {p: i for i, p in enumerate(sorted(set(r["patient"] for r in pool_records)))}

    #  load CONCH per-slide feature caches (see extract_conch.py).
    # `vision` is the ONLY pre-sampling image feature we use; sc_embed is
    # a scGPT embedding of expression and leaks - do NOT load it into state.
    # mol_embed / cell_type remain post-sampling revelations.
    conch_cache: Dict[Tuple[str, str], np.ndarray] = {}
    missing_conch: List[str] = []
    for r in pool_records:
        key = (r["patient"], r["slide"])
        if key in conch_cache:
            continue
        cpath = os.path.join(CONCH_ROOT, dataset, r["patient"], f"{r['slide']}_conch.npy")
        if not os.path.exists(cpath):
            missing_conch.append(cpath)
            continue
        conch_cache[key] = np.load(cpath).astype(np.float32)
    if missing_conch:
        raise FileNotFoundError(
            f"Missing CONCH features for {len(missing_conch)} slides. "
            f"Run extract_conch.py first. e.g. {missing_conch[0]}"
        )

    for pi, r in enumerate(pool_records):
        s = r["sample"]
        # vision: H&E -> CONCH ViT-B-16 contrast head (pre-sampling, no leak)
        key = (r["patient"], r["slide"])
        vision[pi] = conch_cache[key][r["local_i"]]
        # post-sampling revelations (zeroed here; filled in dyn_state on reveal)
        if "sc_embed" in s and s["sc_embed"] is not None:
            # kept for possible use in reveal() as "observed mol embed"
            mol_embed[pi] = s["sc_embed"]
        if "cell_type_counts" in s and s["cell_type_counts"] is not None:
            ct = np.asarray(s["cell_type_counts"], dtype=np.float32)
            cell_type[pi, : len(ct)] = ct
        expression[pi] = np.asarray(s["expression_norm"], dtype=np.float32)
        pos = s.get("position", (0, 0))
        slice_idx_in_patient = npz_cache[r["patient"]]["slice_names"].index(r["slide"])
        xyz[pi] = (float(pos[0]), float(pos[1]), float(slice_idx_in_patient), float(patient_to_int[r["patient"]]))
        sample_id_list.append(s.get("sample_id", r["slide"]))
        patch_id_list.append(s.get("patch_id", f"patch_{r['local_i']}"))
        patient_id_list.append(r["patient"])

    #  5. Test-proximity: for each candidate, sum of cross_weights to val
    # spots of the same patient (using in-patient indices from the npz).
    test_proximity = np.zeros(N, dtype=np.float32)
    same_patient_flag = np.zeros(N, dtype=np.float32)
    for patient, val_gids in val_in_patient_global.items():
        if patient not in pool_idx_in_patient:
            continue  # patient has only val files (no spots in pool - shouldn't happen because anchor middle is in train)
        meta = npz_cache[patient]
        cross_idx = meta["cross_indices"]
        cross_w = meta["cross_weights"]
        val_set = set(val_gids)
        for pool_g, ip_g in zip(pool_idx_in_patient[patient], in_patient_global[patient]):
            same_patient_flag[pool_g] = 1.0
            row_idx = cross_idx[ip_g]
            row_w = cross_w[ip_g]
            mask = np.array([j in val_set for j in row_idx])
            if mask.any():
                test_proximity[pool_g] = float(row_w[mask].sum())

    #  6. Convert dict-of-list -> dict-of-array, and store the patient sub-3d_map
    pool_idx_arr = {p: np.asarray(v, dtype=np.int64) for p, v in pool_idx_in_patient.items()}
    in_patient_arr = {p: np.asarray(v, dtype=np.int64) for p, v in in_patient_global.items()}

    cross_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    intra_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    pool_mask_in_patient: Dict[str, np.ndarray] = {}
    for patient, meta in npz_cache.items():
        Np = meta["cross_indices"].shape[0]
        cross_dict[patient] = (meta["cross_indices"], meta["cross_weights"])
        intra_dict[patient] = (meta["intra_indices"], meta["intra_weights"])
        m = np.zeros(Np, dtype=bool)
        if patient in in_patient_arr:
            m[in_patient_arr[patient]] = True
        pool_mask_in_patient[patient] = m

    #  7. Pre-sampling 3D neighborhood aggregates (v3)
    #  For each pool spot, average the CONCH features of its intra-slice
    #  and cross-slice neighbors that are ALSO in the pool. Uses only H&E
    #  (frozen CONCH) and the pre-known 3D graph - no expression touched.
    intra_agg = np.zeros((N, 512), dtype=np.float32)
    cross_agg = np.zeros((N, 512), dtype=np.float32)

    # inverse map: (patient, in_patient_idx) -> pool_global_idx
    ip_to_pool: Dict[Tuple[str, int], int] = {}
    for patient, pool_arr in pool_idx_arr.items():
        ip_arr = in_patient_arr[patient]
        for pg, ipg in zip(pool_arr, ip_arr):
            ip_to_pool[(patient, int(ipg))] = int(pg)

    for patient in pool_idx_arr:
        cross_idx_p, cross_w_p = cross_dict[patient]
        intra_idx_p, intra_w_p = intra_dict[patient]
        pool_of_patient = pool_idx_arr[patient]
        in_patient_of_patient = in_patient_arr[patient]
        for pg, ipg in zip(pool_of_patient, in_patient_of_patient):
            # intra
            nbs = intra_idx_p[int(ipg)]
            pool_nb_vision: List[np.ndarray] = []
            for nb in nbs:
                key = (patient, int(nb))
                if key in ip_to_pool:
                    pool_nb_vision.append(vision[ip_to_pool[key]])
            if pool_nb_vision:
                intra_agg[int(pg)] = np.mean(pool_nb_vision, axis=0)
            # cross
            nbs = cross_idx_p[int(ipg)]
            pool_nb_vision = []
            for nb in nbs:
                key = (patient, int(nb))
                if key in ip_to_pool:
                    pool_nb_vision.append(vision[ip_to_pool[key]])
            if pool_nb_vision:
                cross_agg[int(pg)] = np.mean(pool_nb_vision, axis=0)

    #  8. SC prototypes (external single-cell atlas, k-means centroids)
    sc_npz_path = os.path.join(SC_PROTO_ROOT, f"{dataset}.npz")
    if os.path.exists(sc_npz_path):
        sc_npz = np.load(sc_npz_path)
        sc_centroids = sc_npz["centroids"].astype(np.float32)
        sc_freq = sc_npz["freq"].astype(np.float32)
    else:
        # graceful degrade: downstream code can check K=0 and skip SC novelty
        sc_centroids = np.zeros((0, 512), dtype=np.float32)
        sc_freq = np.zeros((0,), dtype=np.float32)

    return CandidatePool(
        vision=vision,
        xyz=xyz,
        test_proximity=test_proximity,
        same_patient_flag=same_patient_flag,
        intra_agg=intra_agg,
        cross_agg=cross_agg,
        cell_type=cell_type,
        mol_embed=mol_embed,
        expression=expression,
        sc_centroids=sc_centroids,
        sc_freq=sc_freq,
        sample_id=sample_id_list,
        patch_id=patch_id_list,
        patient_id=patient_id_list,
        pool_idx_in_patient=pool_idx_arr,
        in_patient_global=in_patient_arr,
        cross=cross_dict,
        intra=intra_dict,
        pool_mask_in_patient=pool_mask_in_patient,
    )
