"""Build a 3D spot graph (cross-layer and intra-layer neighbors) for each case."""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

REG_ROOT = os.environ.get("REG_ROOT", "./data/spot_coords_registered")
IOU_ROOT = os.environ.get("IOU_ROOT", "./data/spot_iou")
SIM_ROOT = os.environ.get("SIM_ROOT", "./data/feature_sim")
OUT_ROOT = os.environ.get("GRAPH_ROOT", "./data/3d_map")
DATASETS = ['HER2']

#  hyperparams ----------
K_CROSS = 25
K_INTRA = 18
ALPHA   = 0.7   # weight for cosine sim in fallback score


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def load_centers(csv_path):
    """Return (N, 2) float32 centers from registered contour CSV."""
    df = pd.read_csv(csv_path)
    x = df[[f'now_x_{i}' for i in range(100)]].values.mean(axis=1)
    y = df[[f'now_y_{i}' for i in range(100)]].values.mean(axis=1)
    return np.stack([x, y], axis=1).astype(np.float32)


def get_sim_matrix(sim_data, name_a, name_b):
    """
    Return (N_a, N_b) cosine sim matrix.
    Handles key order: tries 'a_b' then 'b_a' (transposed).
    """
    key_ab = f'{name_a}_{name_b}'
    key_ba = f'{name_b}_{name_a}'
    if key_ab in sim_data:
        return sim_data[key_ab]
    elif key_ba in sim_data:
        return sim_data[key_ba].T
    else:
        raise KeyError(f'No sim data for {name_a} vs {name_b}')


def get_iou_matrix(iou_data, name_a, name_b):
    """Return (N_a, N_b) IoU matrix. Returns zeros if not found."""
    key_ab = f'{name_a}_{name_b}'
    key_ba = f'{name_b}_{name_a}'
    if key_ab in iou_data:
        return iou_data[key_ab]
    elif key_ba in iou_data:
        return iou_data[key_ba].T
    else:
        return None


def build_cross_neighbors(centers_list, slice_names, slice_offsets,
                          iou_data, sim_data, n_total):
    """Build cross-layer neighbor arrays for all spots."""
    cross_indices = np.full((n_total, K_CROSS), -1, dtype=np.int32)
    cross_weights = np.zeros((n_total, K_CROSS), dtype=np.float32)

    S = len(slice_names)
    for i in range(S):
        n_i = len(centers_list[i])
        offset_i = slice_offsets[i]

        # Collect all cross-layer candidates for slice i
        # cand_global_indices[spot_idx] = list of (global_idx, cosine_sim, dist)
        cand_sim = []
        cand_dist = []
        cand_global = []

        for j in range(S):
            if i == j:
                continue
            offset_j = slice_offsets[j]
            n_j = len(centers_list[j])

            sim_ij = get_sim_matrix(sim_data, slice_names[i], slice_names[j])  # (n_i, n_j)
            iou_ij = get_iou_matrix(iou_data, slice_names[i], slice_names[j])  # (n_i, n_j) or None

            # Center distances
            diff = centers_list[i][:, None, :] - centers_list[j][None, :, :]  # (n_i, n_j, 2)
            dist_ij = np.sqrt((diff ** 2).sum(axis=2))  # (n_i, n_j)

            cand_sim.append(sim_ij)
            cand_dist.append(dist_ij)
            cand_global.append(np.arange(n_j, dtype=np.int32) + offset_j)

            if iou_ij is None:
                # No IoU data - treat as all zeros
                iou_ij = np.zeros((n_i, cand_sim[-1].shape[1]), dtype=np.float32)
            cand_sim[-1] = (sim_ij, iou_ij)

        # Now for each spot in slice i, merge candidates from all other slices
        # Rebuild as flat arrays per spot
        for si in range(n_i):
            all_gidx = []
            all_sim_vals = []
            all_dist_vals = []
            all_iou_vals = []

            for k in range(len(cand_sim)):
                sim_ij, iou_ij = cand_sim[k]
                all_gidx.append(cand_global[k])
                all_sim_vals.append(sim_ij[si])
                all_dist_vals.append(cand_dist[k][si])
                all_iou_vals.append(iou_ij[si])

            gidx = np.concatenate(all_gidx)
            sims = np.concatenate(all_sim_vals)
            dists = np.concatenate(all_dist_vals)
            ious = np.concatenate(all_iou_vals)

            # Find IoU > 0 candidates
            iou_mask = ious > 0
            iou_idx = np.where(iou_mask)[0]

            if len(iou_idx) > 0:
                order = np.argsort(-sims[iou_idx])
                iou_chosen = iou_idx[order[:K_CROSS]]
            else:
                iou_chosen = np.array([], dtype=np.int64)

            if len(iou_chosen) >= K_CROSS:
                chosen = iou_chosen
            else:
                # Fill remaining with fallback (distance + sim) from non-IoU candidates
                need = K_CROSS - len(iou_chosen)
                exclude = set(iou_chosen.tolist())
                remain_idx = np.array([k for k in range(len(gidx)) if k not in exclude])

                dist_max = dists[remain_idx].max() if len(remain_idx) > 0 else 1.0
                if dist_max <= 0:
                    dist_max = 1.0
                prox = 1.0 - dists[remain_idx] / dist_max
                score = ALPHA * sims[remain_idx] + (1 - ALPHA) * prox
                fb_order = np.argsort(-score)[:need]
                fb_chosen = remain_idx[fb_order]

                chosen = np.concatenate([iou_chosen, fb_chosen])

            k_actual = len(chosen)
            global_idx = offset_i + si
            cross_indices[global_idx, :k_actual] = gidx[chosen]
            cross_weights[global_idx, :k_actual] = sims[chosen]

    return cross_indices, cross_weights


def build_intra_neighbors(centers_list, slice_offsets, n_total):
    """Build intra-layer neighbor arrays for all spots."""
    intra_indices = np.full((n_total, K_INTRA), -1, dtype=np.int32)
    intra_weights = np.zeros((n_total, K_INTRA), dtype=np.float32)

    for i, centers in enumerate(centers_list):
        n_i = len(centers)
        offset = slice_offsets[i]

        if n_i <= 1:
            continue

        tree = cKDTree(centers)
        # k+1 because query includes the point itself
        k_query = min(K_INTRA + 1, n_i)
        dists, idx = tree.query(centers, k=k_query)

        # Remove self (first column)
        dists = dists[:, 1:]   # (n_i, k_query-1)
        idx = idx[:, 1:]

        k_actual = dists.shape[1]
        intra_indices[offset:offset + n_i, :k_actual] = idx + offset  # global index
        intra_weights[offset:offset + n_i, :k_actual] = dists

    return intra_indices, intra_weights


def process_case(dataset, case):
    """Build and save spot graph for one case."""
    reg_dir = os.path.join(REG_ROOT, dataset, case)
    csvs = sorted(glob.glob(os.path.join(reg_dir, '*.csv')), key=natural_keys)
    slice_names = [os.path.basename(f).replace('.csv', '') for f in csvs]

    # Load centers
    centers_list = [load_centers(f) for f in csvs]
    counts = [len(c) for c in centers_list]
    slice_offsets = np.array([sum(counts[:i]) for i in range(len(counts))], dtype=np.int32)
    n_total = sum(counts)

    # Load precomputed matrices
    iou_path = os.path.join(IOU_ROOT, dataset, case + '.npz')
    sim_path = os.path.join(SIM_ROOT, dataset, case + '.npz')
    iou_data = np.load(iou_path) if os.path.exists(iou_path) else {}
    sim_data = np.load(sim_path)

    # Build neighbors
    cross_indices, cross_weights = build_cross_neighbors(
        centers_list, slice_names, slice_offsets, iou_data, sim_data, n_total)
    intra_indices, intra_weights = build_intra_neighbors(
        centers_list, slice_offsets, n_total)

    # Save
    out_dir = os.path.join(OUT_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, case + '.npz'),
             slice_names=np.array(slice_names),
             slice_offsets=slice_offsets,
             cross_indices=cross_indices,
             cross_weights=cross_weights,
             intra_indices=intra_indices,
             intra_weights=intra_weights)


if __name__ == '__main__':
    for dataset in DATASETS:
        reg_dir = os.path.join(REG_ROOT, dataset)
        cases = sorted(os.listdir(reg_dir))
        for case in tqdm(cases, desc=dataset):
            process_case(dataset, case)

    print('Done.')
