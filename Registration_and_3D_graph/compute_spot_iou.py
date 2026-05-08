"""Compute pairwise circle IoU between spots across slices within each case."""
import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
from itertools import combinations
from tqdm import tqdm

REGISTERED_ROOT = os.environ.get("REG_ROOT", "./data/spot_coords_registered")
OUTPUT_ROOT     = os.environ.get("IOU_ROOT", "./data/spot_iou")
DIST_THRESHOLD  = 1000.0
DATASETS        = ['HER2']


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def load_contours(csv_path):
    """
    Returns:
      centers:  (N, 2) float32 - mean of 100 contour points
      contours: (N, 100, 2) float32 - raw contour points for exact IoU
    """
    df = pd.read_csv(csv_path)
    num_points = 100
    x = df[[f'now_x_{i}' for i in range(num_points)]].values.astype(np.float32)
    y = df[[f'now_y_{i}' for i in range(num_points)]].values.astype(np.float32)
    contours = np.stack([x, y], axis=2)          # (N, 100, 2)
    centers  = contours.mean(axis=1)              # (N, 2)
    return centers, contours


def polygon_iou(contour_a, contour_b):
    """Exact IoU between two convex polygons via cv2."""
    pts_a = cv2.convexHull(contour_a.reshape(-1, 1, 2).astype(np.float32))
    pts_b = cv2.convexHull(contour_b.reshape(-1, 1, 2).astype(np.float32))
    inter, _ = cv2.intersectConvexConvex(pts_a, pts_b)
    if inter == 0:
        return 0.0
    area_a = cv2.contourArea(pts_a)
    area_b = cv2.contourArea(pts_b)
    return float(inter / (area_a + area_b - inter))


def compute_iou_matrix(centers_i, contours_i, centers_j, contours_j,
                       dist_threshold=DIST_THRESHOLD, chunk_size=500):
    """
    (N, M) IoU matrix.
    Step 1: vectorized distance filter -> candidate mask
    Step 2: exact polygon IoU for candidates only
    """
    N, M = len(centers_i), len(centers_j)
    iou = np.zeros((N, M), dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        ci = centers_i[start:end]                          # (chunk, 2)
        diff = ci[:, None, :] - centers_j[None, :, :]     # (chunk, M, 2)
        d = np.sqrt((diff * diff).sum(axis=2))             # (chunk, M)
        rows, cols = np.where(d < dist_threshold)

        for r, c in zip(rows, cols):
            iou_val = polygon_iou(contours_i[start + r], contours_j[c])
            if iou_val > 0:
                iou[start + r, c] = iou_val

    return iou


if __name__ == '__main__':
    for dataset in DATASETS:
        reg_dataset_dir = os.path.join(REGISTERED_ROOT, dataset)
        cases = sorted(os.listdir(reg_dataset_dir))

        for case in tqdm(cases, desc=dataset):
            out_path = os.path.join(OUTPUT_ROOT, dataset, case + '.npz')
            # if os.path.exists(out_path):
            #     continue

            case_dir = os.path.join(reg_dataset_dir, case)
            csvs = sorted(glob.glob(os.path.join(case_dir, '*.csv')),
                          key=natural_keys)
            slice_names = [os.path.basename(f).replace('.csv', '') for f in csvs]

            all_centers  = []
            all_contours = []
            for f in csvs:
                centers, contours = load_contours(f)
                all_centers.append(centers)
                all_contours.append(contours)

            result = {}
            for i, j in combinations(range(len(slice_names)), 2):
                name_i, name_j = slice_names[i], slice_names[j]
                iou = compute_iou_matrix(
                    all_centers[i], all_contours[i],
                    all_centers[j], all_contours[j],
                )
                result[f'{name_i}_{name_j}'] = iou

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez(out_path, **result)

    print('Done.')
