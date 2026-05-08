"""
Extract spot coordinates from h5ad files.
Output structure mirrors wsi_png: spot_coords/{dataset}/{patient}/{slide_id}.csv
CSV format: spot_id (spot_0, spot_1, ...), x, y
"""
import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

ST_DIR = os.environ.get("ST_H5AD_ROOT", "./data/st_h5ad")
WSI_PNG_BASE = os.environ.get("WSI_ROOT", "./data/wsi_png")
OUTPUT_BASE = os.environ.get("SPOT_COORDS_ROOT", "./data/spot_coords")


def extract_coords(slide_id, output_path):
    h5ad_path = os.path.join(ST_DIR, f'{slide_id}.h5ad')
    if not os.path.exists(h5ad_path):
        tqdm.write(f'  [SKIP] {h5ad_path} not found')
        return 0

    adata = sc.read_h5ad(h5ad_path)
    coords = adata.obsm['spatial']  # (N, 2): col0=x (pxl_col), col1=y (pxl_row)
    n = len(coords)

    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
    }, index=[f'spot_{i}' for i in range(n)])
    df.index.name = 'spot_id'

    df.to_csv(output_path)
    return n


if __name__ == '__main__':
    datasets = ['HER2']

    for dataset in datasets:
        patient_dirs = sorted(glob.glob(os.path.join(WSI_PNG_BASE, dataset, '*')))
        for patient_dir in tqdm(patient_dirs, desc=dataset):
            patient = os.path.basename(patient_dir)
            out_dir = os.path.join(OUTPUT_BASE, dataset, patient)
            os.makedirs(out_dir, exist_ok=True)

            slide_pngs = sorted(glob.glob(os.path.join(patient_dir, '*.png')))
            for png_path in slide_pngs:
                slide_id = os.path.basename(png_path).replace('.png', '')
                out_path = os.path.join(out_dir, f'{slide_id}.csv')
                n = extract_coords(slide_id, out_path)
                if n > 0:
                    tqdm.write(f'  {dataset}/{patient}/{slide_id}: {n} spots')
