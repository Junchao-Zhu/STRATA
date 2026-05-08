import os
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def merge_expression_data(our_format_dir, hvg_expr_dir, output_dir, dataset_name, sample_ids):
    """

    Args:
    """
    dataset_out_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    success_count = 0
    skip_count = 0

    for sample_id in sample_ids:
        our_format_file = os.path.join(our_format_dir, dataset_name, f"{sample_id}.npy")
        hvg_expr_file = os.path.join(hvg_expr_dir, f"{sample_id}.npy")

        if not os.path.exists(our_format_file):
            logging.warning(f" our_format file not found: {sample_id}.npy")
            skip_count += 1
            continue

        if not os.path.exists(hvg_expr_file):
            logging.warning(f" hvg_expr file not found: {sample_id}_hvg_expr.npy")
            skip_count += 1
            continue

        try:
            our_format_data = np.load(our_format_file, allow_pickle=True)
            hvg_expr_data = np.load(hvg_expr_file, allow_pickle=True).item()

            n_spots_our = len(our_format_data)
            n_spots_expr = hvg_expr_data['expression'].shape[0]

            if n_spots_our != n_spots_expr:
                logging.error(
                    f"our_format={n_spots_our}, expression={n_spots_expr}"
                )
                skip_count += 1
                continue

            merged_data = []
            for i, spot_dict in enumerate(our_format_data):
                new_dict = dict(spot_dict)
                new_dict['expression_norm'] = hvg_expr_data['expression'][i]
                # new_dict['hvg_genes'] = hvg_expr_data['gene_names']
                merged_data.append(new_dict)

            merged_data = np.array(merged_data, dtype=object)

            output_file = os.path.join(dataset_out_dir, f"{sample_id}.npy")
            np.save(output_file, merged_data)

            logging.info(
                f"{hvg_expr_data['expression'].shape[1]} HVGs) -> {output_file}"
            )
            success_count += 1

        except Exception as e:
            skip_count += 1
            continue

    logging.info(
    )
    return success_count, skip_count


def make_range(prefix, start, end):
    if start <= end:
        return [f"{prefix}{i}" for i in range(start, end + 1)]
    else:
        return [f"{prefix}{i}" for i in range(start, end - 1, -1)]


def main():
    OUR_FORMAT_DIR = os.environ.get("OUR_FORMAT_ROOT", "./data/our_format")
    HVG_EXPR_BASE_DIR = os.environ.get("HVG_EXPR_ROOT", "./data/preprocess/filter_hvg_expression")
    OUTPUT_DIR = os.environ.get("DATA_INFOR_ROOT", "./data/data_infor")

    tasks = [
        ("HER2", make_range("SPA", 154, 119)),
    ]

    logging.info("="*60)
    logging.info("="*60)

    total_success = 0
    total_skip = 0

    for dataset_name, sample_ids in tasks:

        hvg_expr_dir = os.path.join(HVG_EXPR_BASE_DIR, dataset_name)

        if not os.path.exists(hvg_expr_dir):
            continue

        success, skip = merge_expression_data(
            OUR_FORMAT_DIR,
            hvg_expr_dir,
            OUTPUT_DIR,
            dataset_name,
            sample_ids
        )

        total_success += success
        total_skip += skip

    logging.info("\n" + "="*60)
    logging.info("="*60)


if __name__ == "__main__":
    main()
