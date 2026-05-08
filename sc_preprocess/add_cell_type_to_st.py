"""Cluster SC embeddings, then assign each ST spot the majority cell type of its nearest cluster."""
import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import time


# =====================================================
# Build MiniBatchKMeans + cluster->cell type voting
# =====================================================
def build_cluster_model(
        sc_records,
        embed_key="embedding",
        cell_type_key="cell_type_id",
        n_clusters=200
    ):

    print("\n===============================")
    print("  Step 1: MiniBatchKMeans (NO PCA)")
    print("===============================")

    sc_embeds = np.stack([r[embed_key] for r in sc_records]).astype("float32")
    sc_types = np.array([r[cell_type_key] for r in sc_records])

    N, D = sc_embeds.shape
    print(f"[SC] Dataset size = {N}, embed_dim = {D}")
    print(f"[Config] n_clusters = {n_clusters}")

    #  Normalize for cosine similarity clustering --------
    norms = np.sqrt((sc_embeds ** 2).sum(axis=1).reshape(-1, 1))
    sc_embeds_norm = sc_embeds / (norms + 1e-8)

    #  MiniBatchKMeans --------
    print("\n[MiniBatchKMeans] Clustering ...")
    t0 = time.time()

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=65536,
        random_state=0,
        n_init="auto"
    )

    cluster_ids = kmeans.fit_predict(sc_embeds_norm)
    centers = kmeans.cluster_centers_

    print(f"[MiniBatchKMeans] Done. Time = {time.time() - t0:.2f}s")

    #  cluster-type majority voting --------
    print("\n[Voting] Determining cluster -> cell type mapping ...")
    cluster_type_map = {}

    for cid in range(n_clusters):
        types = sc_types[cluster_ids == cid]
        if len(types) == 0:
            cluster_type_map[cid] = "unknown"
        else:
            cluster_type_map[cid] = Counter(types).most_common(1)[0][0]

        if cid < 5:
            print(f"  Cluster {cid}: {len(types)} cells -> {cluster_type_map[cid]}")

    print("[Voting] Completed.\n")

    return centers, cluster_type_map


# =====================================================
# Assign SC type for a slide
# =====================================================
def assign_sc_types_for_single_file(
        in_path, out_path,
        cluster_centers, cluster_type_map,
        embed_key="sc_embed",
        batch_size=2048
    ):

    print(f"\n===============================")
    print(f"Processing Slide: {os.path.basename(in_path)}")
    print("===============================")

    spots = np.load(in_path, allow_pickle=True)
    if spots.dtype != object:
        raise ValueError("Each .npy must contain dict objects")

    spot_embeds = np.stack([s[embed_key] for s in spots]).astype("float32")
    N = len(spots)
    K = cluster_centers.shape[0]

    print(f"[Slide] spots = {N}, clusters = {K}")

    #  Normalize --------
    spot_norms = np.sqrt((spot_embeds ** 2).sum(axis=1).reshape(-1, 1))
    spot_embeds_norm = spot_embeds / (spot_norms + 1e-8)

    center_norms = np.sqrt((cluster_centers ** 2).sum(axis=1).reshape(-1, 1))
    centers_norm = cluster_centers / (center_norms + 1e-8)

    sc_type_list = []

    # tqdm batch progress
    for start in tqdm(range(0, N, batch_size),
                      desc=f"Assigning ({os.path.basename(in_path)})",
                      ncols=100):
        end = min(start + batch_size, N)
        batch = spot_embeds_norm[start:end]

        sims = batch @ centers_norm.T      # (B, K)
        nearest_cluster = np.argmax(sims, axis=1)

        for cid in nearest_cluster:
            sc_type_list.append(cluster_type_map[cid])

    # write back
    for s, ct in zip(spots, sc_type_list):
        s["sc_type"] = ct

    np.save(out_path, spots)
    print(f"[Save] {out_path}")


# =====================================================
# main
# =====================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--spot_dir", default="./data/spot_embed/HER2")
    parser.add_argument("--sc_dataset", default="./data/sc_embedding/HER2/merged_data.npy")
    parser.add_argument("--output_dir", default="./data/spot_embed_final/HER2")

    parser.add_argument("--embed_key", default="embedding")
    parser.add_argument("--cell_type_key", default="cell_type_id")

    parser.add_argument("--n_clusters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[Output] Saving to: {args.output_dir}")

    #  Load SC dataset --------
    print(f"\n[Load] SC dataset = {args.sc_dataset}")
    sc_records = np.load(args.sc_dataset, allow_pickle=True)

    #  Step 1: build clustering model --------
    cluster_centers, cluster_type_map = build_cluster_model(
        sc_records=sc_records,
        embed_key=args.embed_key,
        cell_type_key=args.cell_type_key,
        n_clusters=args.n_clusters
    )

    #  Step 2: process slides --------
    spot_files = sorted(glob(os.path.join(args.spot_dir, "*.npy")))
    print(f"[Batch] Found {len(spot_files)} slides.\n")

    total_start = time.time()

    for f in tqdm(spot_files, desc="Processing Slides", ncols=100):
        name = os.path.basename(f)
        out_path = os.path.join(args.output_dir, name)

        assign_sc_types_for_single_file(
            in_path=f,
            out_path=out_path,
            cluster_centers=cluster_centers,
            cluster_type_map=cluster_type_map,
            embed_key="sc_embed",
            batch_size=args.batch_size
        )

    print(f"\n[Done] Total time = {time.time() - total_start:.2f}s\n")


if __name__ == "__main__":
    main()
