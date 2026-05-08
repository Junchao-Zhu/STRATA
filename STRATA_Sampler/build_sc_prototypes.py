"""Cluster the single-cell atlas into K prototype centroids.

Loads scGPT-style cell embeddings (merged_data.npy under SC_ROOT/<dataset>/),
fits MiniBatchKMeans in 512-d space, and writes ./sc_prototypes/<dataset>.npz
with centroids, cluster frequencies, and metadata.
"""
import argparse
import os
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans

SC_ROOT = Path(os.environ.get("SC_ROOT", "./data/sc_embedding"))
OUT_ROOT = Path(os.environ.get("SC_PROTO_ROOT", "./sc_prototypes"))


def load_sc_embeddings(sc_dir: Path) -> np.ndarray:
    merged = np.load(sc_dir / "merged_data.npy", allow_pickle=True)
    n = len(merged)
    out = np.zeros((n, 512), dtype=np.float32)
    for i, cell in enumerate(merged):
        out[i] = np.asarray(cell["embedding"], dtype=np.float32)
        if i % 500_000 == 0 and i > 0:
            print(f"    loaded {i}/{n}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HER2")
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sc_dir = SC_ROOT / args.dataset
    if not sc_dir.exists():
        raise FileNotFoundError(f"No SC bank for {args.dataset} at {sc_dir}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_file = OUT_ROOT / f"{args.dataset}.npz"
    if out_file.exists() and not args.force:
        z = np.load(out_file)
        print(f"[build_sc_prototypes] {out_file} exists "
              f"(K={z['K']}, n_sc_cells={z['n_sc_cells']}); skip. Use --force to re-run.")
        return

    print(f"[build_sc_prototypes] loading {sc_dir}/merged_data.npy ...")
    X = load_sc_embeddings(sc_dir)
    print(f"  X shape: {X.shape}")

    print(f"[build_sc_prototypes] MiniBatchKMeans K={args.K} ...")
    km = MiniBatchKMeans(
        n_clusters=args.K,
        random_state=args.seed,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        verbose=0,
    )
    km.fit(X)
    labels = km.labels_
    freq = np.bincount(labels, minlength=args.K).astype(np.float64)
    freq /= freq.sum()
    print(f"  cluster size range: {int(freq.min() * len(X))} - {int(freq.max() * len(X))}")
    print(f"  entropy: {(-freq[freq > 0] * np.log(freq[freq > 0])).sum():.3f} "
          f"(max log K = {np.log(args.K):.3f})")

    np.savez(
        out_file,
        centroids=km.cluster_centers_.astype(np.float32),
        freq=freq.astype(np.float32),
        n_sc_cells=np.int64(len(X)),
        K=np.int64(args.K),
    )
    print(f"[build_sc_prototypes] saved to {out_file}")


if __name__ == "__main__":
    main()
