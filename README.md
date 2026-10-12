# STRATA: Reinforcement-driven Active Sampling with Tri-paradigm Adaptation for 3D Spatial Transcriptomics

A reinforcement-driven active sampling framework that reconstructs dense 3D
tissue-level transcriptomic maps from a small set of strategically sampled
voxel-wise ST acquisitions. 

## Repository layout

```
STRATA/
  Registration_and_3D_graph/  Slide-to-slide affine + ANTs registration, 3D spot graph
  STRATA_Tri_network/         Triple-branch predictor (retrieval / regression / 3D interp)
  STRATA_Sampler/             Active-sampling pipeline (online RL with the predictor)
  requirements.txt            Python dependencies
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt
# (also install CONCH, scGPT, and xFeat from their own sources)

# 2. Point env vars at your data, then train the triple-branch predictor
bash STRATA_Tri_network/run_train.sh

# 3. Run active sampling on top
bash STRATA_Sampler/run_st.sh
```

## Pipeline

**1. Register adjacent slides and build the 3D graph** (`Registration_and_3D_graph/`)

- `registration/Step0..Step3*`, xFeat affine init followed by ANTs nonrigid
- Per-spot CONCH features, pairwise IoU and feature similarity, k-NN graph
  with intra-slice and cross-slice edges (output: `3d_map/{patient}.npz`)

**2. Train the triple-branch predictor** (`STRATA_Tri_network/`)

- Joint training of retrieval (DenseNet-121 + BLEEP), regression (ViT-B with
  CONCH weights), and parameter-free 3D interpolation
- Fusion variants V1 / V3 / V4 trained on cached predictions
- Reported metrics: `3d_only` plus fused `V1_2param` at `top_k=25`

**3. Active sampling with online RL** (`STRATA_Sampler/`)

- Build SC prototypes and CONCH feature cache
- Pointer-decoder policy with online REINFORCE + HER replay; the trained
  triple-branch model is the downstream predictor

## Configuration

All paths are read from environment variables with safe relative defaults
(see each script's header). Common ones:

| Variable      | Purpose                                       |
|---------------|-----------------------------------------------|
| `DATA_ROOT`   | Per-spot npy data (`data_infor`)              |
| `IMAGE_ROOT`  | H&E patch directory                           |
| `GRAPH_ROOT`  | 3D spot-graph npz directory                   |
| `CONCH_CKPT`  | CONCH ViT-B/16 checkpoint path                |
| `SCGPT_CKPT`  | scGPT checkpoint directory                    |

## Requirements

```bash
pip install -r requirements.txt
```

The following are not on PyPI and have to be installed from each project's
own source:

- CONCH, pathology ViT-B/16 foundation model
- scGPT, single-cell foundation model
- xFeat, keypoint matcher used in slide registration
