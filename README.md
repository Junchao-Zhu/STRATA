# STRATA: Reinforcement-driven Active Sampling with Tri-paradigm Adaptation for 3D Spatial Transcriptomics

A reinforcement-driven active sampling framework that reconstructs dense 3D
tissue-level transcriptomic maps from a small set of strategically sampled
voxel-wise ST acquisitions. 

## Repository layout

```
STRATA/
  Registration_and_3D_graph/  Slide-to-slide affine + ANTs registration, 3D spot graph
  Tri_network/         Triple-branch predictor 
  Sampler/             Active-sampling pipeline 
  requirements.txt            Python dependencies
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Point env vars at your data, then train the triple-branch predictor
bash Tri_network/run_train.sh

# 3. Run active sampling on top
bash Sampler/run_st.sh
```

## Pipeline

**1. Register adjacent slides and build the 3D graph** (`Registration_and_3D_graph/`)

- `registration/Step0..Step3*`, xFeat affine init followed by ANTs nonrigid
- Per-spot CONCH features, pairwise IoU and feature similarity, k-NN graph
  with intra-slice and cross-slice edges (output: `3d_map/{patient}.npz`)

**2. Train the triple-branch predictor** (`Tri_network/`)

- Joint training of retrieval (DenseNet-121 + BLEEP), regression (ViT-B with
  CONCH weights), and parameter-free 3D interpolation
- Fusion variants V1 / V3 / V4 trained on cached predictions
- Reported metrics: `3d_only` plus fused `V1_2param` at `top_k=25`

**3. Active sampling with online RL** (`Sampler/`)

- Pointer-decoder policy with online REINFORCE; the trained
  triple-branch model is the downstream predictor

## Requirements

```bash
pip install -r requirements.txt
```