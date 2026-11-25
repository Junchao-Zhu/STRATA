# STRATA

A reinforcement-driven sampling framework that reconstructs dense 3D
tissue-level transcriptomic maps from a small set of sampled
voxel-wise ST acquisitions. 

## Repository layout

```
STRATA/
  Registration_and_3D_graph/  Slide-to-slide affine + ANTs registratio
  Tri_network/         Triple-branch predictor 
  Sampler/             Sampling pipeline 
  requirements.txt            Python dependencies
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Point env vars at your data, then train the triple-branch predictor
bash Tri_network/run_train.sh

# 3. Run sampling
bash Sampler/run_st.sh
```

## Requirements

```bash
pip install -r requirements.txt
```
