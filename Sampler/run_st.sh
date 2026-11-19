#!/bin/bash
# Run STRATA active sampling on HER2 with the ST-Net predictor.

set -u
PY=python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

datasets=(HER2)
total_ratios=(0.15 0.3 0.5 0.75)
gpus=(0)

batch_size=128
max_epochs=160
max_iterations=100000000
base_lr=5e-5
lr_decay=0.9
seed=42
warmup_epochs=5
sample_interval=5
sampling_cutoff=25

echo "========================================"
echo "STRATA active sampling"
echo "========================================"

# GPU pool via FIFO.
gpu_fifo=$(mktemp -u)
mkfifo "$gpu_fifo"
exec 3<>"$gpu_fifo"
rm "$gpu_fifo"
for g in "${gpus[@]}"; do echo "$g" >&3; done

i=0
for ds in "${datasets[@]}"; do
  for total_ratio in "${total_ratios[@]}"; do
    read -r gpu <&3

    echo ""
    echo "==== Experiment $((i+1)): ${ds} ratio=${total_ratio} gpu=${gpu} ===="

    (
      CUDA_VISIBLE_DEVICES=$gpu $PY "$SCRIPT_DIR/train_online.py" \
        --dataset "$ds" \
        --total_ratio $total_ratio \
        --warmup_epochs $warmup_epochs \
        --sample_interval $sample_interval \
        --sampling_cutoff $sampling_cutoff \
        --gpu "$gpu" \
        --batch_size $batch_size \
        --max_epochs $max_epochs \
        --max_iterations $max_iterations \
        --base_lr $base_lr \
        --lr_decay $lr_decay \
        --her_replay_epochs 20 \
        --seed $seed
      echo "$gpu" >&3
    ) &

    i=$((i+1))
  done
done

wait
exec 3>&-

echo "All experiments done."
