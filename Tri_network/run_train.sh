#!/bin/bash
# Triple-Branch 3D Gene Expression Prediction
# Usage: bash run_train.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Triple-Branch 3D Training"
echo "  HER2 mask_300, GPU 0"
echo "============================================"

# HER2 mask_300 (single run to test)
nohup python "$SCRIPT_DIR/train.py" \
    --dataset HER2 \
    --mask_ratio 300 \
    --gpu 0 \
    --stage1_epochs 150 \
    --stage2_epochs 50 \
    --fusion_epochs 50 \
    --lambda_3d 1.0 \
    --lambda_con 0.1 \
    > "$LOG_DIR/her2_mask300.log" 2>&1 &

echo "PID: $!"
echo "Log: tail -f $LOG_DIR/her2_mask300.log"
