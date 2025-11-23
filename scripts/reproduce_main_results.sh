#!/bin/bash
# Reproduce Main Results (Table 1) from FAORU paper
# Runs all methods with 5 seeds on ImageNet-1K with ViT-B

set -e

# Configuration
DATA_PATH="/path/to/imagenet"
OUTPUT_DIR="outputs/main_results"
NUM_GPUS=8
SEEDS=(42 1337 2024 3141 9999)

echo "================================"
echo "Reproducing FAORU Main Results"
echo "Dataset: ImageNet-1K"
echo "Model: ViT-B"
echo "Seeds: ${SEEDS[@]}"
echo "================================"

# 1. Standard Residual Baseline
echo ""
echo "[1/5] Training Standard Residual..."
for seed in "${SEEDS[@]}"; do
    echo "  Running seed ${seed}..."
    torchrun --nproc_per_node=${NUM_GPUS} train.py \
        --config configs/vit_b_imagenet.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_DIR}/standard_seed${seed} \
        --seed ${seed} \
        --model.faoru.enabled false
done

# 2. Spatial Orthogonal Baseline
echo ""
echo "[2/5] Training Spatial Orthogonal..."
for seed in "${SEEDS[@]}"; do
    echo "  Running seed ${seed}..."
    torchrun --nproc_per_node=${NUM_GPUS} train.py \
        --config configs/vit_b_imagenet.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_DIR}/spatial_orth_seed${seed} \
        --seed ${seed} \
        --model.faoru.enabled true \
        --model.faoru.variant piecewise \
        --model.faoru.transform identity \
        --model.faoru.lambda_low 0.0 \
        --model.faoru.lambda_high 1.0
done

# 3. FAORU-PC (Piecewise Constant)
echo ""
echo "[3/5] Training FAORU-PC..."
for seed in "${SEEDS[@]}"; do
    echo "  Running seed ${seed}..."
    torchrun --nproc_per_node=${NUM_GPUS} train.py \
        --config configs/vit_b_imagenet.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_DIR}/faoru_pc_seed${seed} \
        --seed ${seed} \
        --model.faoru.enabled true \
        --model.faoru.variant piecewise
done

# 4. FAORU-ST (Smooth Transition)
echo ""
echo "[4/5] Training FAORU-ST..."
for seed in "${SEEDS[@]}"; do
    echo "  Running seed ${seed}..."
    torchrun --nproc_per_node=${NUM_GPUS} train.py \
        --config configs/vit_b_imagenet.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_DIR}/faoru_st_seed${seed} \
        --seed ${seed} \
        --model.faoru.enabled true \
        --model.faoru.variant smooth
done

# 5. FAORU-L (Learnable)
echo ""
echo "[5/5] Training FAORU-L..."
for seed in "${SEEDS[@]}"; do
    echo "  Running seed ${seed}..."
    torchrun --nproc_per_node=${NUM_GPUS} train.py \
        --config configs/vit_b_imagenet.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_DIR}/faoru_l_seed${seed} \
        --seed ${seed} \
        --model.faoru.enabled true \
        --model.faoru.variant learnable
done

echo ""
echo "================================"
echo "Training Complete!"
echo "================================"

# Aggregate results
echo ""
echo "Aggregating results..."
python tools/aggregate_results.py \
    --results-dir ${OUTPUT_DIR} \
    --output ${OUTPUT_DIR}/summary.csv

echo ""
echo "Results saved to: ${OUTPUT_DIR}/summary.csv"
echo "To view formatted table:"
echo "  python tools/format_table.py ${OUTPUT_DIR}/summary.csv"
