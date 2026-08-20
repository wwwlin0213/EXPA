#!/usr/bin/env bash
set -e

# ============================================================================
# 0. 总配置：以后主要只改这里
# ============================================================================
BATCH_SIZE=8
GPU=cuda:0

MODEL_TAG="pe_l14_336"
BACKBONE="pe_l14_336"
# pe_l14_336    clip_vitl14_336     dinov2_vitl14

TRAIN_SCRIPT="train.py"
TEST_SCRIPT="test.py"

# ./experiments/layer/pe
EXP_ROOT="./experiments/${MODEL_TAG}"
LOG_DIR="./logs/${MODEL_TAG}"

FEATURES_LIST=(6 12 18 24)

USE_CEPA=true
USE_AMRF=true

MVTEC_TO_VISA_EPOCHS=2
VISA_TO_MVTEC_EPOCHS=1

# Please modify the following paths according to your environment.
MVTEC_PATH="/path/to/MVTec_AD"
VISA_PATH="/path/to/VisA"
# ============================================================================
# 1. 日志配置
# ============================================================================
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "Cross-Dataset Evaluation - ${MODEL_TAG}"
echo "================================================================"
echo "Batch Size: ${BATCH_SIZE}"
echo "GPU: ${GPU}"
echo "Backbone: ${BACKBONE}"
echo "Train Script: ${TRAIN_SCRIPT}"
echo "Test Script: ${TEST_SCRIPT}"
echo "Experiment Root: ${EXP_ROOT}"
echo "Log File: ${LOG_FILE}"
echo "Dataset Paths:"
echo "  - MVTec: ${MVTEC_PATH}"
echo "  - VisA:  ${VISA_PATH}"
echo "Epoch Configuration:"
echo "  - MVTec → VisA: ${MVTEC_TO_VISA_EPOCHS} epoch"
echo "  - VisA → MVTec: ${VISA_TO_MVTEC_EPOCHS} epoch"
echo "================================================================"
echo ""

# ============================================================================
# 2. 训练测试函数
# ============================================================================
run_experiment() {
    local train_dataset=$1
    local test_dataset=$2
    local train_path=$3
    local test_path=$4
    local epochs=$5

    local exp_name="${train_dataset}_to_${test_dataset}"
    local exp_dir="${EXP_ROOT}/${exp_name}"
    local checkpoint_dir="${exp_dir}/checkpoints"
    local result_dir="${exp_dir}/results"

    mkdir -p "${checkpoint_dir}"
    mkdir -p "${result_dir}/epoch_${epochs}"

    echo ""
    echo "=========================================="
    echo "Experiment: ${exp_name}"
    echo "Training on: ${train_dataset} (${epochs} epochs)"
    echo "Testing on: ${test_dataset}"
    echo "Checkpoint Dir: ${checkpoint_dir}"
    echo "Result Dir: ${result_dir}/epoch_${epochs}"
    echo "=========================================="

    # ==========================
    # 训练阶段
    # ==========================
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training on ${train_dataset}..."

    python "${TRAIN_SCRIPT}" \
        --train_data_path "${train_path}" \
        --save_path "${checkpoint_dir}" \
        --train_dataset "${train_dataset}" \
        --backbone "${BACKBONE}" \
        --features_list "${FEATURES_LIST[@]}" \
        --use_amrf "${USE_AMRF}" \
        --use_cepa "${USE_CEPA}" \
        --epoch "${epochs}" \
        --batch_size "${BATCH_SIZE}" \
        --device "${GPU}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Training completed!"

    # ==========================
    # 测试阶段
    # ==========================
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing on ${test_dataset}..."

    python "${TEST_SCRIPT}" \
        --test_data_path "${test_path}" \
        --checkpoint_path "${checkpoint_dir}/epoch_${epochs}.pth" \
        --test_dataset "${test_dataset}" \
        --save_path "${result_dir}/epoch_${epochs}" \
        --device "${GPU}" \
        --use_amrf "${USE_AMRF}" \
        --use_cepa "${USE_CEPA}"

    echo "✅ Test completed: ${exp_name}"
}

# ============================================================================
# 3. 实验 1: MVTec → VisA
# ============================================================================
echo ""
echo "################################################################"
echo "# Experiment 1/2: MVTec → VisA"
echo "################################################################"

run_experiment \
    "mvtec" \
    "visa" \
    "${MVTEC_PATH}" \
    "${VISA_PATH}" \
    "${MVTEC_TO_VISA_EPOCHS}"

# ============================================================================
# 4. 实验 2: VisA → MVTec
# ============================================================================
echo ""
echo "################################################################"
echo "# Experiment 2/2: VisA → MVTec"
echo "################################################################"

run_experiment \
    "visa" \
    "mvtec" \
    "${VISA_PATH}" \
    "${MVTEC_PATH}" \
    "${VISA_TO_MVTEC_EPOCHS}"

# ============================================================================
# 5. 完成
# ============================================================================
echo ""
echo "================================================================"
echo "ALL ${MODEL_TAG} EXPERIMENTS COMPLETED!"
echo "Log File: ${LOG_FILE}"
echo "================================================================"