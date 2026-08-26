#!/usr/bin/env bash
# ==============================================================================
# EHRAgent Evaluation Runner Script
# Usage:
#   ./run_eval.sh [MODEL_NAME] [NUM_QUESTIONS] [SEED] [NUM_SHOTS] [START_ID]
#
# Examples:
#   ./run_eval.sh deepseek-v4-flash-0731 50 0 4 0
#   ./run_eval.sh gpt-oss-120b-openrouter 50 0 4 0
#   ./run_eval.sh all 50 0 4 0
# ==============================================================================

set -euo pipefail

# 1. 인자 기본값 설정
MODEL="${1:-deepseek-v4-flash-0731}"
NUM_QUESTIONS="${2:-50}"
SEED="${3:-0}"
NUM_SHOTS="${4:-4}"
START_ID="${5:-0}"
DATA_PATH="data/ehrsql-ehragent/mimic_iii/valid_preprocessed.jsonl"
LOGS_DIR="logs"

# 2. 실행 환경 검증
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: 데이터셋 파일이 존재하지 않습니다: $DATA_PATH"
    exit 1
fi

mkdir -p "$LOGS_DIR"

# 3. 단일 모델 실행 함수
run_single_model() {
    local target_model="$1"
    echo "======================================================="
    echo "🚀 Starting Evaluation: [$target_model]"
    echo "Questions: $START_ID ~ $NUM_QUESTIONS | Seed: $SEED | Shots: $NUM_SHOTS"
    echo "Dataset: $DATA_PATH"
    echo "======================================================="

    uv run python evaluate.py \
        --llm "$target_model" \
        --num_questions "$NUM_QUESTIONS" \
        --seed "$SEED" \
        --num_shots "$NUM_SHOTS" \
        --start_id "$START_ID" \
        --data_path "$DATA_PATH" \
        --logs_path "$LOGS_DIR"

    echo ""
    echo "✅ Evaluation completed for [$target_model]"
    echo "Summary saved at: $LOGS_DIR/$NUM_SHOTS/summary_${target_model}_${SEED}.json"
    echo "======================================================="
}

# 4. 모델별 실행 분기
if [ "$MODEL" = "all" ]; then
    MODELS=("deepseek-v4-flash-0731" "gpt-oss-120b-openrouter")
    echo "🚀 Running evaluation for all models: ${MODELS[*]}"
    for m in "${MODELS[@]}"; do
        run_single_model "$m"
    done
else
    run_single_model "$MODEL"
fi
