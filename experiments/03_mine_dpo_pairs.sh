#!/usr/bin/env bash
set -euo pipefail

: "${SPIDER_ROOT:=data/spider_data}"
: "${MODEL_PATH:=models/Qwen2.5-Coder-3B-Instruct}"
: "${OUTPUT_ROOT:=outputs}"
: "${TRAIN_LIMIT:=1000}"

ROLLOUT_DIR="$OUTPUT_ROOT/train${TRAIN_LIMIT}_rollout_qwen25_coder_3b"
SYNTH_DIR="$OUTPUT_ROOT/train${TRAIN_LIMIT}_synthetic_errors"
RAW_PAIRS="$OUTPUT_ROOT/train${TRAIN_LIMIT}_raw_dpo_pairs.json"
FILTERED_PAIRS="$OUTPUT_ROOT/train${TRAIN_LIMIT}_filtered_synth20_dpo_pairs.json"

mkdir -p "$OUTPUT_ROOT/reports"

python3 text2sql_trajectory_builder.py \
  --dataset_path "$SPIDER_ROOT/train_spider.json" \
  --db_root "$SPIDER_ROOT/database" \
  --output_dir "$ROLLOUT_DIR" \
  --generator hf \
  --model_path "$MODEL_PATH" \
  --offset 0 \
  --limit "$TRAIN_LIMIT" \
  --max_turns 3 \
  --num_samples 4 \
  --temperature 0.7 \
  --top_p 0.9 \
  --schema_mode retrieved \
  --top_k_tables 6 \
  --feedback_mode oracle_rows \
  --feedback_detail minimal \
  --use_gold_when_failed \
  --progress_every 50

python3 scripts/build_error_sql_samples.py \
  --dataset_path "$SPIDER_ROOT/train_spider.json" \
  --db_root "$SPIDER_ROOT/database" \
  --output_dir "$SYNTH_DIR" \
  --offset 0 \
  --limit "$TRAIN_LIMIT" \
  --max_errors_per_example 4 \
  --schema_mode retrieved \
  --top_k_tables 6 \
  --prefer_executable_wrong \
  --max_schema_error_fraction 0.35

python3 scripts/merge_dpo_pairs.py \
  --inputs \
    "$ROLLOUT_DIR/dpo_pairs.json" \
    "$SYNTH_DIR/dpo_pairs.json" \
  --output "$RAW_PAIRS" \
  --shuffle

python3 scripts/merge_dpo_pairs.py \
  --inputs \
    "$ROLLOUT_DIR/dpo_pairs.json" \
    "$SYNTH_DIR/dpo_pairs.json" \
  --output "$FILTERED_PAIRS" \
  --input_fraction_limits 1=0.20 \
  --pair_type_max_fractions gold_vs_failed_attempt=0.20 \
  --min_reward_margin 0.3 \
  --max_sql_edit_distance_ratio 1.0 \
  --shuffle

python3 analysis/pair_quality_report.py \
  --datasets \
    Raw="$RAW_PAIRS" \
    Filtered="$FILTERED_PAIRS" \
  --output_json "$OUTPUT_ROOT/reports/train${TRAIN_LIMIT}_pair_quality.json" \
  --output_md "$OUTPUT_ROOT/reports/train${TRAIN_LIMIT}_pair_quality.md"
