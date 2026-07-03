#!/usr/bin/env bash
set -euo pipefail

: "${SPIDER_ROOT:=data/spider_data}"
: "${MODEL_PATH:=models/Qwen2.5-Coder-3B-Instruct}"
: "${OUTPUT_ROOT:=outputs}"

python3 text2sql_trajectory_builder.py \
  --dataset_path "$SPIDER_ROOT/dev.json" \
  --db_root "$SPIDER_ROOT/database" \
  --output_dir "$OUTPUT_ROOT/devfull_base_oneshot" \
  --generator hf \
  --model_path "$MODEL_PATH" \
  --max_turns 1 \
  --num_samples 1 \
  --temperature 0 \
  --top_p 1 \
  --schema_mode retrieved \
  --top_k_tables 6 \
  --feedback_mode result_status \
  --feedback_detail minimal \
  --progress_every 50
