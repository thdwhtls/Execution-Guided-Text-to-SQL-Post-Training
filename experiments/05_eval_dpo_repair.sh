#!/usr/bin/env bash
set -euo pipefail

: "${SPIDER_ROOT:=data/spider_data}"
: "${MODEL_PATH:=models/Qwen2.5-Coder-3B-Instruct}"
: "${OUTPUT_ROOT:=outputs}"
: "${TRAIN_LIMIT:=1000}"

python3 text2sql_trajectory_builder.py \
  --dataset_path "$SPIDER_ROOT/dev.json" \
  --db_root "$SPIDER_ROOT/database" \
  --output_dir "$OUTPUT_ROOT/devfull_dpo_train${TRAIN_LIMIT}_synth20_repair_cm" \
  --generator hf \
  --model_path "$MODEL_PATH" \
  --adapter_path "$OUTPUT_ROOT/dpo_train${TRAIN_LIMIT}_synth20_qwen25_coder_3b/adapter" \
  --max_turns 3 \
  --num_samples 1 \
  --temperature 0 \
  --top_p 1 \
  --schema_mode retrieved \
  --top_k_tables 6 \
  --feedback_mode result_status \
  --feedback_detail minimal \
  --progress_every 50
