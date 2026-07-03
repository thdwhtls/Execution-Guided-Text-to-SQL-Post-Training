#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=models/Qwen2.5-Coder-3B-Instruct}"
: "${OUTPUT_ROOT:=outputs}"
: "${TRAIN_LIMIT:=1000}"

python3 training/train_lora_dpo.py \
  --model_name_or_path "$MODEL_PATH" \
  --dpo_pairs "$OUTPUT_ROOT/train${TRAIN_LIMIT}_filtered_synth20_dpo_pairs.json" \
  --output_dir "$OUTPUT_ROOT/dpo_train${TRAIN_LIMIT}_synth20_qwen25_coder_3b" \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --bf16
