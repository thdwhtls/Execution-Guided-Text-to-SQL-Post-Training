#!/usr/bin/env bash
set -euo pipefail

: "${SPIDER_ROOT:=data/spider_data}"
: "${MODEL_PATH:=models/Qwen2.5-Coder-3B-Instruct}"
: "${OUTPUT_ROOT:=outputs}"
: "${ADAPTER_PATH:=$OUTPUT_ROOT/sft_dpo_train1000_synth20_qwen25_coder_3b/adapter}"

python3 text2sql_trajectory_builder.py \
  --dataset_path "$SPIDER_ROOT/dev.json" \
  --db_root "$SPIDER_ROOT/database" \
  --output_dir "$OUTPUT_ROOT/devfull_base_online_repair_cm" \
  --generator hf \
  --model_path "$MODEL_PATH" \
  --max_turns 3 \
  --num_samples 1 \
  --temperature 0 \
  --top_p 1 \
  --schema_mode retrieved \
  --top_k_tables 6 \
  --feedback_mode online_visible \
  --feedback_detail minimal \
  --repair_scope online_guarded \
  --progress_every 50

python3 text2sql_trajectory_builder.py \
  --dataset_path "$SPIDER_ROOT/dev.json" \
  --db_root "$SPIDER_ROOT/database" \
  --output_dir "$OUTPUT_ROOT/devfull_sft_dpo_online_repair_cm" \
  --generator hf \
  --model_path "$MODEL_PATH" \
  --adapter_path "$ADAPTER_PATH" \
  --max_turns 3 \
  --num_samples 1 \
  --temperature 0 \
  --top_p 1 \
  --schema_mode retrieved \
  --top_k_tables 6 \
  --feedback_mode online_visible \
  --feedback_detail minimal \
  --repair_scope online_guarded \
  --progress_every 50

python3 analysis/online_repair_report.py \
  --runs \
    Base-OnlineCM="$OUTPUT_ROOT/devfull_base_online_repair_cm/trajectories.jsonl" \
    SFT+DPO-OnlineCM="$OUTPUT_ROOT/devfull_sft_dpo_online_repair_cm/trajectories.jsonl" \
  --output_json "$OUTPUT_ROOT/reports/devfull_online_repair_only.json" \
  --output_md "$OUTPUT_ROOT/reports/devfull_online_repair_only.md"
