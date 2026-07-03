#!/usr/bin/env bash
set -euo pipefail

: "${OUTPUT_ROOT:=outputs}"

mkdir -p "$OUTPUT_ROOT/reports"

summaries=(
  "$OUTPUT_ROOT/devfull_base_oneshot/summary.json"
  "$OUTPUT_ROOT/devfull_base_repair_minimal/summary.json"
  "$OUTPUT_ROOT/devfull_dpo_train1000_synth20_repair_cm/summary.json"
)

labels=(
  Base-OneShot
  Base-CM
  DPO-Synth20-CM
)

if [[ -f "$OUTPUT_ROOT/devfull_sft_dpo_train1000_synth20_repair_cm/summary.json" ]]; then
  summaries+=("$OUTPUT_ROOT/devfull_sft_dpo_train1000_synth20_repair_cm/summary.json")
  labels+=(SFT+DPO-Synth20-CM)
fi

python3 analysis/compare_runs.py \
  "${summaries[@]}" \
  --labels "${labels[@]}" \
  --output "$OUTPUT_ROOT/reports/devfull_main_compare.md"

python3 analysis/repair_breakdown.py \
  --runs \
    Base-CM="$OUTPUT_ROOT/devfull_base_repair_minimal/trajectories.jsonl" \
    DPO-Synth20-CM="$OUTPUT_ROOT/devfull_dpo_train1000_synth20_repair_cm/trajectories.jsonl" \
  --output_json "$OUTPUT_ROOT/reports/devfull_repair_breakdown.json" \
  --output_md "$OUTPUT_ROOT/reports/devfull_repair_breakdown.md"

python3 analysis/repair_routing_report.py \
  --runs \
    Base-CM="$OUTPUT_ROOT/devfull_base_repair_minimal/trajectories.jsonl" \
    DPO-Synth20-CM="$OUTPUT_ROOT/devfull_dpo_train1000_synth20_repair_cm/trajectories.jsonl" \
  --output_json "$OUTPUT_ROOT/reports/devfull_repair_routing.json" \
  --output_md "$OUTPUT_ROOT/reports/devfull_repair_routing.md"
