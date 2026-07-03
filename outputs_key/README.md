# Key Outputs

This directory is intended for small, resume-friendly artifacts only. Keep large
trajectory files, model adapters, and raw rollouts under `outputs/` or external
storage.

Suggested layout:

- `clean_split_summaries/`: final `summary.json` files for Base, DPO, SFT, and SFT+DPO runs.
- `spider200_ablation/`: early development ablation tables.
- `reports/`: Markdown reports such as pair quality, repair breakdown, routing, and run comparison.

The final clean-split numbers in the README are produced from Spider
`train_spider.json` mining and Spider dev full 1034 evaluation.
