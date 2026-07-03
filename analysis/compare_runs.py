import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_METRICS = [
    "first_turn_example_accuracy",
    "final_example_execution_accuracy",
    "repair_success_rate",
    "schema_alignment_rate",
    "executable_rate",
]


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_path"] = path
    status = data.get("status", {})
    evaluable = sum(status.get(key, 0) for key in ["correct_first_turn", "repaired", "failed"])
    if evaluable:
        data.setdefault("first_turn_example_accuracy", status.get("correct_first_turn", 0) / evaluable)
        data.setdefault(
            "final_example_execution_accuracy",
            (status.get("correct_first_turn", 0) + status.get("repaired", 0)) / evaluable,
        )
    return data


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if 0.0 <= value <= 1.0:
            return f"{value * 100:.2f}%"
        return f"{value:.4f}"
    if isinstance(value, dict):
        return ", ".join(f"{key}:{val}" for key, val in sorted(value.items()))
    if value is None:
        return "-"
    return str(value)


def build_table(rows: List[Dict[str, Any]], labels: List[str], metrics: List[str]) -> str:
    header = ["Run", *metrics]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", *[":---:" for _ in metrics]]) + " |",
    ]
    for label, row in zip(labels, rows):
        values = [label]
        for metric in metrics:
            values.append(format_value(row.get(metric)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def infer_labels(paths: List[str]) -> List[str]:
    labels = []
    for path in paths:
        parent = Path(path).parent.name
        labels.append(parent or Path(path).stem)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Text-to-SQL run summaries as a Markdown table.")
    parser.add_argument("summaries", nargs="+", help="Paths to summary.json files.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels, same count/order as summaries.")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--output", default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    rows = [load_summary(path) for path in args.summaries]
    labels = args.labels or infer_labels(args.summaries)
    if len(labels) != len(rows):
        raise ValueError("--labels must have the same length as summaries.")

    table = build_table(rows, labels, args.metrics)
    print(table)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
