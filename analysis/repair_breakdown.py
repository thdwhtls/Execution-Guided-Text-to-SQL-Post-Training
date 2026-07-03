import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ERROR_PATTERNS = [
    ("no_such_table", "no such table"),
    ("no_such_column", "no such column"),
    ("ambiguous_column", "ambiguous column name"),
    ("syntax_error", "syntax error"),
    ("timeout", "interrupted"),
    ("timeout", "timeout"),
]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_run(value: str) -> Tuple[str, str]:
    if "=" not in value:
        path = value
        return Path(path).parent.name or Path(path).stem, path
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --runs item: {value}")
    return label, path


def classify_candidate(candidate: Dict[str, Any], gold_row_count: Any = None) -> str:
    if not candidate:
        return "missing_candidate"
    if candidate.get("correct"):
        return "execution_correct"

    error = (candidate.get("error") or "").lower()
    for label, pattern in ERROR_PATTERNS:
        if pattern in error:
            return label

    if candidate.get("ok"):
        row_count = candidate.get("row_count")
        if row_count == 0:
            return "empty_result"
        if isinstance(row_count, int) and isinstance(gold_row_count, int) and row_count != gold_row_count:
            return "wrong_row_count"
        return "wrong_result"

    return "other_execution_error"


def first_turn_candidates(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in traj.get("candidates", []) if item.get("turn") == 1]


def representative_first_turn_error(traj: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [item for item in first_turn_candidates(traj) if not item.get("correct")]
    if not candidates:
        return {}

    # The rollout uses the highest-reward failed candidate as repair context, so
    # this is the most faithful per-example error attribution.
    return sorted(
        candidates,
        key=lambda item: (
            item.get("reward", -999),
            item.get("ok", False),
            -item.get("elapsed_ms", 0.0),
        ),
        reverse=True,
    )[0]


def successful_repair_candidate(traj: Dict[str, Any]) -> Dict[str, Any]:
    repaired = [
        item
        for item in traj.get("candidates", [])
        if item.get("turn", 1) > 1 and item.get("correct")
    ]
    if not repaired:
        return {}
    return sorted(repaired, key=lambda item: (item.get("turn", 99), item.get("elapsed_ms", 0.0)))[0]


def final_candidate(traj: Dict[str, Any]) -> Dict[str, Any]:
    candidates = traj.get("candidates", [])
    return candidates[-1] if candidates else {}


def pct(numerator: int, denominator: int) -> str:
    if not denominator:
        return "/"
    return f"{100.0 * numerator / denominator:.2f}%"


def analyze_run(label: str, path: str) -> Dict[str, Any]:
    status_counts = Counter()
    breakdown: Dict[str, Counter] = defaultdict(Counter)
    transition_counts = Counter()
    total = 0

    for traj in read_jsonl(path):
        total += 1
        status = traj.get("status", "unknown")
        status_counts[status] += 1
        if status == "bad_gold":
            continue

        first_turn = first_turn_candidates(traj)
        if any(item.get("correct") for item in first_turn):
            continue

        first_error = representative_first_turn_error(traj)
        first_type = classify_candidate(first_error, traj.get("gold_row_count"))
        final = final_candidate(traj)
        final_type = classify_candidate(final, traj.get("gold_row_count"))
        repaired = status == "repaired" or bool(successful_repair_candidate(traj))

        row = breakdown[first_type]
        row["first_turn_errors"] += 1
        if repaired:
            row["repaired"] += 1
        else:
            row["final_failed"] += 1
            row[f"final_{final_type}"] += 1
        transition_counts[(first_type, "repaired" if repaired else final_type)] += 1

    rows = []
    for error_type, counts in sorted(
        breakdown.items(),
        key=lambda item: (-item[1]["first_turn_errors"], item[0]),
    ):
        first_turn_errors = counts["first_turn_errors"]
        repaired = counts["repaired"]
        rows.append(
            {
                "error_type": error_type,
                "first_turn_errors": first_turn_errors,
                "repaired": repaired,
                "repair_rate": round(repaired / first_turn_errors, 4) if first_turn_errors else 0.0,
                "final_failed": counts["final_failed"],
                "final_failure_types": {
                    key.replace("final_", ""): value
                    for key, value in sorted(counts.items())
                    if key.startswith("final_") and key != "final_failed" and value
                },
            }
        )

    transitions = [
        {"from": src, "to": dst, "count": count}
        for (src, dst), count in transition_counts.most_common()
    ]

    return {
        "label": label,
        "path": path,
        "total": total,
        "status_counts": dict(status_counts),
        "first_turn_error_total": sum(item["first_turn_errors"] for item in rows),
        "repaired_total": sum(item["repaired"] for item in rows),
        "overall_repair_rate": round(
            sum(item["repaired"] for item in rows) / sum(item["first_turn_errors"] for item in rows),
            4,
        )
        if rows
        else 0.0,
        "breakdown": rows,
        "transitions": transitions,
    }


def format_failure_types(value: Dict[str, int]) -> str:
    if not value:
        return "-"
    return ", ".join(f"`{key}`:{count}" for key, count in value.items())


def build_markdown(results: List[Dict[str, Any]]) -> str:
    lines = ["# Error-Type Repair Breakdown", ""]
    lines.append("## Overall")
    lines.append("")
    lines.append("| Run | Total | First-turn Errors | Repaired | Repair Rate | Status Counts |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for result in results:
        status = ", ".join(f"`{key}`:{value}" for key, value in sorted(result["status_counts"].items()))
        lines.append(
            f"| {result['label']} | {result['total']} | {result['first_turn_error_total']} | "
            f"{result['repaired_total']} | {pct(result['repaired_total'], result['first_turn_error_total'])} | {status} |"
        )

    for result in results:
        lines.extend(["", f"## {result['label']}", ""])
        lines.append("| Error Type | First-turn Errors | Repaired | Repair Rate | Final Failed | Final Failure Types |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for row in result["breakdown"]:
            lines.append(
                f"| `{row['error_type']}` | {row['first_turn_errors']} | {row['repaired']} | "
                f"{pct(row['repaired'], row['first_turn_errors'])} | {row['final_failed']} | "
                f"{format_failure_types(row['final_failure_types'])} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break down first-turn Text-to-SQL errors by repair outcome."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more LABEL=trajectories.jsonl entries.",
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    results = [analyze_run(*parse_run(item)) for item in args.runs]
    payload = {"runs": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
