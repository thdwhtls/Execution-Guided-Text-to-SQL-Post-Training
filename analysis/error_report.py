import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ERROR_PATTERNS = [
    ("no_such_table", "no such table"),
    ("no_such_column", "no such column"),
    ("ambiguous_column", "ambiguous column name"),
    ("syntax_error", "syntax error"),
    ("timeout", "interrupted"),
]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def classify_candidate(candidate: Dict[str, Any]) -> str:
    if candidate.get("correct"):
        return "execution_correct"
    error = (candidate.get("error") or "").lower()
    for label, pattern in ERROR_PATTERNS:
        if pattern in error:
            return label
    if candidate.get("ok"):
        if candidate.get("row_count") == 0:
            return "empty_result_or_wrong_result"
        return "wrong_result"
    return "other_execution_error"


def first_wrong_candidate(traj: Dict[str, Any]) -> Dict[str, Any]:
    for item in traj.get("candidates", []):
        if item.get("turn") == 1 and not item.get("correct"):
            return item
    for item in traj.get("candidates", []):
        if not item.get("correct"):
            return item
    return {}


def final_candidate(traj: Dict[str, Any]) -> Dict[str, Any]:
    candidates = traj.get("candidates", [])
    return candidates[-1] if candidates else {}


def truncate(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_markdown(
    summary: Dict[str, Any],
    examples: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    lines = ["# Text-to-SQL Error Report", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")

    lines.append("")
    lines.append("## Sample Cases")
    lines.append("")
    for idx, item in enumerate(examples, start=1):
        lines.append(f"### Case {idx}: {item['status']} / {item['error_type']}")
        lines.append("")
        lines.append(f"- db_id: `{item.get('db_id')}`")
        lines.append(f"- question: {truncate(item.get('question', ''), max_chars)}")
        lines.append(f"- gold_sql: `{truncate(item.get('gold_sql', ''), max_chars)}`")
        lines.append(f"- first_wrong_sql: `{truncate(item.get('first_wrong_sql', ''), max_chars)}`")
        lines.append(f"- first_wrong_error: `{truncate(item.get('first_wrong_error', ''), max_chars)}`")
        lines.append(f"- final_sql: `{truncate(item.get('final_sql', ''), max_chars)}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Text-to-SQL trajectory errors.")
    parser.add_argument("--trajectories", required=True, help="Path to trajectories.jsonl.")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--max_chars", type=int, default=260)
    args = parser.parse_args()

    status_counts = Counter()
    error_counts = Counter()
    first_turn_error_counts = Counter()
    examples = []

    for traj in read_jsonl(args.trajectories):
        status = traj.get("status", "unknown")
        status_counts[status] += 1
        candidates = traj.get("candidates", [])

        for candidate in candidates:
            error_counts[classify_candidate(candidate)] += 1
            if candidate.get("turn") == 1:
                first_turn_error_counts[classify_candidate(candidate)] += 1

        if status != "correct_first_turn" and len(examples) < args.max_examples:
            first_wrong = first_wrong_candidate(traj)
            final = final_candidate(traj)
            examples.append(
                {
                    "status": status,
                    "error_type": classify_candidate(first_wrong) if first_wrong else "missing_candidate",
                    "db_id": traj.get("db_id"),
                    "question": traj.get("question"),
                    "gold_sql": traj.get("gold_sql"),
                    "first_wrong_sql": first_wrong.get("sql"),
                    "first_wrong_error": first_wrong.get("error"),
                    "final_sql": final.get("sql"),
                }
            )

    summary = {
        "total_trajectories": sum(status_counts.values()),
        "status_counts": dict(status_counts),
        "candidate_error_counts": dict(error_counts),
        "first_turn_error_counts": dict(first_turn_error_counts),
        "failed_repair": status_counts.get("failed", 0),
        "successful_repair": status_counts.get("repaired", 0),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps({"summary": summary, "examples": examples}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown(summary, examples, args.max_chars), encoding="utf-8")


if __name__ == "__main__":
    main()
