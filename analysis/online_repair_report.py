import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ERROR_PATTERNS = [
    ("no_such_table", "no such table"),
    ("no_such_column", "no such column"),
    ("ambiguous_column", "ambiguous column name"),
    ("syntax_error", "syntax error"),
    ("timeout", "interrupted"),
    ("timeout", "timeout"),
    ("permission_violation", "permission"),
    ("permission_violation", "only select/with read queries are allowed"),
]

STRICT_ONLINE_TYPES = {
    "syntax_error",
    "no_such_table",
    "no_such_column",
    "ambiguous_column",
    "timeout",
    "permission_violation",
    "other_execution_error",
    "missing_candidate",
}

GUARDED_TYPES = {"empty_result"}
RULE_GUARDED_TYPES = {
    "cartesian_join_risk",
    "explain_plan_error",
    "syntax_guard_failed",
    "schema_guard_failed",
    "not_executable",
}


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


def first_turn_candidates(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in traj.get("candidates", []) if item.get("turn") == 1]


def representative_first_turn_error(traj: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [item for item in first_turn_candidates(traj) if not item.get("correct")]
    if not candidates:
        return {}
    return sorted(
        candidates,
        key=lambda item: (
            item.get("reward", -999),
            item.get("ok", False),
            -item.get("elapsed_ms", 0.0),
        ),
        reverse=True,
    )[0]


def classify_online_failure(candidate: Dict[str, Any]) -> str:
    if not candidate:
        return "missing_candidate"
    if candidate.get("correct"):
        return "execution_correct"

    error = (candidate.get("error") or "").lower()
    for label, pattern in ERROR_PATTERNS:
        if pattern in error:
            return label

    flags = candidate.get("cost_flags") or {}
    if flags.get("syntax_valid") == 0:
        return "syntax_guard_failed"
    if flags.get("schema_valid") == 0:
        return "schema_guard_failed"
    if flags.get("executable") == 0:
        return "not_executable"
    if flags.get("explain_error"):
        return "explain_plan_error"

    if candidate.get("ok"):
        if candidate.get("row_count") == 0:
            return "empty_result"
        if flags.get("cartesian"):
            return "cartesian_join_risk"
        return "external_semantic_failure"

    return "other_execution_error"


def repair_bucket(error_type: str) -> str:
    if error_type in STRICT_ONLINE_TYPES:
        return "strict_online"
    if error_type in GUARDED_TYPES or error_type in RULE_GUARDED_TYPES:
        return "guarded"
    if error_type == "execution_correct":
        return "serve"
    return "external_verifier"


def pct(count: int, total: int) -> str:
    if not total:
        return "/"
    return f"{100.0 * count / total:.2f}%"


def analyze_run(label: str, path: str) -> Dict[str, Any]:
    status_counts = Counter()
    error_counts = Counter()
    bucket_counts = Counter()
    fixed_counts = Counter()
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

        error_type = classify_online_failure(representative_first_turn_error(traj))
        bucket = repair_bucket(error_type)
        error_counts[error_type] += 1
        bucket_counts[bucket] += 1
        if status == "repaired":
            fixed_counts[bucket] += 1

    online_total = bucket_counts["strict_online"]
    guarded_total = bucket_counts["guarded"]
    online_guarded_total = online_total + guarded_total
    online_fixed = fixed_counts["strict_online"]
    guarded_fixed = fixed_counts["guarded"]
    online_guarded_fixed = online_fixed + guarded_fixed

    return {
        "label": label,
        "path": path,
        "total": total,
        "status_counts": dict(status_counts),
        "first_turn_failure_total": sum(error_counts.values()),
        "error_counts": dict(error_counts),
        "bucket_counts": dict(bucket_counts),
        "fixed_counts": dict(fixed_counts),
        "strict_online_repairable": online_total,
        "strict_online_fixed": online_fixed,
        "strict_online_repair_sr": online_fixed / online_total if online_total else None,
        "guarded_warnings": guarded_total,
        "guarded_fixed": guarded_fixed,
        "online_guarded_repairable": online_guarded_total,
        "online_guarded_fixed": online_guarded_fixed,
        "online_guarded_repair_sr": online_guarded_fixed / online_guarded_total if online_guarded_total else None,
        "external_verifier_failures": bucket_counts["external_verifier"],
    }


def build_markdown(results: List[Dict[str, Any]]) -> str:
    labels = [result["label"] for result in results]
    lines = ["# Online-Repair-Only Report", ""]
    lines.append(
        "This report excludes gold SQL, gold execution rows, and hidden result-mismatch feedback. "
        "Strict online repair counts only executor-visible failures; guarded repair additionally "
        "counts empty-result sanity warnings."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Run | First-turn failures | Strict online-repairable | Strict fixed | Strict Online Repair SR | "
        "Guarded warnings | Guarded fixed | Online+Guarded Repair SR | External-verifier failures |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        lines.append(
            f"| {result['label']} | {result['first_turn_failure_total']} | "
            f"{result['strict_online_repairable']} | {result['strict_online_fixed']} | "
            f"{pct(result['strict_online_fixed'], result['strict_online_repairable'])} | "
            f"{result['guarded_warnings']} | {result['guarded_fixed']} | "
            f"{pct(result['online_guarded_fixed'], result['online_guarded_repairable'])} | "
            f"{result['external_verifier_failures']} |"
        )

    lines.append("")
    lines.append("## Error Types")
    lines.append("")
    all_error_types = sorted({key for result in results for key in result["error_counts"]})
    lines.append("| Error Type | Bucket | " + " | ".join(labels) + " |")
    lines.append("| --- | --- | " + " | ".join([":---:" for _ in labels]) + " |")
    for error_type in all_error_types:
        counts = [str(result["error_counts"].get(error_type, 0)) for result in results]
        lines.append("| " + " | ".join([f"`{error_type}`", repair_bucket(error_type), *counts]) + " |")

    lines.append("")
    lines.append("## Bucket Definitions")
    lines.append("")
    lines.append("| Bucket | Meaning |")
    lines.append("| --- | --- |")
    lines.append("| `strict_online` | SQL executor, schema checker, parser, or timeout can detect the failure online. |")
    lines.append("| `guarded` | Weak online sanity warning, currently empty result. No gold answer is exposed. |")
    lines.append("| `external_verifier` | Executable semantic failure; needs tests, user feedback, business rule, or offline verifier. |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure online-repair-only success on executor-visible and guarded first-turn failures."
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
    payload = {
        "runs": results,
        "strict_online_types": sorted(STRICT_ONLINE_TYPES),
        "guarded_types": sorted(GUARDED_TYPES),
        "rule_guarded_types": sorted(RULE_GUARDED_TYPES),
    }
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
