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
]


ROUTING_POLICY = {
    "syntax_error": {
        "online_detectable": "yes",
        "verifier": "SQL executor",
        "route": "Column-Minimal Repair",
        "deployment_scope": "online repair",
    },
    "no_such_table": {
        "online_detectable": "yes",
        "verifier": "SQL executor / schema checker",
        "route": "schema-aware Column-Minimal Repair",
        "deployment_scope": "online repair",
    },
    "no_such_column": {
        "online_detectable": "yes",
        "verifier": "SQL executor / schema checker",
        "route": "schema-aware Column-Minimal Repair",
        "deployment_scope": "online repair",
    },
    "ambiguous_column": {
        "online_detectable": "yes",
        "verifier": "SQL executor",
        "route": "alias disambiguation repair",
        "deployment_scope": "online repair",
    },
    "timeout": {
        "online_detectable": "yes",
        "verifier": "SQL executor timeout",
        "route": "cost-aware rewrite / join simplification",
        "deployment_scope": "online repair",
    },
    "other_execution_error": {
        "online_detectable": "yes",
        "verifier": "SQL executor",
        "route": "execution-error repair or safe retry",
        "deployment_scope": "online repair",
    },
    "empty_result": {
        "online_detectable": "weak",
        "verifier": "result sanity check / business rule",
        "route": "condition-value grounding feedback",
        "deployment_scope": "guarded repair",
    },
    "wrong_row_count": {
        "online_detectable": "external verifier required",
        "verifier": "tests / business rule / user feedback",
        "route": "verified semantic repair / DPO mining",
        "deployment_scope": "offline or external-verifier repair",
    },
    "wrong_result": {
        "online_detectable": "external verifier required",
        "verifier": "gold result / tests / business rule / user feedback",
        "route": "verified semantic repair / DPO mining",
        "deployment_scope": "offline or external-verifier repair",
    },
    "missing_candidate": {
        "online_detectable": "yes",
        "verifier": "generation parser",
        "route": "generation retry / stricter output parser",
        "deployment_scope": "online retry",
    },
    "execution_correct": {
        "online_detectable": "not needed",
        "verifier": "executor/verifier passed",
        "route": "return SQL",
        "deployment_scope": "serve",
    },
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
    return sorted(
        candidates,
        key=lambda item: (
            item.get("reward", -999),
            item.get("ok", False),
            -item.get("elapsed_ms", 0.0),
        ),
        reverse=True,
    )[0]


def route_for(error_type: str) -> Dict[str, str]:
    return ROUTING_POLICY.get(
        error_type,
        {
            "online_detectable": "unknown",
            "verifier": "unknown",
            "route": "manual inspection",
            "deployment_scope": "unknown",
        },
    )


def analyze_run(label: str, path: str) -> Dict[str, Any]:
    status_counts = Counter()
    error_counts = Counter()
    route_counts = Counter()
    deployment_counts = Counter()
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

        error_type = classify_candidate(
            representative_first_turn_error(traj),
            traj.get("gold_row_count"),
        )
        policy = route_for(error_type)
        error_counts[error_type] += 1
        route_counts[policy["route"]] += 1
        deployment_counts[policy["deployment_scope"]] += 1

    return {
        "label": label,
        "path": path,
        "total": total,
        "status_counts": dict(status_counts),
        "first_turn_error_total": sum(error_counts.values()),
        "error_counts": dict(error_counts),
        "route_counts": dict(route_counts),
        "deployment_scope_counts": dict(deployment_counts),
    }


def pct(count: int, total: int) -> str:
    if not total:
        return "/"
    return f"{100.0 * count / total:.2f}%"


def build_policy_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_error_types = sorted(
        {error_type for result in results for error_type in result["error_counts"]}
    )
    rows = []
    for error_type in all_error_types:
        policy = route_for(error_type)
        row = {
            "error_type": error_type,
            "online_detectable": policy["online_detectable"],
            "verifier": policy["verifier"],
            "recommended_route": policy["route"],
            "deployment_scope": policy["deployment_scope"],
            "counts": {result["label"]: result["error_counts"].get(error_type, 0) for result in results},
        }
        rows.append(row)
    return rows


def build_markdown(results: List[Dict[str, Any]]) -> str:
    labels = [result["label"] for result in results]
    lines = ["# Verifier-Aware Repair Routing Report", ""]
    lines.append(
        "This report separates executor-visible failures that can be repaired online "
        "from semantic failures that require an external verifier or offline mining."
    )
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append("| Run | Total | First-turn Errors | Online Repairable | External/Guarded | Status Counts |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for result in results:
        online = sum(
            count
            for scope, count in result["deployment_scope_counts"].items()
            if scope in {"online repair", "online retry"}
        )
        external = result["first_turn_error_total"] - online
        status = ", ".join(f"`{key}`:{value}" for key, value in sorted(result["status_counts"].items()))
        lines.append(
            f"| {result['label']} | {result['total']} | {result['first_turn_error_total']} | "
            f"{online} ({pct(online, result['first_turn_error_total'])}) | "
            f"{external} ({pct(external, result['first_turn_error_total'])}) | {status} |"
        )

    lines.append("")
    lines.append("## Routing Policy")
    lines.append("")
    header = [
        "Error Type",
        *labels,
        "Online Detectable",
        "Verifier",
        "Recommended Route",
        "Deployment Scope",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---", *[":---:" for _ in labels], "---", "---", "---", "---"]) + " |")
    for row in build_policy_rows(results):
        counts = [str(row["counts"].get(label, 0)) for label in labels]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['error_type']}`",
                    *counts,
                    row["online_detectable"],
                    row["verifier"],
                    row["recommended_route"],
                    row["deployment_scope"],
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Deployment Scope Counts")
    lines.append("")
    all_scopes = sorted({scope for result in results for scope in result["deployment_scope_counts"]})
    lines.append("| Scope | " + " | ".join(labels) + " |")
    lines.append("| --- | " + " | ".join([":---:" for _ in labels]) + " |")
    for scope in all_scopes:
        counts = [str(result["deployment_scope_counts"].get(scope, 0)) for result in results]
        lines.append("| " + " | ".join([scope, *counts]) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route first-turn Text-to-SQL failures by verifier availability."
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
        "routing_policy": ROUTING_POLICY,
        "routing_rows": build_policy_rows(results),
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
