import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_pairs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def parse_dataset(value: str) -> Tuple[str, str]:
    if "=" not in value:
        path = value
        return Path(path).stem, path
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid dataset item: {value}")
    return label, path


def pair_type(pair: Dict[str, Any]) -> str:
    return pair.get("meta", {}).get("pair_type", "unknown")


def reward_margin(pair: Dict[str, Any]) -> float:
    meta = pair.get("meta", {})
    return float(meta.get("chosen_reward", 0.0)) - float(meta.get("rejected_reward", 0.0))


def strip_completion_prefix(text: str) -> str:
    text = str(text or "").strip()
    return re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def sql_tokens(sql: str) -> List[str]:
    sql = strip_completion_prefix(sql).lower()
    return re.findall(r"[a-z_][a-z0-9_]*|\d+|[<>=!]+|[,().*+-/]", sql)


def edit_distance(left: List[str], right: List[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def normalized_sql_edit_distance(pair: Dict[str, Any]) -> float:
    chosen_tokens = sql_tokens(pair.get("chosen", ""))
    rejected_tokens = sql_tokens(pair.get("rejected", ""))
    denom = max(len(chosen_tokens), len(rejected_tokens), 1)
    return edit_distance(chosen_tokens, rejected_tokens) / denom


def pair_key(pair: Dict[str, Any]) -> Tuple[str, str, str]:
    meta = pair.get("meta", {})
    return (
        meta.get("db_id") or "",
        pair.get("prompt") or "",
        pair.get("chosen") or "",
        pair.get("rejected") or "",
    )


def flag_value(pair: Dict[str, Any], side: str, key: str) -> Any:
    flags = pair.get("meta", {}).get(f"{side}_cost_flags", {})
    if key not in flags:
        return None
    return flags.get(key)


def known_rate(values: Iterable[Any], positive_value: int = 1) -> Tuple[float, int, int]:
    known = [value for value in values if value is not None]
    if not known:
        return 0.0, 0, 0
    positives = sum(1 for value in known if value == positive_value or value is True)
    return positives / len(known), positives, len(known)


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    result = 0.0
    for count in counts.values():
        p = count / total
        result -= p * math.log2(p)
    return result


def synthetic_count(type_counts: Counter) -> int:
    return sum(count for key, count in type_counts.items() if str(key).startswith("synthetic_"))


def natural_count(type_counts: Counter) -> int:
    return sum(count for key, count in type_counts.items() if not str(key).startswith("synthetic_"))


def summarize_pairs(label: str, path: str) -> Dict[str, Any]:
    pairs = load_pairs(path)
    type_counts = Counter(pair_type(pair) for pair in pairs)
    margins = [reward_margin(pair) for pair in pairs]
    distances = [normalized_sql_edit_distance(pair) for pair in pairs]
    keys = [pair_key(pair) for pair in pairs]
    duplicate_count = len(keys) - len(set(keys))

    chosen_exec_rate, chosen_exec_positive, chosen_exec_known = known_rate(
        flag_value(pair, "chosen", "executable") for pair in pairs
    )
    rejected_exec_rate, rejected_exec_positive, rejected_exec_known = known_rate(
        flag_value(pair, "rejected", "executable") for pair in pairs
    )
    rejected_wrong_executable_rate, rejected_wrong_executable_positive, rejected_wrong_executable_known = known_rate(
        (
            1
            if flag_value(pair, "rejected", "executable") == 1
            and float(pair.get("meta", {}).get("rejected_reward", 0.0)) < 1.0
            else 0
            if flag_value(pair, "rejected", "executable") is not None
            else None
        )
        for pair in pairs
    )

    total = len(pairs)
    return {
        "label": label,
        "path": path,
        "pairs": total,
        "natural_pairs": natural_count(type_counts),
        "synthetic_pairs": synthetic_count(type_counts),
        "synthetic_fraction": round(synthetic_count(type_counts) / total, 4) if total else 0.0,
        "avg_reward_margin": round(sum(margins) / len(margins), 4) if margins else 0.0,
        "min_reward_margin": round(min(margins), 4) if margins else 0.0,
        "max_reward_margin": round(max(margins), 4) if margins else 0.0,
        "avg_sql_edit_distance_ratio": round(sum(distances) / len(distances), 4) if distances else 0.0,
        "duplicate_count": duplicate_count,
        "duplicate_rate": round(duplicate_count / total, 4) if total else 0.0,
        "chosen_executable_rate": round(chosen_exec_rate, 4),
        "chosen_executable_known": chosen_exec_known,
        "chosen_executable_unknown": total - chosen_exec_known,
        "rejected_executable_rate": round(rejected_exec_rate, 4),
        "rejected_executable_known": rejected_exec_known,
        "rejected_executable_unknown": total - rejected_exec_known,
        "rejected_executable_hard_negative_rate": round(rejected_wrong_executable_rate, 4),
        "rejected_executable_hard_negative_known": rejected_wrong_executable_known,
        "rejected_executable_hard_negative_unknown": total - rejected_wrong_executable_known,
        "pair_type_entropy": round(entropy(type_counts), 4),
        "pair_types": dict(type_counts),
    }


def format_value(value: Any, as_percent: bool = False) -> str:
    if isinstance(value, float):
        if as_percent:
            return f"{value * 100:.2f}%"
        return f"{value:.4f}"
    return str(value)


def metric_table(results: List[Dict[str, Any]]) -> str:
    metric_specs = [
        ("pairs", "pairs", False),
        ("natural pairs", "natural_pairs", False),
        ("synthetic pairs", "synthetic_pairs", False),
        ("synthetic fraction", "synthetic_fraction", True),
        ("avg reward margin", "avg_reward_margin", False),
        ("min reward margin", "min_reward_margin", False),
        ("avg SQL edit distance ratio", "avg_sql_edit_distance_ratio", False),
        ("duplicate rate", "duplicate_rate", True),
        ("chosen executable rate", "chosen_executable_rate", True),
        ("rejected executable rate", "rejected_executable_rate", True),
        ("executable hard-negative rate", "rejected_executable_hard_negative_rate", True),
        ("pair type entropy", "pair_type_entropy", False),
    ]
    labels = [result["label"] for result in results]
    lines = [
        "| Metric | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join([":---:" for _ in labels]) + " |",
    ]
    for metric_name, key, as_percent in metric_specs:
        values = [format_value(result.get(key), as_percent=as_percent) for result in results]
        lines.append("| " + " | ".join([metric_name, *values]) + " |")
    return "\n".join(lines)


def pair_type_table(results: List[Dict[str, Any]]) -> str:
    all_types = sorted({key for result in results for key in result["pair_types"]})
    labels = [result["label"] for result in results]
    lines = [
        "| Pair Type | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join([":---:" for _ in labels]) + " |",
    ]
    for item in all_types:
        values = [str(result["pair_types"].get(item, 0)) for result in results]
        lines.append("| " + " | ".join([f"`{item}`", *values]) + " |")
    return "\n".join(lines)


def build_markdown(results: List[Dict[str, Any]]) -> str:
    lines = ["# DPO Pair Quality Report", ""]
    lines.append("## Quality Metrics")
    lines.append("")
    lines.append(metric_table(results))
    lines.append("")
    lines.append("## Pair Type Distribution")
    lines.append("")
    lines.append(pair_type_table(results))
    lines.append("")
    lines.append(
        "Note: executable rates are computed from `chosen_cost_flags` / `rejected_cost_flags` when available; "
        "older or external pairs without flags are counted as unknown in the JSON output."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and compare DPO pair quality.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="One or more LABEL=dpo_pairs.json entries, e.g. Raw=raw.json Filtered=filtered.json.",
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    results = [summarize_pairs(*parse_dataset(item)) for item in args.datasets]
    payload = {"datasets": results}
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
