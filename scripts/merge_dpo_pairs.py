import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_pairs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def pair_key(pair: Dict[str, Any]) -> Tuple[str, str, str]:
    meta = pair.get("meta", {})
    return (
        meta.get("db_id") or "",
        pair.get("prompt") or "",
        pair.get("rejected") or "",
    )


def parse_limits(items: List[str]) -> Dict[str, int]:
    limits = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Limit must be KEY=N, got: {item}")
        key, value = item.split("=", 1)
        limits[key] = int(value)
    return limits


def parse_fractions(items: List[str]) -> Dict[str, float]:
    fractions = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Fraction must be KEY=FLOAT, got: {item}")
        key, value = item.split("=", 1)
        fraction = float(value)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Fraction for {key} must be in [0, 1], got: {fraction}")
        fractions[key] = fraction
    return fractions


def pair_type(pair: Dict[str, Any]) -> str:
    return pair.get("meta", {}).get("pair_type", "unknown")


def reward_margin(pair: Dict[str, Any]) -> float:
    meta = pair.get("meta", {})
    chosen_reward = float(meta.get("chosen_reward", 0.0))
    rejected_reward = float(meta.get("rejected_reward", 0.0))
    return chosen_reward - rejected_reward


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


def apply_pair_type_fraction_caps(
    pairs: List[Dict[str, Any]],
    max_fractions: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], Counter]:
    if not max_fractions:
        return pairs, Counter()

    kept = pairs
    skipped = Counter()
    for limited_type, fraction in max_fractions.items():
        if fraction >= 1.0:
            continue

        limited = [pair for pair in kept if pair_type(pair) == limited_type]
        others = [pair for pair in kept if pair_type(pair) != limited_type]
        if not limited:
            continue
        if fraction <= 0.0:
            skipped[limited_type] += len(limited)
            kept = others
            continue

        max_limited = int((fraction * len(others)) // (1.0 - fraction))
        keep_limited = limited[:max_limited]
        skipped[limited_type] += max(0, len(limited) - len(keep_limited))

        rebuilt = []
        remaining_limited = iter(keep_limited)
        kept_limited_ids = {id(pair) for pair in keep_limited}
        for pair in kept:
            if pair_type(pair) != limited_type:
                rebuilt.append(pair)
            elif id(pair) in kept_limited_ids:
                rebuilt.append(next(remaining_limited))
        kept = rebuilt
    return kept, skipped


def apply_input_fraction_caps(
    pairs: List[Dict[str, Any]],
    max_fractions: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], Counter]:
    if not max_fractions:
        return pairs, Counter()

    kept = pairs
    skipped = Counter()
    for input_idx, fraction in max_fractions.items():
        if fraction >= 1.0:
            continue

        limited = [pair for pair in kept if str(pair.get("_input_idx", "")) == input_idx]
        others = [pair for pair in kept if str(pair.get("_input_idx", "")) != input_idx]
        if not limited:
            continue
        if fraction <= 0.0:
            skipped[input_idx] += len(limited)
            kept = others
            continue

        max_limited = int((fraction * len(others)) // (1.0 - fraction))
        keep_limited_ids = {id(pair) for pair in limited[:max_limited]}
        skipped[input_idx] += max(0, len(limited) - max_limited)

        kept = [
            pair
            for pair in kept
            if str(pair.get("_input_idx", "")) != input_idx or id(pair) in keep_limited_ids
        ]
    return kept, skipped


def strip_internal_fields(pair: Dict[str, Any]) -> Dict[str, Any]:
    if "_input_idx" not in pair:
        return pair
    cleaned = dict(pair)
    cleaned.pop("_input_idx", None)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge DPO pair JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument(
        "--input_limits",
        nargs="*",
        default=None,
        help="Optional per-input limits by 0-based input index, e.g. 0=308 1=150.",
    )
    parser.add_argument(
        "--input_fraction_limits",
        nargs="*",
        default=None,
        help="Optional max fraction per input index in the accepted output, e.g. 1=0.25 for synthetic input.",
    )
    parser.add_argument(
        "--pair_type_limits",
        nargs="*",
        default=None,
        help="Optional pair type limits, e.g. synthetic_wrong_table=30 gold_vs_failed_attempt=120.",
    )
    parser.add_argument(
        "--pair_type_max_fractions",
        nargs="*",
        default=None,
        help=(
            "Optional max fraction per pair type in the accepted output, e.g. "
            "gold_vs_failed_attempt=0.2. Fractions are enforced online while merging."
        ),
    )
    parser.add_argument(
        "--min_reward_margin",
        type=float,
        default=None,
        help="Keep only pairs with chosen_reward - rejected_reward >= this value.",
    )
    parser.add_argument(
        "--max_sql_edit_distance_ratio",
        type=float,
        default=None,
        help=(
            "Keep only pairs whose normalized token edit distance between chosen and rejected SQL "
            "is <= this value. Useful for repair-style pairs where chosen/rejected should be related."
        ),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    input_limits = parse_limits(args.input_limits or [])
    input_fraction_limits = parse_fractions(args.input_fraction_limits or [])
    pair_type_limits = parse_limits(args.pair_type_limits or [])
    pair_type_max_fractions = parse_fractions(args.pair_type_max_fractions or [])

    merged = []
    seen = set()
    skipped_duplicates = 0
    skipped_input_limit = Counter()
    skipped_input_fraction = Counter()
    skipped_pair_type_limit = Counter()
    skipped_reward_margin = Counter()
    skipped_sql_distance = Counter()
    pair_type_counts = Counter()
    edit_distance_stats = []

    for input_idx, path in enumerate(args.inputs):
        pairs = load_pairs(path)
        if args.shuffle:
            rng.shuffle(pairs)

        accepted_for_input = 0
        input_key = str(input_idx)
        for pair in pairs:
            current_pair_type = pair_type(pair)
            if input_key in input_limits and accepted_for_input >= input_limits[input_key]:
                skipped_input_limit[input_key] += 1
                continue
            if (
                current_pair_type in pair_type_limits
                and pair_type_counts[current_pair_type] >= pair_type_limits[current_pair_type]
            ):
                skipped_pair_type_limit[current_pair_type] += 1
                continue
            if args.min_reward_margin is not None and reward_margin(pair) < args.min_reward_margin:
                skipped_reward_margin[current_pair_type] += 1
                continue
            if args.max_sql_edit_distance_ratio is not None:
                distance_ratio = normalized_sql_edit_distance(pair)
                if distance_ratio > args.max_sql_edit_distance_ratio:
                    skipped_sql_distance[current_pair_type] += 1
                    continue
            else:
                distance_ratio = None

            key = pair_key(pair)
            if key in seen:
                skipped_duplicates += 1
                continue

            if distance_ratio is not None:
                edit_distance_stats.append(distance_ratio)
            seen.add(key)
            tagged_pair = dict(pair)
            tagged_pair["_input_idx"] = input_key
            merged.append(tagged_pair)
            accepted_for_input += 1
            pair_type_counts[current_pair_type] += 1

    if args.shuffle:
        rng.shuffle(merged)
    if args.max_pairs is not None:
        merged = merged[: args.max_pairs]
    merged, skipped_input_fraction = apply_input_fraction_caps(
        merged,
        input_fraction_limits,
    )
    merged, skipped_pair_type_fraction = apply_pair_type_fraction_caps(
        merged,
        pair_type_max_fractions,
    )
    merged = [strip_internal_fields(pair) for pair in merged]
    pair_type_counts = Counter(pair_type(pair) for pair in merged)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    pair_types = Counter(pair_type(pair) for pair in merged)
    summary = {
        "output": str(output),
        "pairs": len(merged),
        "pair_types": dict(pair_types),
        "skipped_duplicates": skipped_duplicates,
        "skipped_input_limit": dict(skipped_input_limit),
        "skipped_input_fraction": dict(skipped_input_fraction),
        "skipped_pair_type_limit": dict(skipped_pair_type_limit),
        "skipped_pair_type_fraction": dict(skipped_pair_type_fraction),
        "skipped_reward_margin": dict(skipped_reward_margin),
        "skipped_sql_distance": dict(skipped_sql_distance),
        "min_reward_margin": args.min_reward_margin,
        "max_sql_edit_distance_ratio": args.max_sql_edit_distance_ratio,
        "accepted_sql_edit_distance_avg": round(sum(edit_distance_stats) / len(edit_distance_stats), 4)
        if edit_distance_stats
        else None,
        "input_limits": input_limits,
        "input_fraction_limits": input_fraction_limits,
        "pair_type_limits": pair_type_limits,
        "pair_type_max_fractions": pair_type_max_fractions,
        "inputs": args.inputs,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
