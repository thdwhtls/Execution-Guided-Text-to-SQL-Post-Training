import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from text2sql_trajectory_builder import (  # noqa: E402
    build_initial_prompt,
    build_schema_text,
    clean_sql,
    execute_sql,
    execution_match,
    get_field,
    get_schema_catalog,
    open_readonly_sqlite,
    read_json_or_jsonl,
    resolve_db_path,
    score_candidate,
)


ERROR_TYPES = [
    "missing_join",
    "wrong_aggregation",
    "wrong_condition",
    "wrong_group_by",
    "wrong_order_limit",
    "wrong_column",
    "wrong_table",
]

EXECUTABLE_WRONG_ERROR_TYPES = {
    "missing_join",
    "wrong_aggregation",
    "wrong_condition",
    "wrong_group_by",
    "wrong_order_limit",
}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_columns(catalog: Dict[str, List[str]]) -> List[str]:
    return [column for columns in catalog.values() for column in columns]


def replace_first_table(sql: str, replacement: str) -> Optional[str]:
    pattern = re.compile(r"\b(FROM|JOIN)\s+([`\"\[]?[A-Za-z_][A-Za-z0-9_]*[`\"\]]?)", re.IGNORECASE)
    match = pattern.search(sql)
    if not match:
        return None
    return sql[: match.start(2)] + replacement + sql[match.end(2) :]


def replace_first_column(sql: str, replacement: str) -> Optional[str]:
    dotted = re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)\b", sql)
    if dotted:
        return sql[: dotted.start(1)] + replacement + sql[dotted.end(1) :]

    select_match = re.search(r"\bSELECT\s+(.+?)\s+\bFROM\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if not select_match:
        return None
    expr = select_match.group(1)
    bare = re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr)
    if not bare:
        return None
    start = select_match.start(1) + bare.start()
    end = select_match.start(1) + bare.end()
    return sql[:start] + replacement + sql[end:]


def remove_first_join(sql: str) -> Optional[str]:
    pattern = re.compile(
        r"\s+\bJOIN\b\s+[A-Za-z_][A-Za-z0-9_]*(?:\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_]*)?\s+\bON\b\s+.*?(?=\s+\b(?:JOIN|WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT)\b|$)",
        re.IGNORECASE | re.DOTALL,
    )
    mutated, count = pattern.subn("", sql, count=1)
    return mutated if count else None


def replace_aggregation(sql: str) -> Optional[str]:
    replacements = {
        "COUNT": "SUM",
        "SUM": "AVG",
        "AVG": "MAX",
        "MAX": "MIN",
        "MIN": "MAX",
    }
    match = re.search(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", sql, flags=re.IGNORECASE)
    if not match:
        return None
    old = match.group(1).upper()
    new = replacements[old]
    return sql[: match.start(1)] + new + sql[match.end(1) :]


def perturb_condition(sql: str) -> Optional[str]:
    string_match = re.search(r"'([^']*)'", sql)
    if string_match:
        return sql[: string_match.start()] + "'__wrong_value__'" + sql[string_match.end() :]

    number_match = re.search(r"(?<![A-Za-z_])\b\d+(?:\.\d+)?\b", sql)
    if number_match:
        return sql[: number_match.start()] + "999999" + sql[number_match.end() :]

    where_match = re.search(r"\bWHERE\b", sql, flags=re.IGNORECASE)
    if where_match:
        return sql + " AND 1 = 0"
    return None


def remove_group_by(sql: str) -> Optional[str]:
    pattern = re.compile(r"\s+\bGROUP\s+BY\b\s+.*?(?=\s+\b(?:ORDER\s+BY|LIMIT)\b|$)", re.IGNORECASE | re.DOTALL)
    mutated, count = pattern.subn("", sql, count=1)
    return mutated if count else None


def perturb_order_limit(sql: str) -> Optional[str]:
    if re.search(r"\bORDER\s+BY\b", sql, flags=re.IGNORECASE):
        if re.search(r"\bDESC\b", sql, flags=re.IGNORECASE):
            return re.sub(r"\bDESC\b", "ASC", sql, count=1, flags=re.IGNORECASE)
        if re.search(r"\bASC\b", sql, flags=re.IGNORECASE):
            return re.sub(r"\bASC\b", "DESC", sql, count=1, flags=re.IGNORECASE)
        return sql + " DESC"
    if re.search(r"\bLIMIT\s+\d+", sql, flags=re.IGNORECASE):
        return re.sub(r"\bLIMIT\s+\d+", "LIMIT 1", sql, count=1, flags=re.IGNORECASE)
    return sql + " LIMIT 1"


def mutate_sql(sql: str, error_type: str, catalog: Dict[str, List[str]], rng: random.Random) -> Optional[str]:
    sql = clean_sql(sql)
    tables = list(catalog.keys())
    columns = flatten_columns(catalog)

    if error_type == "wrong_table":
        return replace_first_table(sql, "__wrong_table__")
    if error_type == "wrong_column":
        return replace_first_column(sql, "__wrong_column__")
    if error_type == "missing_join":
        return remove_first_join(sql)
    if error_type == "wrong_aggregation":
        return replace_aggregation(sql)
    if error_type == "wrong_condition":
        return perturb_condition(sql)
    if error_type == "wrong_group_by":
        return remove_group_by(sql)
    if error_type == "wrong_order_limit":
        return perturb_order_limit(sql)

    if tables or columns:
        rng.choice(tables + columns)
    return None


def make_pair(prompt: str, chosen_sql: str, rejected: Dict[str, Any], example: Dict[str, Any], error_type: str) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "chosen": f"SQL: {chosen_sql}",
        "rejected": f"SQL: {rejected['sql']}",
        "meta": {
            "pair_type": f"synthetic_{error_type}",
            "db_id": example.get("db_id") or example.get("database_id"),
            "question": get_field(example, ["question", "utterance", "nl"], required=False),
            "gold_sql": get_field(example, ["query", "sql", "gold_sql"], required=False),
            "chosen_reward": 1.0,
            "rejected_reward": rejected["reward"],
            "chosen_turn": 1,
            "rejected_turn": 1,
            "rejected_error": rejected.get("error", ""),
            "rejected_cost_flags": rejected.get("cost_flags", {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic erroneous SQL samples and DPO pairs.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--db_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N examples before applying --limit.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_errors_per_example", type=int, default=4)
    parser.add_argument("--timeout_sec", type=float, default=5.0)
    parser.add_argument("--schema_mode", choices=["full", "retrieved", "bm25", "bm25_fk"], default="retrieved")
    parser.add_argument("--top_k_tables", type=int, default=6)
    parser.add_argument("--fk_hops", type=int, default=1)
    parser.add_argument(
        "--prefer_executable_wrong",
        action="store_true",
        help="Prefer synthetic negatives that execute but return the wrong result.",
    )
    parser.add_argument(
        "--max_schema_error_fraction",
        type=float,
        default=1.0,
        help=(
            "Maximum fraction of accepted synthetic negatives that are not executable. "
            "Use with --prefer_executable_wrong to keep shallow schema-error negatives bounded."
        ),
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if not 0.0 <= args.max_schema_error_fraction <= 1.0:
        raise ValueError("--max_schema_error_fraction must be in [0, 1].")

    rng = random.Random(args.seed)
    examples = read_json_or_jsonl(args.dataset_path)
    if args.offset < 0:
        raise ValueError("--offset must be non-negative.")
    if args.offset:
        examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]

    output_dir = Path(args.output_dir)
    sample_rows = []
    dpo_pairs = []
    skipped = Counter()
    error_type_counts = Counter()
    executable_wrong = 0
    schema_or_syntax_wrong = 0

    for example in examples:
        try:
            question = get_field(example, ["question", "utterance", "nl"])
            gold_sql = get_field(example, ["query", "sql", "gold_sql"])
            db_path = resolve_db_path(example, args.db_root)

            with open_readonly_sqlite(db_path) as conn:
                catalog = get_schema_catalog(conn)
                schema_text, _ = build_schema_text(conn, question, args)
                prompt = build_initial_prompt(question, schema_text)
                gold_result = execute_sql(conn, gold_sql, args.timeout_sec)
                if not gold_result.ok:
                    skipped["bad_gold"] += 1
                    continue

                if args.prefer_executable_wrong:
                    priority_types = list(EXECUTABLE_WRONG_ERROR_TYPES)
                    fallback_types = [item for item in ERROR_TYPES if item not in EXECUTABLE_WRONG_ERROR_TYPES]
                    rng.shuffle(priority_types)
                    rng.shuffle(fallback_types)
                    error_types = priority_types + fallback_types
                else:
                    error_types = ERROR_TYPES[:]
                    rng.shuffle(error_types)
                added = 0
                seen_sql = {clean_sql(gold_sql)}

                for error_type in error_types:
                    if added >= args.max_errors_per_example:
                        break
                    mutated_sql = mutate_sql(gold_sql, error_type, catalog, rng)
                    if not mutated_sql:
                        continue
                    mutated_sql = clean_sql(mutated_sql)
                    if mutated_sql in seen_sql:
                        continue
                    seen_sql.add(mutated_sql)

                    scored = score_candidate(
                        conn=conn,
                        candidate_sql=mutated_sql,
                        gold_result=gold_result,
                        turn=1,
                        max_turns=1,
                        timeout_sec=args.timeout_sec,
                    )
                    if scored["correct"]:
                        skipped[f"{error_type}_still_correct"] += 1
                        continue
                    if args.prefer_executable_wrong and not scored["ok"]:
                        next_total = len(sample_rows) + 1
                        next_schema_errors = schema_or_syntax_wrong + 1
                        if next_schema_errors / next_total > args.max_schema_error_fraction:
                            skipped[f"{error_type}_schema_error_fraction"] += 1
                            continue

                    error_type_counts[error_type] += 1
                    executable_wrong += int(scored["ok"])
                    schema_or_syntax_wrong += int(not scored["ok"])
                    added += 1

                    sample_rows.append(
                        {
                            "db_id": example.get("db_id") or example.get("database_id"),
                            "question": question,
                            "gold_sql": gold_sql,
                            "error_type": error_type,
                            "mutated_sql": mutated_sql,
                            "score": scored,
                        }
                    )
                    dpo_pairs.append(make_pair(prompt, gold_sql, scored, example, error_type))
        except Exception as exc:
            skipped[type(exc).__name__] += 1

    summary = {
        "total_examples": len(examples),
        "synthetic_errors": len(sample_rows),
        "dpo_pairs": len(dpo_pairs),
        "error_type_counts": dict(error_type_counts),
        "executable_wrong": executable_wrong,
        "schema_or_syntax_wrong": schema_or_syntax_wrong,
        "skipped": dict(skipped),
        "args": vars(args),
    }

    write_jsonl(output_dir / "synthetic_error_sql.jsonl", sample_rows)
    write_json(output_dir / "dpo_pairs.json", dpo_pairs)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
