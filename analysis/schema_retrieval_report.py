import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


CLAUSE_ENDERS = {
    "where",
    "group",
    "order",
    "having",
    "limit",
    "union",
    "intersect",
    "except",
    "minus",
    "qualify",
}

JOIN_KEYWORDS = {
    "join",
    "inner",
    "left",
    "right",
    "full",
    "cross",
    "outer",
    "natural",
}

SKIP_TABLE_TOKENS = {
    "select",
    "values",
    "unnest",
    "lateral",
}


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_identifier(identifier: str) -> str:
    identifier = identifier.strip()
    identifier = identifier.strip("`\"[]")
    if "." in identifier:
        identifier = identifier.split(".")[-1]
    return identifier.lower()


def strip_sql_literals(sql: str) -> str:
    sql = re.sub(r"--.*?(?=\n|$)", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:''|[^'])*'", " ", sql)
    sql = re.sub(r'"(?:\"\"|[^"])*"', lambda m: m.group(0), sql)
    return sql


def tokenize_sql(sql: str) -> List[str]:
    sql = strip_sql_literals(sql)
    return re.findall(
        r"`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_.$]*|[(),]",
        sql,
    )


def get_catalog(db_path: Optional[str]) -> Optional[Dict[str, str]]:
    if not db_path or not Path(db_path).exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
    except sqlite3.Error:
        return None
    return {normalize_identifier(row[0]): row[0] for row in rows}


def extract_gold_tables(gold_sql: str, catalog: Optional[Dict[str, str]] = None) -> List[str]:
    tokens = tokenize_sql(gold_sql)
    found: List[str] = []
    found_keys: Set[str] = set()
    expect_table = False
    in_from_clause = False

    for token in tokens:
        lower = token.lower()

        if lower in CLAUSE_ENDERS:
            in_from_clause = False
            expect_table = False
            continue

        if lower == "from":
            in_from_clause = True
            expect_table = True
            continue

        if lower == "join" or (in_from_clause and lower in JOIN_KEYWORDS):
            in_from_clause = True
            expect_table = lower == "join"
            continue

        if in_from_clause and token == ",":
            expect_table = True
            continue

        if not expect_table:
            continue

        if token == "(":
            expect_table = False
            continue

        normalized = normalize_identifier(token)
        if not normalized or normalized in SKIP_TABLE_TOKENS:
            expect_table = False
            continue

        if catalog is not None:
            if normalized not in catalog:
                expect_table = False
                continue
            table_name = catalog[normalized]
        else:
            table_name = normalized

        if normalized not in found_keys:
            found_keys.add(normalized)
            found.append(table_name)
        expect_table = False

    return found


def parse_run_arg(item: str) -> Tuple[str, str]:
    if "=" in item:
        label, path = item.split("=", 1)
        return label, path
    path = item
    return Path(path).parent.name or Path(path).stem, path


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_float(value: float) -> str:
    return f"{value:.2f}"


def analyze_run(label: str, path: str) -> Dict[str, Any]:
    catalog_cache: Dict[str, Optional[Dict[str, str]]] = {}
    rows = list(read_jsonl(path))
    total_gold_tables = 0
    hit_gold_tables = 0
    selected_counts: List[int] = []
    gold_counts: List[int] = []
    all_recalled = 0
    any_recalled = 0
    no_gold_table = 0
    missed_examples: List[Dict[str, Any]] = []
    missed_table_counts: Counter[str] = Counter()
    selected_table_sets: List[List[str]] = []

    for idx, traj in enumerate(rows):
        db_path = traj.get("db_path")
        if db_path not in catalog_cache:
            catalog_cache[db_path] = get_catalog(db_path)

        gold_tables = extract_gold_tables(traj.get("gold_sql", ""), catalog_cache[db_path])
        retrieved_tables = traj.get("retrieved_tables") or []
        retrieved_keys = {normalize_identifier(table) for table in retrieved_tables}
        gold_keys = {normalize_identifier(table) for table in gold_tables}
        hits = gold_keys & retrieved_keys
        missed = gold_keys - retrieved_keys

        selected_counts.append(len(retrieved_tables))
        selected_table_sets.append(sorted(retrieved_keys))
        gold_counts.append(len(gold_tables))
        total_gold_tables += len(gold_keys)
        hit_gold_tables += len(hits)

        if not gold_keys:
            no_gold_table += 1
            continue

        if hits:
            any_recalled += 1
        if not missed:
            all_recalled += 1
        else:
            missed_names = [table for table in gold_tables if normalize_identifier(table) in missed]
            missed_table_counts.update(missed_names)
            missed_examples.append(
                {
                    "index": idx,
                    "db_id": traj.get("db_id"),
                    "question": traj.get("question"),
                    "status": traj.get("status"),
                    "gold_sql": traj.get("gold_sql"),
                    "gold_tables": gold_tables,
                    "retrieved_tables": retrieved_tables,
                    "missed_tables": missed_names,
                }
            )

    evaluable = len(rows) - no_gold_table
    return {
        "label": label,
        "path": path,
        "total": len(rows),
        "evaluable": evaluable,
        "gold_table_recall": round(hit_gold_tables / total_gold_tables, 4) if total_gold_tables else 0.0,
        "example_all_tables_recalled": round(all_recalled / evaluable, 4) if evaluable else 0.0,
        "example_any_table_recalled": round(any_recalled / evaluable, 4) if evaluable else 0.0,
        "missed_examples": len(missed_examples),
        "avg_selected_tables": round(sum(selected_counts) / len(selected_counts), 4) if selected_counts else 0.0,
        "avg_gold_tables": round(sum(gold_counts) / len(gold_counts), 4) if gold_counts else 0.0,
        "total_gold_tables": total_gold_tables,
        "hit_gold_tables": hit_gold_tables,
        "no_gold_table": no_gold_table,
        "top_missed_tables": dict(missed_table_counts.most_common(20)),
        "missed_example_rows": missed_examples,
        "_selected_table_sets": selected_table_sets,
    }


def truncate(text: Any, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_markdown(results: Sequence[Dict[str, Any]], max_examples: int, max_chars: int) -> str:
    lines = ["# Schema Retrieval Report", ""]
    lines.append("## Retrieval Mode Comparison")
    lines.append("")
    lines.append(
        "| Run | total | gold table recall | all-table recall | any-table recall | avg selected tables | avg gold tables | missed examples |"
    )
    lines.append("| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result["label"],
                    str(result["total"]),
                    percent(result["gold_table_recall"]),
                    percent(result["example_all_tables_recalled"]),
                    percent(result["example_any_table_recalled"]),
                    format_float(result["avg_selected_tables"]),
                    format_float(result["avg_gold_tables"]),
                    str(result["missed_examples"]),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Top Missed Tables")
    lines.append("")
    for result in results:
        missed = result["top_missed_tables"]
        rendered = ", ".join(f"`{table}`:{count}" for table, count in missed.items()) or "-"
        lines.append(f"- {result['label']}: {rendered}")

    if len(results) >= 2:
        lines.append("")
        lines.append("## Pairwise Selected-Set Differences")
        lines.append("")
        lines.append("| Pair | comparable examples | changed examples | avg Jaccard |")
        lines.append("| --- | :---: | :---: | :---: |")
        for left_idx in range(len(results)):
            for right_idx in range(left_idx + 1, len(results)):
                left = results[left_idx]
                right = results[right_idx]
                comparable = min(len(left["_selected_table_sets"]), len(right["_selected_table_sets"]))
                changed = 0
                jaccards: List[float] = []
                for idx in range(comparable):
                    left_set = set(left["_selected_table_sets"][idx])
                    right_set = set(right["_selected_table_sets"][idx])
                    if left_set != right_set:
                        changed += 1
                    union = left_set | right_set
                    jaccards.append(len(left_set & right_set) / len(union) if union else 1.0)
                avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
                lines.append(
                    f"| {left['label']} vs {right['label']} | {comparable} | {changed} | {avg_jaccard:.4f} |"
                )

    lines.append("")
    lines.append("## Missed Table Examples")
    lines.append("")
    for result in results:
        lines.append(f"### {result['label']}")
        lines.append("")
        examples = result["missed_example_rows"][:max_examples]
        if not examples:
            lines.append("No missed gold tables.")
            lines.append("")
            continue
        for idx, example in enumerate(examples, start=1):
            lines.append(f"#### Case {idx}")
            lines.append("")
            lines.append(f"- db_id: `{example.get('db_id')}`")
            lines.append(f"- status: `{example.get('status')}`")
            lines.append(f"- question: {truncate(example.get('question'), max_chars)}")
            lines.append(f"- gold_tables: `{', '.join(example.get('gold_tables', []))}`")
            lines.append(f"- retrieved_tables: `{', '.join(example.get('retrieved_tables', []))}`")
            lines.append(f"- missed_tables: `{', '.join(example.get('missed_tables', []))}`")
            lines.append(f"- gold_sql: `{truncate(example.get('gold_sql'), max_chars)}`")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report gold-table recall for schema retrieval outputs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Trajectory paths as LABEL=path or plain path.",
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--max_chars", type=int, default=260)
    args = parser.parse_args()

    results = [analyze_run(*parse_run_arg(item)) for item in args.runs]
    printable = [
        {
            key: value
            for key, value in result.items()
            if key not in {"missed_example_rows", "_selected_table_sets"}
        }
        for result in results
    ]
    print(json.dumps(printable, ensure_ascii=False, indent=2))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        json_results = [
            {key: value for key, value in result.items() if key != "_selected_table_sets"}
            for result in results
        ]
        output_json.write_text(json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(
            build_markdown(results, args.max_examples, args.max_chars),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
