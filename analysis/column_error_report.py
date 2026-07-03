import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SQL_KEYWORDS = {
    "as",
    "on",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "cross",
    "outer",
    "natural",
    "group",
    "order",
    "having",
    "limit",
    "union",
    "intersect",
    "except",
    "select",
    "from",
}


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_identifier(identifier: str) -> str:
    identifier = str(identifier or "").strip().strip("`\"[]")
    if "." in identifier:
        identifier = identifier.split(".")[-1]
    return identifier.lower()


def strip_sql_literals(sql: str) -> str:
    sql = re.sub(r"--.*?(?=\n|$)", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:''|[^'])*'", " ", sql)
    sql = re.sub(r'"(?:\"\"|[^"])*"', " ", sql)
    return sql


def tokenize_sql(sql: str) -> List[str]:
    return re.findall(
        r"`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_.$]*|[(),]",
        strip_sql_literals(sql),
    )


def parse_schema_from_prompt(prompt: str) -> Dict[str, List[str]]:
    catalog: Dict[str, List[str]] = {}
    for line in str(prompt or "").splitlines():
        line = line.strip()
        if not line.startswith("Table ") or ":" not in line:
            continue
        table_part, column_part = line.split(":", 1)
        table = table_part.replace("Table ", "", 1).strip()
        columns = []
        for column_desc in column_part.split(","):
            column_desc = column_desc.strip()
            if not column_desc:
                continue
            columns.append(column_desc.split()[0].strip("`\"[]"))
        catalog[table] = columns
    return catalog


def column_owner_index(catalog: Dict[str, List[str]]) -> Dict[str, List[str]]:
    owners: Dict[str, List[str]] = {}
    for table, columns in catalog.items():
        for column in columns:
            owners.setdefault(normalize_identifier(column), []).append(table)
    return owners


def parse_table_aliases(sql: str) -> Dict[str, str]:
    tokens = tokenize_sql(sql)
    aliases: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        lower = tokens[i].lower()
        if lower not in {"from", "join"}:
            i += 1
            continue

        i += 1
        if i >= len(tokens) or tokens[i] == "(":
            continue

        table = tokens[i].strip("`\"[]")
        table_key = normalize_identifier(table)
        if not table_key or table_key in SQL_KEYWORDS:
            i += 1
            continue

        aliases[table_key] = table
        i += 1
        if i < len(tokens) and tokens[i].lower() == "as":
            i += 1
        if i < len(tokens):
            alias = tokens[i].strip("`\"[]")
            alias_key = normalize_identifier(alias)
            if alias_key and alias_key not in SQL_KEYWORDS and alias not in {",", "(", ")"}:
                aliases[alias_key] = table
    return aliases


def referenced_table_names(sql: str) -> Set[str]:
    return {
        normalize_identifier(table)
        for table in parse_table_aliases(sql).values()
        if normalize_identifier(table)
    }


def extract_gold_tables(gold_sql: str) -> Set[str]:
    tokens = tokenize_sql(gold_sql)
    tables: Set[str] = set()
    expect_table = False
    for token in tokens:
        lower = token.lower()
        if lower in {"where", "group", "order", "having", "limit", "union", "intersect", "except"}:
            expect_table = False
            continue
        if lower in {"from", "join"}:
            expect_table = True
            continue
        if expect_table and token == ",":
            continue
        if expect_table:
            if token != "(":
                table = normalize_identifier(token)
                if table and table not in SQL_KEYWORDS:
                    tables.add(table)
            expect_table = False
    return tables


def parse_missing_identifier(error: str) -> Optional[str]:
    match = re.search(r"no such column:\s*([A-Za-z_][A-Za-z0-9_.$]*)", error or "", flags=re.IGNORECASE)
    return match.group(1) if match else None


def qualified_column_refs(sql: str) -> List[Tuple[str, str]]:
    return [
        (alias, column)
        for alias, column in re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
            strip_sql_literals(sql),
        )
    ]


def diagnose_column_error(traj: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    sql = candidate.get("sql") or ""
    error = candidate.get("error") or ""
    catalog = parse_schema_from_prompt(traj.get("prompt", ""))
    catalog_by_key = {normalize_identifier(table): table for table in catalog}
    columns_by_table_key = {
        normalize_identifier(table): {normalize_identifier(column): column for column in columns}
        for table, columns in catalog.items()
    }
    owners = column_owner_index(catalog)
    aliases = parse_table_aliases(sql)
    gold_tables = extract_gold_tables(traj.get("gold_sql", ""))
    sql_tables = referenced_table_names(sql)
    extra_tables = sorted(table for table in sql_tables if table not in gold_tables)

    missing_identifier = parse_missing_identifier(error)
    invalid_refs = []
    tags: Set[str] = set()

    refs = qualified_column_refs(sql)
    if missing_identifier and "." in missing_identifier:
        alias, column = missing_identifier.split(".", 1)
        refs = [(alias, column), *[ref for ref in refs if f"{ref[0]}.{ref[1]}".lower() != missing_identifier.lower()]]
    elif missing_identifier:
        refs = [("", missing_identifier), *refs]

    seen_refs = set()
    for alias, column in refs:
        ref_key = (normalize_identifier(alias), normalize_identifier(column))
        if ref_key in seen_refs:
            continue
        seen_refs.add(ref_key)

        alias_key = normalize_identifier(alias)
        column_key = normalize_identifier(column)
        owner_tables = owners.get(column_key, [])
        alias_table = aliases.get(alias_key) if alias_key else ""
        alias_table_key = normalize_identifier(alias_table)
        alias_columns = columns_by_table_key.get(alias_table_key, {})

        if alias_key and alias_key not in aliases:
            tags.add("extra_join_alias_mismatch")
            invalid_refs.append(
                {
                    "ref": f"{alias}.{column}",
                    "issue": "undefined_alias",
                    "alias": alias,
                    "alias_table": None,
                    "candidate_owner_tables": owner_tables,
                }
            )
        elif alias_key and alias_table_key in catalog_by_key and column_key not in alias_columns:
            if owner_tables:
                tags.add("wrong_column_owner")
                tags.add("wrong_table_column_mapping")
                if extra_tables:
                    tags.add("extra_join_alias_mismatch")
                issue = "column_belongs_to_other_table"
            else:
                tags.add("hallucinated_column")
                issue = "column_not_in_selected_schema"
            invalid_refs.append(
                {
                    "ref": f"{alias}.{column}",
                    "issue": issue,
                    "alias": alias,
                    "alias_table": alias_table,
                    "candidate_owner_tables": owner_tables,
                }
            )
        elif not alias_key and column_key not in owners:
            tags.add("hallucinated_column")
            invalid_refs.append(
                {
                    "ref": column,
                    "issue": "column_not_in_selected_schema",
                    "alias": None,
                    "alias_table": None,
                    "candidate_owner_tables": [],
                }
            )

    if not tags and "no such column" in error.lower():
        tags.add("unclassified_no_such_column")

    return {
        "tags": sorted(tags),
        "missing_identifier": missing_identifier,
        "invalid_refs": invalid_refs,
        "aliases": aliases,
        "sql_tables": sorted(sql_tables),
        "gold_tables": sorted(gold_tables),
        "extra_tables": extra_tables,
        "available_columns": catalog,
    }


def truncate(text: Any, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def first_no_such_column_candidate(traj: Dict[str, Any]) -> Dict[str, Any]:
    for candidate in traj.get("candidates", []):
        if "no such column" in (candidate.get("error") or "").lower():
            return candidate
    return {}


def analyze_run(label: str, path: str, max_examples: int) -> Dict[str, Any]:
    tag_counts: Counter[str] = Counter()
    invalid_ref_counts: Counter[str] = Counter()
    owner_suggestion_counts: Counter[str] = Counter()
    no_such_column_candidates = 0
    first_turn_no_such_column = 0
    affected_examples = 0
    examples = []

    for traj in read_jsonl(path):
        candidate = first_no_such_column_candidate(traj)
        if not candidate:
            continue
        affected_examples += 1

        for item in traj.get("candidates", []):
            if "no such column" not in (item.get("error") or "").lower():
                continue
            no_such_column_candidates += 1
            if item.get("turn") == 1:
                first_turn_no_such_column += 1

        diagnosis = diagnose_column_error(traj, candidate)
        tag_counts.update(diagnosis["tags"])
        for invalid_ref in diagnosis["invalid_refs"]:
            invalid_ref_counts[invalid_ref["ref"]] += 1
            for owner in invalid_ref.get("candidate_owner_tables") or []:
                owner_suggestion_counts[f"{invalid_ref['ref']} -> {owner}"] += 1

        if len(examples) < max_examples:
            examples.append(
                {
                    "db_id": traj.get("db_id"),
                    "status": traj.get("status"),
                    "question": traj.get("question"),
                    "gold_sql": traj.get("gold_sql"),
                    "sql": candidate.get("sql"),
                    "error": candidate.get("error"),
                    "turn": candidate.get("turn"),
                    "tags": diagnosis["tags"],
                    "invalid_refs": diagnosis["invalid_refs"],
                    "sql_tables": diagnosis["sql_tables"],
                    "gold_tables": diagnosis["gold_tables"],
                    "extra_tables": diagnosis["extra_tables"],
                }
            )

    return {
        "label": label,
        "path": path,
        "affected_examples": affected_examples,
        "no_such_column_candidates": no_such_column_candidates,
        "first_turn_no_such_column": first_turn_no_such_column,
        "tag_counts": dict(tag_counts),
        "top_invalid_refs": dict(invalid_ref_counts.most_common(20)),
        "top_owner_suggestions": dict(owner_suggestion_counts.most_common(20)),
        "examples": examples,
    }


def parse_run_arg(item: str) -> Tuple[str, str]:
    if "=" in item:
        return tuple(item.split("=", 1))  # type: ignore[return-value]
    return Path(item).parent.name or Path(item).stem, item


def build_markdown(results: Sequence[Dict[str, Any]], max_chars: int) -> str:
    lines = ["# Column Error Report", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("| Run | affected examples | no_such_column candidates | first-turn no_such_column | tag counts |")
    lines.append("| --- | :---: | :---: | :---: | --- |")
    for result in results:
        tags = ", ".join(f"`{tag}`:{count}" for tag, count in result["tag_counts"].items()) or "-"
        lines.append(
            f"| {result['label']} | {result['affected_examples']} | "
            f"{result['no_such_column_candidates']} | {result['first_turn_no_such_column']} | {tags} |"
        )

    lines.append("")
    lines.append("## Top Invalid References")
    lines.append("")
    for result in results:
        refs = ", ".join(f"`{ref}`:{count}" for ref, count in result["top_invalid_refs"].items()) or "-"
        lines.append(f"- {result['label']}: {refs}")

    lines.append("")
    lines.append("## Top Owner Suggestions")
    lines.append("")
    for result in results:
        suggestions = ", ".join(
            f"`{suggestion}`:{count}" for suggestion, count in result["top_owner_suggestions"].items()
        ) or "-"
        lines.append(f"- {result['label']}: {suggestions}")

    lines.append("")
    lines.append("## Examples")
    lines.append("")
    for result in results:
        lines.append(f"### {result['label']}")
        lines.append("")
        for idx, example in enumerate(result["examples"], start=1):
            tags = ", ".join(example["tags"]) or "-"
            invalid_refs = "; ".join(
                f"{item['ref']} ({item['issue']}; alias_table={item['alias_table']}; owners={item['candidate_owner_tables']})"
                for item in example["invalid_refs"]
            ) or "-"
            lines.append(f"#### Case {idx}")
            lines.append("")
            lines.append(f"- db_id: `{example.get('db_id')}`")
            lines.append(f"- status: `{example.get('status')}`, turn: `{example.get('turn')}`, tags: `{tags}`")
            lines.append(f"- question: {truncate(example.get('question'), max_chars)}")
            lines.append(f"- error: `{truncate(example.get('error'), max_chars)}`")
            lines.append(f"- invalid_refs: {truncate(invalid_refs, max_chars)}")
            lines.append(f"- gold_tables: `{', '.join(example.get('gold_tables', []))}`")
            lines.append(f"- sql_tables: `{', '.join(example.get('sql_tables', []))}`")
            lines.append(f"- extra_tables: `{', '.join(example.get('extra_tables', []))}`")
            lines.append(f"- sql: `{truncate(example.get('sql'), max_chars)}`")
            lines.append(f"- gold_sql: `{truncate(example.get('gold_sql'), max_chars)}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose no_such_column Text-to-SQL errors.")
    parser.add_argument("--runs", nargs="+", required=True, help="Trajectory paths as LABEL=path or plain path.")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--max_chars", type=int, default=280)
    args = parser.parse_args()

    results = [analyze_run(*parse_run_arg(item), max_examples=args.max_examples) for item in args.runs]
    printable = [{key: value for key, value in result.items() if key != "examples"} for result in results]
    print(json.dumps(printable, ensure_ascii=False, indent=2))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown(results, args.max_chars), encoding="utf-8")


if __name__ == "__main__":
    main()
