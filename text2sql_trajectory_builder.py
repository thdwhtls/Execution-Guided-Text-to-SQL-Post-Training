import argparse
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_MAX_TURNS = 3
DEFAULT_NUM_SAMPLES = 1
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9
DEFAULT_TIMEOUT_SEC = 5.0

SCHEMA_ERROR_MARKERS = (
    "no such table",
    "no such column",
    "ambiguous column name",
    "unknown database",
)

@dataclass
class ExecutionResult:
    ok: bool
    rows: List[Tuple[Any, ...]]
    error: str = ""
    elapsed_ms: float = 0.0


@dataclass
class CostResult:
    reward: float
    explain: List[str]
    flags: Dict[str, int]


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of examples.")
        return data

    examples = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return examples


def get_field(example: Dict[str, Any], names: Iterable[str], required: bool = True) -> Optional[str]:
    for name in names:
        value = example.get(name)
        if value is not None and str(value).strip():
            return str(value)
    if required:
        raise KeyError(f"Missing required field. Tried: {', '.join(names)}")
    return None


def resolve_db_path(example: Dict[str, Any], db_root: Optional[str]) -> str:
    explicit = get_field(example, ["db_path", "database_path", "sqlite_path"], required=False)
    if explicit:
        return explicit

    db_id = get_field(example, ["db_id", "database_id"])
    if not db_root:
        raise ValueError("db_root is required when examples only contain db_id.")

    spider_path = Path(db_root) / db_id / f"{db_id}.sqlite"
    if spider_path.exists():
        return str(spider_path)

    flat_path = Path(db_root) / f"{db_id}.sqlite"
    if flat_path.exists():
        return str(flat_path)

    raise FileNotFoundError(f"Cannot find SQLite DB for db_id={db_id} under {db_root}")


def open_readonly_sqlite(db_path: str) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).absolute()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip()


def extract_sql(text: str) -> str:
    text = text.strip()

    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return clean_sql(fenced.group(1))

    sql_label = re.search(r"(?:^|\n)\s*SQL\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if sql_label:
        candidate = sql_label.group(1).strip()
        candidate = re.split(r"\n\s*(?:Explanation|Reasoning|Answer)\s*:", candidate, flags=re.IGNORECASE)[0]
        return clean_sql(candidate)

    select_match = re.search(r"\b(WITH|SELECT)\b.+", text, flags=re.IGNORECASE | re.DOTALL)
    if select_match:
        return clean_sql(select_match.group(0))

    return clean_sql(text)


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", text.lower())


def is_query_like(sql: str) -> bool:
    sql = clean_sql(sql).lstrip(" (")
    return bool(re.match(r"^(SELECT|WITH)\b", sql, flags=re.IGNORECASE))


def execute_sql(conn: sqlite3.Connection, sql: str, timeout_sec: float) -> ExecutionResult:
    sql = clean_sql(sql)
    if not is_query_like(sql):
        return ExecutionResult(False, [], "Only SELECT/WITH read queries are allowed.")

    deadline = time.time() + timeout_sec

    def progress_handler() -> int:
        return 1 if time.time() > deadline else 0

    start = time.time()
    conn.set_progress_handler(progress_handler, 1000)
    try:
        cursor = conn.execute(sql)
        rows = [tuple(row) for row in cursor.fetchall()]
        elapsed_ms = (time.time() - start) * 1000.0
        return ExecutionResult(True, rows, elapsed_ms=elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.time() - start) * 1000.0
        return ExecutionResult(False, [], str(exc), elapsed_ms=elapsed_ms)
    finally:
        conn.set_progress_handler(None, 0)


def online_guarded_signals(pred: ExecutionResult, candidate: Optional[Dict[str, Any]] = None) -> List[str]:
    signals: List[str] = []
    candidate = candidate or {}
    flags = candidate.get("cost_flags") or {}

    if not pred.ok:
        signals.append("executor_error")
    if pred.ok and len(pred.rows) == 0:
        signals.append("empty_result")
    if flags.get("syntax_valid") == 0:
        signals.append("syntax_guard_failed")
    if flags.get("schema_valid") == 0:
        signals.append("schema_guard_failed")
    if flags.get("executable") == 0:
        signals.append("not_executable")
    if flags.get("explain_error"):
        signals.append("explain_plan_error")
    if flags.get("cartesian"):
        signals.append("cartesian_join_risk")

    deduped: List[str] = []
    seen: Set[str] = set()
    for signal in signals:
        if signal not in seen:
            deduped.append(signal)
            seen.add(signal)
    return deduped


def should_attempt_repair(
    pred: ExecutionResult,
    repair_scope: str,
    candidate: Optional[Dict[str, Any]] = None,
) -> bool:
    if repair_scope == "verified":
        return True
    if not pred.ok:
        return True
    if repair_scope == "online_guarded":
        return bool(online_guarded_signals(pred, candidate))
    return False


def normalize_cell(value: Any) -> Any:
    if value is None:
        return "<NULL>"
    if isinstance(value, float):
        if math.isfinite(value):
            return round(value, 6)
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value).strip() if not isinstance(value, (int, float)) else value


def canonicalize_rows(rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    normalized = [tuple(normalize_cell(cell) for cell in row) for row in rows]
    return sorted(normalized, key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))


def execution_match(pred: ExecutionResult, gold: ExecutionResult) -> bool:
    if not pred.ok or not gold.ok:
        return False
    return canonicalize_rows(pred.rows) == canonicalize_rows(gold.rows)


def get_schema_text(conn: sqlite3.Connection, max_columns_per_table: int = 80) -> str:
    return get_schema_text_for_tables(conn, None, max_columns_per_table=max_columns_per_table)


def get_schema_catalog(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    catalog = {}
    for row in table_rows:
        table = row["name"]
        columns = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
        catalog[table] = [col["name"] for col in columns]
    return catalog


def normalize_identifier(name: str) -> str:
    name = str(name or "").strip().strip("`\"[]")
    if "." in name:
        name = name.split(".")[-1]
    return name.lower()


def get_column_owner_index(catalog: Dict[str, List[str]]) -> Dict[str, List[str]]:
    owners: Dict[str, List[str]] = defaultdict(list)
    for table, columns in catalog.items():
        for column in columns:
            owners[normalize_identifier(column)].append(table)
    return dict(owners)


def get_foreign_key_graph(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    graph = defaultdict(set)
    for table in get_schema_catalog(conn):
        fks = conn.execute(f"PRAGMA foreign_key_list({quote_identifier(table)})").fetchall()
        graph.setdefault(table, set())
        for fk in fks:
            ref_table = fk["table"]
            graph[table].add(ref_table)
            graph[ref_table].add(table)
    return {table: sorted(neighbors) for table, neighbors in graph.items()}


def get_schema_text_for_tables(
    conn: sqlite3.Connection,
    table_filter: Optional[Iterable[str]],
    max_columns_per_table: int = 80,
) -> str:
    selected = set(table_filter) if table_filter else None
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    lines = []
    for row in table_rows:
        table = row["name"]
        if selected is not None and table not in selected:
            continue
        columns = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
        fks = conn.execute(f"PRAGMA foreign_key_list({quote_identifier(table)})").fetchall()

        col_parts = []
        for col in columns[:max_columns_per_table]:
            name = col["name"]
            col_type = col["type"] or "TEXT"
            pk = " primary_key" if col["pk"] else ""
            col_parts.append(f"{name} {col_type}{pk}")

        lines.append(f"Table {table}: " + ", ".join(col_parts))
        for fk in fks:
            if selected is None or fk["table"] in selected:
                lines.append(
                    f"Foreign key: {table}.{fk['from']} -> {fk['table']}.{fk['to']}"
                )
    return "\n".join(lines)


def retrieve_schema_tables(conn: sqlite3.Connection, question: str, top_k: int) -> List[str]:
    catalog = get_schema_catalog(conn)
    if top_k <= 0 or top_k >= len(catalog):
        return list(catalog.keys())

    question_tokens = set(tokenize_text(question))
    scored = []
    for table, columns in catalog.items():
        schema_tokens = set(tokenize_text(table))
        for column in columns:
            schema_tokens.update(tokenize_text(column))
        overlap = len(question_tokens & schema_tokens)
        soft_hits = sum(
            1
            for qt in question_tokens
            for st in schema_tokens
            if len(qt) >= 4 and len(st) >= 4 and (qt in st or st in qt)
        )
        scored.append((overlap * 3 + soft_hits, table))

    ranked = [table for score, table in sorted(scored, key=lambda item: (-item[0], item[1]))]
    positive = [table for score, table in sorted(scored, key=lambda item: (-item[0], item[1])) if score > 0]
    return (positive or ranked)[:top_k]


def schema_document_tokens(table: str, columns: List[str]) -> List[str]:
    tokens = []
    tokens.extend(tokenize_text(table.replace("_", " ")))
    for column in columns:
        tokens.extend(tokenize_text(column.replace("_", " ")))
    return tokens


def retrieve_schema_tables_bm25(
    conn: sqlite3.Connection,
    question: str,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[str]:
    catalog = get_schema_catalog(conn)
    if top_k <= 0 or top_k >= len(catalog):
        return list(catalog.keys())

    docs = {
        table: schema_document_tokens(table, columns)
        for table, columns in catalog.items()
    }
    doc_count = len(docs)
    avgdl = sum(len(tokens) for tokens in docs.values()) / max(doc_count, 1)
    df = Counter()
    for tokens in docs.values():
        df.update(set(tokens))

    query_tokens = tokenize_text(question.replace("_", " "))
    query_counts = Counter(query_tokens)
    scored = []
    for table, tokens in docs.items():
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for token, query_weight in query_counts.items():
            if token not in tf:
                continue
            idf = math.log(1 + (doc_count - df[token] + 0.5) / (df[token] + 0.5))
            denom = tf[token] + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            score += query_weight * idf * (tf[token] * (k1 + 1)) / denom

        # Soft substring rescue helps natural-language variants like "singers" vs "singer".
        schema_terms = set(tokens)
        soft_hits = sum(
            1
            for qt in set(query_tokens)
            for st in schema_terms
            if len(qt) >= 4 and len(st) >= 4 and (qt in st or st in qt)
        )
        score += 0.15 * soft_hits
        scored.append((score, table))

    ranked = [table for score, table in sorted(scored, key=lambda item: (-item[0], item[1]))]
    positive = [table for score, table in sorted(scored, key=lambda item: (-item[0], item[1])) if score > 0]
    return (positive or ranked)[:top_k]


def expand_tables_by_foreign_keys(conn: sqlite3.Connection, tables: Iterable[str], hops: int) -> List[str]:
    if hops <= 0:
        return list(dict.fromkeys(tables))

    graph = get_foreign_key_graph(conn)
    selected = list(dict.fromkeys(tables))
    selected_set = set(selected)
    frontier = set(selected)
    for _ in range(hops):
        next_frontier = set()
        for table in frontier:
            for neighbor in graph.get(table, []):
                if neighbor not in selected_set:
                    selected_set.add(neighbor)
                    selected.append(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return selected


def build_schema_text(conn: sqlite3.Connection, question: str, args: argparse.Namespace) -> Tuple[str, List[str]]:
    if args.schema_mode == "full":
        catalog = get_schema_catalog(conn)
        return get_schema_text(conn), list(catalog.keys())

    if args.schema_mode == "bm25":
        selected_tables = retrieve_schema_tables_bm25(conn, question, args.top_k_tables)
    elif args.schema_mode == "bm25_fk":
        base_tables = retrieve_schema_tables_bm25(conn, question, args.top_k_tables)
        selected_tables = expand_tables_by_foreign_keys(conn, base_tables, args.fk_hops)
    else:
        selected_tables = retrieve_schema_tables(conn, question, args.top_k_tables)
    return get_schema_text_for_tables(conn, selected_tables), selected_tables


def quote_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def explain_query_plan(conn: sqlite3.Connection, sql: str) -> Tuple[List[str], str]:
    if not is_query_like(sql):
        return [], "Only SELECT/WITH read queries are allowed."
    try:
        rows = conn.execute(f"EXPLAIN QUERY PLAN {clean_sql(sql)}").fetchall()
        return [str(row["detail"]) for row in rows], ""
    except Exception as exc:
        return [], str(exc)


def cost_reward(conn: sqlite3.Connection, sql: str) -> CostResult:
    explain, error = explain_query_plan(conn, sql)
    if error:
        return CostResult(-0.10, [], {"explain_error": 1})

    flags = Counter()
    reward = 0.0
    access_ops = 0

    for detail in explain:
        upper = detail.upper()
        if "SCAN" in upper:
            access_ops += 1
            if "USING INDEX" not in upper and "USING COVERING INDEX" not in upper:
                flags["full_scan"] += 1
                reward -= 0.06
        if "SEARCH" in upper:
            access_ops += 1
            if "USING INDEX" in upper or "USING COVERING INDEX" in upper:
                flags["index_search"] += 1
                reward += 0.025
        if "USE TEMP B-TREE" in upper:
            flags["temp_btree"] += 1
            reward -= 0.04
        if "CARTESIAN" in upper:
            flags["cartesian"] += 1
            reward -= 0.10

    if access_ops >= 3:
        flags["multi_table"] = access_ops
        reward -= min(0.06, 0.015 * (access_ops - 2))

    reward = max(-0.25, min(0.10, reward))
    return CostResult(round(reward, 4), explain, dict(flags))


def schema_error(error: str) -> bool:
    lower = error.lower()
    return any(marker in lower for marker in SCHEMA_ERROR_MARKERS)


def rule_reward(
    pred_result: ExecutionResult,
    is_correct: bool,
    turn: int,
    max_turns: int,
) -> float:
    if is_correct:
        if turn == 1:
            return 1.0
        return round(max(0.0, 0.7 - 0.1 * (turn - 2)), 4)
    if turn == max_turns:
        return -0.5
    if pred_result.ok:
        return 0.2
    return -0.2


def validity_flags(sql: str, pred_result: ExecutionResult) -> Dict[str, int]:
    error = pred_result.error or ""
    is_select = is_query_like(sql)
    syntax_valid = int(is_select and "syntax error" not in error.lower())
    schema_valid = int(pred_result.ok or (syntax_valid and not schema_error(error)))
    return {
        "syntax_valid": syntax_valid,
        "schema_valid": schema_valid,
        "executable": int(pred_result.ok),
    }


def build_initial_prompt(question: str, schema_text: str) -> str:
    return (
        "You are a Text-to-SQL assistant. Generate a valid SQLite query for the question.\n"
        "Use only the provided schema. Return only one SQL query after `SQL:`.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Question: {question}\n"
        "SQL:"
    )


def build_repair_prompt(
    question: str,
    schema_text: str,
    previous_sql: str,
    feedback: str,
    turn: int,
    feedback_detail: str = "basic",
) -> str:
    minimal_guidance = ""
    if feedback_detail == "minimal":
        minimal_guidance = (
            "Column-Minimal Repair rules:\n"
            "- If the error indicates a column belongs to another table or the current SQL contains unnecessary joins, do not only patch aliases.\n"
            "- Regenerate the minimal SQL using only tables needed by the question.\n"
            "- Prefer removing extra joins when all required columns exist in one table.\n\n"
        )
    return (
        "You are debugging a SQLite query. Revise the previous SQL using the schema and feedback.\n"
        "Return only one corrected SQL query after `SQL:`.\n\n"
        f"{minimal_guidance}"
        f"Schema:\n{schema_text}\n\n"
        f"Question: {question}\n"
        f"Previous SQL:\n{previous_sql}\n"
        f"Execution feedback at turn {turn}:\n{feedback}\n"
        "SQL:"
    )


SQL_ALIAS_STOPWORDS = {
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


def strip_sql_literals(sql: str) -> str:
    sql = re.sub(r"--.*?(?=\n|$)", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:''|[^'])*'", " ", sql)
    sql = re.sub(r'"(?:\"\"|[^"])*"', " ", sql)
    return sql


def tokenize_sql_identifiers(sql: str) -> List[str]:
    return re.findall(
        r"`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_.$]*|[(),]",
        strip_sql_literals(sql),
    )


def parse_table_aliases(sql: str) -> Dict[str, str]:
    tokens = tokenize_sql_identifiers(sql)
    aliases: Dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        if tokens[idx].lower() not in {"from", "join"}:
            idx += 1
            continue

        idx += 1
        if idx >= len(tokens) or tokens[idx] == "(":
            continue

        table = tokens[idx].strip("`\"[]")
        table_key = normalize_identifier(table)
        if not table_key or table_key in SQL_ALIAS_STOPWORDS:
            idx += 1
            continue

        aliases[table_key] = table
        idx += 1

        if idx < len(tokens) and tokens[idx].lower() == "as":
            idx += 1
        if idx < len(tokens):
            alias = tokens[idx].strip("`\"[]")
            alias_key = normalize_identifier(alias)
            if alias_key and alias_key not in SQL_ALIAS_STOPWORDS and alias not in {",", "(", ")"}:
                aliases[alias_key] = table
    return aliases


def parse_missing_identifier(error: str, kind: str) -> Optional[str]:
    match = re.search(rf"no such {kind}:\s*([A-Za-z_][A-Za-z0-9_.$]*)", error or "", flags=re.IGNORECASE)
    return match.group(1) if match else None


def qualified_column_refs(sql: str) -> List[Tuple[str, str]]:
    return [
        (alias, column)
        for alias, column in re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
            strip_sql_literals(sql),
        )
    ]


def selected_catalog(catalog: Dict[str, List[str]], selected_tables: Iterable[str]) -> Dict[str, List[str]]:
    selected_keys = {normalize_identifier(table) for table in selected_tables}
    if not selected_keys:
        return catalog
    return {
        table: columns
        for table, columns in catalog.items()
        if normalize_identifier(table) in selected_keys
    }


def format_available_columns(catalog: Dict[str, List[str]], max_tables: int = 8, max_columns: int = 18) -> str:
    parts = []
    for table in sorted(catalog)[:max_tables]:
        columns = catalog[table][:max_columns]
        suffix = ", ..." if len(catalog[table]) > max_columns else ""
        parts.append(f"{table}({', '.join(columns)}{suffix})")
    return "; ".join(parts)


def sample_rows(rows: List[Tuple[Any, ...]], max_rows: int = 3) -> List[Tuple[Any, ...]]:
    return canonicalize_rows(rows)[:max_rows]


def build_identifier_diagnostics(
    conn: sqlite3.Connection,
    sql: str,
    error: str,
    selected_tables: Iterable[str],
) -> List[str]:
    full_catalog = get_schema_catalog(conn)
    visible_catalog = selected_catalog(full_catalog, selected_tables)
    visible_table_keys = {normalize_identifier(table): table for table in visible_catalog}
    columns_by_table_key = {
        normalize_identifier(table): {normalize_identifier(column): column for column in columns}
        for table, columns in visible_catalog.items()
    }
    owners = get_column_owner_index(visible_catalog)
    aliases = parse_table_aliases(sql)
    lines: List[str] = []

    missing_table = parse_missing_identifier(error, "table")
    if missing_table:
        table_key = normalize_identifier(missing_table)
        if table_key not in visible_table_keys:
            lines.append(
                f"Invalid table `{missing_table}` is not in the selected schema. "
                f"Use only these tables: {', '.join(sorted(visible_catalog))}."
            )

    missing_column = parse_missing_identifier(error, "column")
    refs = qualified_column_refs(sql)
    if missing_column and "." in missing_column:
        alias, column = missing_column.split(".", 1)
        refs = [(alias, column), *[ref for ref in refs if f"{ref[0]}.{ref[1]}".lower() != missing_column.lower()]]
    elif missing_column:
        refs = [("", missing_column), *refs]

    seen_refs: Set[Tuple[str, str]] = set()
    invalid_lines = []
    for alias, column in refs:
        alias_key = normalize_identifier(alias)
        column_key = normalize_identifier(column)
        ref_key = (alias_key, column_key)
        if ref_key in seen_refs:
            continue
        seen_refs.add(ref_key)

        candidate_owners = owners.get(column_key, [])
        if alias_key:
            alias_table = aliases.get(alias_key)
            alias_table_key = normalize_identifier(alias_table)
            if alias_key not in aliases:
                owner_text = f"; `{column}` exists in: {', '.join(candidate_owners)}" if candidate_owners else ""
                invalid_lines.append(f"`{alias}.{column}` uses undefined alias `{alias}`{owner_text}.")
            elif alias_table_key in columns_by_table_key and column_key not in columns_by_table_key[alias_table_key]:
                if candidate_owners:
                    invalid_lines.append(
                        f"`{alias}.{column}` is invalid: alias `{alias}` maps to table `{alias_table}`, "
                        f"but `{column}` belongs to: {', '.join(candidate_owners)}."
                    )
                else:
                    invalid_lines.append(
                        f"`{alias}.{column}` is invalid: alias `{alias}` maps to table `{alias_table}`, "
                        f"and `{column}` is not in the selected schema."
                    )
        elif missing_column and not candidate_owners:
            invalid_lines.append(f"`{column}` is not in the selected schema.")
        elif missing_column and candidate_owners:
            invalid_lines.append(f"`{column}` exists in: {', '.join(candidate_owners)}.")

    if invalid_lines:
        lines.append("Invalid identifier diagnostics: " + " ".join(invalid_lines[:6]))

    available = format_available_columns(visible_catalog)
    if available:
        lines.append("Available columns in selected tables: " + available + ".")
    return lines


def build_feedback(
    pred: ExecutionResult,
    gold: ExecutionResult,
    feedback_mode: str,
    conn: Optional[sqlite3.Connection] = None,
    previous_sql: str = "",
    selected_tables: Optional[Iterable[str]] = None,
    feedback_detail: str = "basic",
    max_rows: int = 3,
    online_signals: Optional[Iterable[str]] = None,
) -> str:
    include_column_diagnostics = feedback_detail in {"column", "minimal"}
    online_signal_list = list(online_signals or [])
    details: List[str] = []
    if not pred.ok:
        details.append(f"SQLite error: {pred.error}")
        if feedback_mode == "online_visible" and online_signal_list:
            details.append(f"Online-observable signals: {', '.join(online_signal_list)}.")
        if include_column_diagnostics and conn is not None and previous_sql:
            details.extend(build_identifier_diagnostics(conn, previous_sql, pred.error, selected_tables or []))
        return "\n".join(details)

    if feedback_mode == "error_only":
        return "The SQL executed, but it did not pass the hidden result check."

    if feedback_mode == "online_visible":
        if len(pred.rows) == 0:
            details.append(
                "The SQL executed without a runtime error, but it returned an empty result. "
                "Treat this as an online result-sanity warning and revise only if the question implies non-empty output."
            )
        if "cartesian_join_risk" in online_signal_list:
            details.append(
                "The query plan indicates a Cartesian join risk. Check whether a join condition is missing or an unnecessary table should be removed."
            )
        if "explain_plan_error" in online_signal_list:
            details.append("The SQL could not be analyzed by EXPLAIN QUERY PLAN; simplify the query and keep it valid SQLite.")
        guard_failures = [
            signal
            for signal in online_signal_list
            if signal in {"syntax_guard_failed", "schema_guard_failed", "not_executable"}
        ]
        if guard_failures:
            details.append(f"Rule-based SQL guard signals: {', '.join(guard_failures)}.")
        if details:
            if include_column_diagnostics and conn is not None and previous_sql:
                details.extend(build_identifier_diagnostics(conn, previous_sql, pred.error, selected_tables or []))
            return "\n".join(details)
        return "The SQL executed without an executor-visible error; no online repair feedback is available."

    if feedback_mode == "result_status":
        if include_column_diagnostics:
            details.append(
                "The SQL executed, but it returned the wrong result. "
                f"Predicted row count: {len(pred.rows)}; expected row count: {len(gold.rows) if gold.ok else 'unknown'}."
            )
        else:
            details.append(
                "The SQL executed, but it returned the wrong result. "
                f"Predicted row count: {len(pred.rows)}."
            )
        if include_column_diagnostics:
            details.append(f"Predicted sample: {sample_rows(pred.rows, max_rows)}.")
        if include_column_diagnostics and conn is not None and previous_sql:
            details.extend(build_identifier_diagnostics(conn, previous_sql, pred.error, selected_tables or []))
        return "\n".join(details)

    if not gold.ok:
        return "The reference SQL failed in the sandbox; check the database and gold SQL."

    details.append(
        "The SQL executed but returned the wrong result. "
        f"Predicted row count: {len(pred.rows)}; expected row count: {len(gold.rows)}. "
        f"Predicted sample: {sample_rows(pred.rows, max_rows)}; expected sample: {sample_rows(gold.rows, max_rows)}."
    )
    if include_column_diagnostics and conn is not None and previous_sql:
        details.extend(build_identifier_diagnostics(conn, previous_sql, pred.error, selected_tables or []))
    return "\n".join(details)


def load_hf_model(model_path: str, device_map: str = "auto", adapter_path: Optional[str] = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
) -> List[str]:
    import torch

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        model_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        model_input = prompt

    inputs = tokenizer(model_input, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    do_sample = num_return_sequences > 1 or temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    texts = []
    for output in outputs:
        texts.append(tokenizer.decode(output[input_len:], skip_special_tokens=True).strip())
    return texts


def perturb_gold_sql(gold_sql: str) -> str:
    sql = clean_sql(gold_sql)
    if re.search(r"\bFROM\b", sql, flags=re.IGNORECASE):
        return re.sub(r"\bFROM\s+([`\"\[]?[A-Za-z_][A-Za-z0-9_]*[`\"\]]?)", "FROM __missing_table__", sql, count=1, flags=re.IGNORECASE)
    return "SELECT * FROM __missing_table__"


def generate_sql_samples(
    tokenizer,
    model,
    prompt: str,
    gold_sql: str,
    turn: int,
    args: argparse.Namespace,
) -> List[str]:
    if args.generator == "mock":
        if turn == 1:
            return [perturb_gold_sql(gold_sql)]
        return [gold_sql]

    return generate_text(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.num_samples,
    )


def score_candidate(
    conn: sqlite3.Connection,
    candidate_sql: str,
    gold_result: ExecutionResult,
    turn: int,
    max_turns: int,
    timeout_sec: float,
) -> Dict[str, Any]:
    pred_result = execute_sql(conn, candidate_sql, timeout_sec)
    is_correct = execution_match(pred_result, gold_result)
    cost = cost_reward(conn, candidate_sql) if pred_result.ok else CostResult(-0.10, [], {"not_executable": 1})
    base = rule_reward(pred_result, is_correct, turn, max_turns)
    reward_with_cost = round(base + cost.reward, 4)
    flags = validity_flags(candidate_sql, pred_result)
    return {
        "sql": clean_sql(candidate_sql),
        "turn": turn,
        "ok": pred_result.ok,
        "correct": is_correct,
        "syntax_valid": bool(flags["syntax_valid"]),
        "schema_valid": bool(flags["schema_valid"]),
        "error": pred_result.error,
        "row_count": len(pred_result.rows) if pred_result.ok else 0,
        "elapsed_ms": round(pred_result.elapsed_ms, 3),
        "reward": base,
        "rule_reward": base,
        "reward_with_cost": reward_with_cost,
        "cost_reward": cost.reward,
        "cost_flags": {**cost.flags, **flags},
        "explain": cost.explain,
    }


def choose_rejected(candidates: List[Dict[str, Any]], first_turn_only: bool = True) -> Optional[Dict[str, Any]]:
    pool_candidates = [item for item in candidates if item["turn"] == 1] if first_turn_only else candidates
    wrong = [item for item in pool_candidates if not item["correct"]]
    if not wrong:
        return None
    executable_wrong = [item for item in wrong if item["ok"]]
    pool = executable_wrong or wrong
    return sorted(pool, key=lambda item: (item["reward"], -item["turn"]))[0]


def choose_cost_pair(
    candidates: List[Dict[str, Any]],
    min_gap: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    correct_by_sql = {}
    for item in candidates:
        if item["correct"] and item["ok"]:
            correct_by_sql.setdefault(item["sql"], item)

    correct = list(correct_by_sql.values())
    if len(correct) < 2:
        return None, None

    ordered = sorted(correct, key=lambda item: (item["reward"], -item["elapsed_ms"]), reverse=True)
    chosen = ordered[0]
    rejected = ordered[-1]
    if chosen["sql"] == rejected["sql"]:
        return None, None
    if chosen["reward"] - rejected["reward"] < min_gap:
        return None, None
    return chosen, rejected


def make_dpo_pair(
    prompt: str,
    chosen: Dict[str, Any],
    rejected: Dict[str, Any],
    example: Dict[str, Any],
    pair_type: str,
) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "chosen": f"SQL: {chosen['sql']}",
        "rejected": f"SQL: {rejected['sql']}",
        "meta": {
            "pair_type": pair_type,
            "db_id": example.get("db_id") or example.get("database_id"),
            "question": get_field(example, ["question", "utterance", "nl"], required=False),
            "gold_sql": get_field(example, ["query", "sql", "gold_sql"], required=False),
            "chosen_reward": chosen["reward"],
            "rejected_reward": rejected["reward"],
            "chosen_turn": chosen["turn"],
            "rejected_turn": rejected["turn"],
            "chosen_cost_flags": chosen.get("cost_flags", {}),
            "rejected_cost_flags": rejected.get("cost_flags", {}),
        },
    }


def gold_candidate(conn: sqlite3.Connection, gold_sql: str, gold_result: ExecutionResult, timeout_sec: float) -> Dict[str, Any]:
    item = score_candidate(
        conn=conn,
        candidate_sql=gold_sql,
        gold_result=gold_result,
        turn=1,
        max_turns=1,
        timeout_sec=timeout_sec,
    )
    item["source"] = "gold"
    return item


def rollout_example(
    example: Dict[str, Any],
    db_path: str,
    tokenizer,
    model,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    question = get_field(example, ["question", "utterance", "nl"])
    gold_sql = get_field(example, ["query", "sql", "gold_sql"])

    with open_readonly_sqlite(db_path) as conn:
        schema_text, retrieved_tables = build_schema_text(conn, question, args)
        prompt = build_initial_prompt(question, schema_text)
        gold_result = execute_sql(conn, gold_sql, args.timeout_sec)
        if not gold_result.ok:
            trajectory = {
                "question": question,
                "db_path": db_path,
                "db_id": example.get("db_id") or example.get("database_id"),
                "gold_sql": gold_sql,
                "gold_error": gold_result.error,
                "candidates": [],
                "status": "bad_gold",
            }
            return trajectory, [], []

        candidates = []
        current_prompt = prompt
        repaired = None
        repair_policy_pair = None

        for turn in range(1, args.max_turns + 1):
            samples = generate_sql_samples(
                tokenizer=tokenizer,
                model=model,
                prompt=current_prompt,
                gold_sql=gold_sql,
                turn=turn,
                args=args,
            )
            turn_candidates = []
            seen_sql = set()
            for raw_text in samples:
                sql = extract_sql(raw_text)
                if not sql or sql in seen_sql:
                    continue
                seen_sql.add(sql)
                scored = score_candidate(conn, sql, gold_result, turn, args.max_turns, args.timeout_sec)
                scored["raw_text"] = raw_text
                scored["source"] = "model"
                candidates.append(scored)
                turn_candidates.append(scored)

            correct = [item for item in turn_candidates if item["correct"]]
            if correct:
                repaired = sorted(correct, key=lambda item: (-item["reward"], item["elapsed_ms"]))[0]
                if turn > 1:
                    same_prompt_rejected = choose_rejected(turn_candidates, first_turn_only=False)
                    if (
                        same_prompt_rejected
                        and repaired["sql"] != same_prompt_rejected["sql"]
                        and repaired["reward"] > same_prompt_rejected["reward"]
                    ):
                        repair_policy_pair = make_dpo_pair(
                            current_prompt,
                            repaired,
                            same_prompt_rejected,
                            example,
                            "repair_correct_vs_repair_failed",
                        )
                break

            if turn_candidates and turn < args.max_turns:
                best_for_feedback = sorted(turn_candidates, key=lambda item: item["reward"], reverse=True)[0]
                pred_result = execute_sql(conn, best_for_feedback["sql"], args.timeout_sec)
                online_signals = online_guarded_signals(pred_result, best_for_feedback)
                if not should_attempt_repair(pred_result, args.repair_scope, best_for_feedback):
                    break
                feedback = build_feedback(
                    pred_result,
                    gold_result,
                    args.feedback_mode,
                    conn=conn,
                    previous_sql=best_for_feedback["sql"],
                    selected_tables=retrieved_tables,
                    feedback_detail=args.feedback_detail,
                    online_signals=online_signals,
                )
                current_prompt = build_repair_prompt(
                    question=question,
                    schema_text=schema_text,
                    previous_sql=best_for_feedback["sql"],
                    feedback=feedback,
                    turn=turn,
                    feedback_detail=args.feedback_detail,
                )

        status = "correct_first_turn" if repaired and repaired["turn"] == 1 else None
        if repaired and repaired["turn"] > 1:
            status = "repaired"
        if not repaired:
            status = "failed"

        trajectory = {
            "question": question,
            "db_path": db_path,
            "db_id": example.get("db_id") or example.get("database_id"),
            "gold_sql": gold_sql,
            "gold_row_count": len(gold_result.rows),
            "prompt": prompt,
            "schema_mode": args.schema_mode,
            "retrieved_tables": retrieved_tables,
            "status": status,
            "candidates": candidates,
        }

        dpo_pairs_for_example = []

        def add_pair(pair: Optional[Dict[str, Any]]) -> None:
            if not pair:
                return
            key = (pair.get("prompt"), pair.get("chosen"), pair.get("rejected"))
            existing = {
                (item.get("prompt"), item.get("chosen"), item.get("rejected"))
                for item in dpo_pairs_for_example
            }
            if key not in existing:
                dpo_pairs_for_example.append(pair)

        add_pair(repair_policy_pair)

        dpo_pair = None
        rejected = choose_rejected(candidates, first_turn_only=True)
        if rejected:
            if repaired:
                chosen = repaired
                if repaired["turn"] > rejected["turn"]:
                    pair_type = "self_repair_success_vs_failed_attempt"
                else:
                    pair_type = "first_turn_correct_sample_vs_failed_sample"
            elif args.use_gold_when_failed:
                chosen = gold_candidate(conn, gold_sql, gold_result, args.timeout_sec)
                pair_type = "gold_vs_failed_attempt"
            else:
                chosen = None
                pair_type = ""

            if chosen and chosen["sql"] != rejected["sql"] and chosen["reward"] > rejected["reward"]:
                dpo_pair = make_dpo_pair(prompt, chosen, rejected, example, pair_type)
                add_pair(dpo_pair)

        if not dpo_pair:
            chosen_cost, rejected_cost = choose_cost_pair(candidates, args.min_cost_reward_gap)
            if chosen_cost and rejected_cost:
                dpo_pair = make_dpo_pair(
                    prompt,
                    chosen_cost,
                    rejected_cost,
                    example,
                    "correct_low_cost_vs_correct_high_cost",
                )
                add_pair(dpo_pair)

        grpo_record = {
            "prompt": prompt,
            "db_id": example.get("db_id") or example.get("database_id"),
            "question": question,
            "gold_sql": gold_sql,
            "completions": [
                {
                    "response": f"SQL: {item['sql']}",
                    "reward": item["reward"],
                    "meta": {
                        "turn": item["turn"],
                        "correct": item["correct"],
                        "ok": item["ok"],
                        "error": item["error"],
                        "rule_reward": item["rule_reward"],
                        "cost_reward": item["cost_reward"],
                        "cost_flags": item["cost_flags"],
                    },
                }
                for item in candidates
            ],
        }

        return trajectory, [grpo_record], dpo_pairs_for_example


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def summarize(trajectories: List[Dict[str, Any]], dpo_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    status = Counter(item["status"] for item in trajectories)
    pair_types = Counter(pair["meta"]["pair_type"] for pair in dpo_pairs)
    cost_flags = Counter()
    rewards = []
    first_turn = []
    all_candidates = []
    for traj in trajectories:
        for item in traj.get("candidates", []):
            all_candidates.append(item)
            if item["turn"] == 1:
                first_turn.append(item)
            rewards.append(item["reward"])
            cost_flags.update(item.get("cost_flags", {}))

    executable = [item for item in all_candidates if item.get("ok")]
    syntax_valid = [item for item in all_candidates if item.get("syntax_valid")]
    schema_valid = [item for item in all_candidates if item.get("schema_valid")]
    repaired = status.get("repaired", 0)
    failed_after_first = status.get("failed", 0) + repaired
    evaluable_total = sum(
        count
        for name, count in status.items()
        if name not in {"bad_gold", "skipped"}
    )

    schema_valid_rate = round(len(schema_valid) / len(all_candidates), 4) if all_candidates else 0.0

    return {
        "total": len(trajectories),
        "status": dict(status),
        "dpo_pairs": len(dpo_pairs),
        "pair_types": dict(pair_types),
        "first_turn_example_accuracy": round(
            status.get("correct_first_turn", 0) / evaluable_total, 4
        ) if evaluable_total else 0.0,
        "final_example_execution_accuracy": round(
            (status.get("correct_first_turn", 0) + repaired) / evaluable_total, 4
        ) if evaluable_total else 0.0,
        "first_turn_execution_accuracy": round(
            sum(1 for item in first_turn if item.get("correct")) / len(first_turn), 4
        ) if first_turn else 0.0,
        "candidate_execution_accuracy": round(
            sum(1 for item in all_candidates if item.get("correct")) / len(all_candidates), 4
        ) if all_candidates else 0.0,
        "syntax_error_rate": round(
            1.0 - (len(syntax_valid) / len(all_candidates))
        , 4) if all_candidates else 0.0,
        "schema_valid_rate": schema_valid_rate,
        "schema_alignment_rate": schema_valid_rate,
        "executable_rate": round(len(executable) / len(all_candidates), 4) if all_candidates else 0.0,
        "repair_success_rate": round(repaired / failed_after_first, 4) if failed_after_first else 0.0,
        "avg_candidate_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "cost_flags": dict(cost_flags),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Text-to-SQL self-correction trajectories with lightweight cost-aware rewards."
    )
    parser.add_argument("--dataset_path", required=True, help="JSON/JSONL dataset path. Supports Spider-style fields.")
    parser.add_argument("--db_root", default=None, help="Root dir for Spider-style database/{db_id}/{db_id}.sqlite files.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--model_path", default=None, help="Local or HF model path for SQL generation.")
    parser.add_argument("--adapter_path", default=None, help="Optional PEFT/LoRA adapter path for evaluation.")
    parser.add_argument("--generator", choices=["hf", "mock"], default="hf")
    parser.add_argument("--schema_mode", choices=["full", "retrieved", "bm25", "bm25_fk"], default="retrieved")
    parser.add_argument("--top_k_tables", type=int, default=6)
    parser.add_argument("--fk_hops", type=int, default=1, help="Foreign-key expansion hops for --schema_mode bm25_fk.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N examples before applying --limit.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--timeout_sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--feedback_mode",
        choices=["error_only", "result_status", "oracle_rows", "online_visible"],
        default="result_status",
        help=(
            "Repair feedback detail. online_visible only exposes executor-visible errors and empty-result "
            "sanity warnings; oracle_rows uses gold result samples and is strongest but less realistic."
        ),
    )
    parser.add_argument(
        "--repair_scope",
        choices=["verified", "online", "online_guarded"],
        default="verified",
        help=(
            "verified repairs every verifier-caught failure; online repairs only executor/runtime failures; "
            "online_guarded also allows empty-result sanity warnings and conservative rule-guard signals."
        ),
    )
    parser.add_argument(
        "--feedback_detail",
        choices=["basic", "column", "minimal"],
        default="basic",
        help=(
            "basic keeps legacy feedback; column adds invalid identifier diagnostics and selected-table columns; "
            "minimal also asks repair to regenerate a minimal SQL and remove unnecessary joins."
        ),
    )
    parser.add_argument(
        "--min_cost_reward_gap",
        type=float,
        default=0.05,
        help="Minimum reward gap for cost-only DPO pairs among correct SQL candidates.",
    )
    parser.add_argument(
        "--use_gold_when_failed",
        action="store_true",
        help="If no self-repair succeeds, use gold SQL as chosen and a failed attempt as rejected.",
    )
    parser.add_argument("--progress_every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    examples = read_json_or_jsonl(args.dataset_path)
    if args.offset < 0:
        raise ValueError("--offset must be non-negative.")
    if args.offset:
        examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]

    tokenizer = model = None
    if args.generator == "hf":
        if not args.model_path:
            raise ValueError("--model_path is required when --generator hf")
        print(f"Loading model: {args.model_path}")
        tokenizer, model = load_hf_model(args.model_path, args.device_map, args.adapter_path)
    else:
        print("Using mock generator: turn 1 emits an invalid SQL, repair turns emit gold SQL.")

    trajectories = []
    grpo_records = []
    dpo_pairs = []
    skipped = Counter()

    print(
        f"Building trajectories: {len(examples)} examples | max_turns={args.max_turns} | "
        f"num_samples={args.num_samples}"
    )

    for idx, example in enumerate(examples, start=1):
        try:
            db_path = resolve_db_path(example, args.db_root)
            trajectory, grpo, example_dpo_pairs = rollout_example(example, db_path, tokenizer, model, args)
            trajectories.append(trajectory)
            grpo_records.extend(grpo)
            dpo_pairs.extend(example_dpo_pairs)
        except Exception as exc:
            skipped[type(exc).__name__] += 1
            trajectories.append(
                {
                    "status": "skipped",
                    "error": str(exc),
                    "question": example.get("question"),
                    "db_id": example.get("db_id") or example.get("database_id"),
                }
            )

        if idx % args.progress_every == 0 or idx == len(examples):
            summary = summarize(trajectories, dpo_pairs)
            print(
                f"Processed {idx}/{len(examples)} | "
                f"first-turn={summary['status'].get('correct_first_turn', 0)} | "
                f"repaired={summary['status'].get('repaired', 0)} | "
                f"failed={summary['status'].get('failed', 0)} | "
                f"dpo_pairs={summary['dpo_pairs']}"
            )

    summary = summarize(trajectories, dpo_pairs)
    summary["skipped_errors"] = dict(skipped)
    summary["args"] = vars(args)

    trajectories_path = os.path.join(args.output_dir, "trajectories.jsonl")
    grpo_path = os.path.join(args.output_dir, "grpo_rollouts.jsonl")
    dpo_path = os.path.join(args.output_dir, "dpo_pairs.json")
    summary_path = os.path.join(args.output_dir, "summary.json")

    write_jsonl(trajectories_path, trajectories)
    write_jsonl(grpo_path, grpo_records)
    write_json(dpo_path, dpo_pairs)
    write_json(summary_path, summary)

    print("\nDone")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Trajectories: {trajectories_path}")
    print(f"GRPO rollouts: {grpo_path}")
    print(f"DPO pairs: {dpo_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
