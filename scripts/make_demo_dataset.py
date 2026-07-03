import argparse
import json
import sqlite3
from pathlib import Path


EXAMPLES = [
    {
        "db_id": "company",
        "question": "List the names of employees in the Engineering department.",
        "query": (
            "SELECT e.name FROM employees AS e "
            "JOIN departments AS d ON e.department_id = d.id "
            "WHERE d.name = 'Engineering'"
        ),
    },
    {
        "db_id": "company",
        "question": "How many employees work in Sales?",
        "query": (
            "SELECT COUNT(*) FROM employees AS e "
            "JOIN departments AS d ON e.department_id = d.id "
            "WHERE d.name = 'Sales'"
        ),
    },
    {
        "db_id": "company",
        "question": "What is the average salary for employees in each department?",
        "query": (
            "SELECT d.name, AVG(e.salary) FROM departments AS d "
            "JOIN employees AS e ON d.id = e.department_id "
            "GROUP BY d.name"
        ),
    },
    {
        "db_id": "company",
        "question": "Which projects are owned by employees hired after 2021-01-01?",
        "query": (
            "SELECT p.name FROM projects AS p "
            "JOIN employees AS e ON p.owner_id = e.id "
            "WHERE e.hire_date > '2021-01-01'"
        ),
    },
]


def build_company_db(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE departments (
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE employees (
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              department_id INTEGER NOT NULL,
              salary INTEGER NOT NULL,
              hire_date TEXT NOT NULL,
              FOREIGN KEY (department_id) REFERENCES departments(id)
            );

            CREATE TABLE projects (
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              owner_id INTEGER NOT NULL,
              budget INTEGER NOT NULL,
              FOREIGN KEY (owner_id) REFERENCES employees(id)
            );

            CREATE INDEX idx_employees_department ON employees(department_id);
            CREATE INDEX idx_projects_owner ON projects(owner_id);
            """
        )
        conn.executemany(
            "INSERT INTO departments(id, name) VALUES (?, ?)",
            [(1, "Engineering"), (2, "Sales"), (3, "Finance")],
        )
        conn.executemany(
            """
            INSERT INTO employees(id, name, department_id, salary, hire_date)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (1, "Ada", 1, 145000, "2020-03-15"),
                (2, "Grace", 1, 138000, "2022-07-01"),
                (3, "Lin", 2, 99000, "2021-11-20"),
                (4, "Maya", 2, 105000, "2019-05-08"),
                (5, "Noor", 3, 112000, "2023-02-12"),
            ],
        )
        conn.executemany(
            "INSERT INTO projects(id, name, owner_id, budget) VALUES (?, ?, ?, ?)",
            [
                (1, "Forecasting", 2, 500000),
                (2, "Pipeline", 1, 700000),
                (3, "Expansion", 3, 250000),
                (4, "Audit", 5, 120000),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny Spider-style Text-to-SQL demo dataset.")
    parser.add_argument("--out_dir", default="data/demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    db_dir = out_dir / "database" / "company"
    db_dir.mkdir(parents=True, exist_ok=True)
    build_company_db(db_dir / "company.sqlite")

    dataset_path = out_dir / "dev.jsonl"
    with dataset_path.open("w", encoding="utf-8") as f:
        for example in EXAMPLES:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {dataset_path}")
    print(f"Wrote {db_dir / 'company.sqlite'}")


if __name__ == "__main__":
    main()
