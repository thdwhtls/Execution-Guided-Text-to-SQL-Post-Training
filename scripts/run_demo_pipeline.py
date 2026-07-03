import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "demo_pipeline"


def run(cmd: Iterable[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=ROOT, check=True)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    run([sys.executable, "scripts/make_demo_dataset.py"])
    run(
        [
            sys.executable,
            "text2sql_trajectory_builder.py",
            "--dataset_path",
            "data/demo/dev.jsonl",
            "--db_root",
            "data/demo/database",
            "--output_dir",
            str(OUTPUT_DIR.relative_to(ROOT)),
            "--generator",
            "mock",
            "--max_turns",
            "3",
            "--feedback_detail",
            "minimal",
            "--use_gold_when_failed",
        ]
    )

    summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    print("\nDemo summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    first = next(read_jsonl(OUTPUT_DIR / "trajectories.jsonl"))
    print("\nExample repair trace")
    print(f"Question: {first.get('question')}")
    for item in first.get("candidates", []):
        status = "correct" if item.get("correct") else "wrong"
        error = item.get("error") or ""
        print(f"- turn {item.get('turn')} [{status}]: {item.get('sql')}")
        if error:
            print(f"  error: {error}")

    print(f"\nArtifacts written to: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
