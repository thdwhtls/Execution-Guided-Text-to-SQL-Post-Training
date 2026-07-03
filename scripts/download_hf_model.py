import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot to a local directory.")
    parser.add_argument("--repo_id", required=True, help="Example: Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--local_dir", required=True, help="Local target directory, for example models/qwen2.5-coder-3b")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional mirror endpoint. Example: https://hf-mirror.com. Also supports HF_ENDPOINT env var.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for gated/private repos. Public Qwen2.5 models do not need this.",
    )
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=args.token,
        resume_download=True,
        ignore_patterns=[
            "*.msgpack",
            "*.h5",
            "flax_model*",
            "tf_model*",
            "onnx/*",
        ],
    )
    print(path)


if __name__ == "__main__":
    main()
