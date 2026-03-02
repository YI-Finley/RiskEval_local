from __future__ import annotations

import argparse
import os

from datasets import DatasetDict, IterableDatasetDict, load_dataset
from huggingface_hub import login


TARGETS = [
    ("fingertap/GPQA-Diamond", None),
    ("cais/hle", None),
    ("openai/gsm8k", "main"),
    ("openai/gsm8k", "socratic"),
]


def _summary(obj: DatasetDict | IterableDatasetDict) -> str:
    parts = []
    for split_name, split in obj.items():
        try:
            size = len(split)
            parts.append(f"{split_name}: {size}")
        except TypeError:
            parts.append(f"{split_name}: iterable")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download requested Hugging Face datasets")
    parser.add_argument("--token", default=None, help="Hugging Face token, optional")
    parser.add_argument("--cache-dir", default=None, help="Optional cache directory")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("Logged in to Hugging Face Hub")
    else:
        print("No token provided, trying anonymous access")

    for dataset_name, config_name in TARGETS:
        label = dataset_name if config_name is None else f"{dataset_name} [{config_name}]"
        print(f"\\nLoading: {label}")
        ds = load_dataset(dataset_name, config_name, cache_dir=args.cache_dir)
        print(_summary(ds))


if __name__ == "__main__":
    main()
