from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.ipc as ipc


HF_CACHE = Path("/Users/wuyuchen/.cache/huggingface/datasets")
DATA_DIR = Path("/Users/wuyuchen/Desktop/RiskEval/data")


def _read_arrow_rows(path: Path) -> list[dict]:
    with ipc.open_stream(str(path)) as reader:
        table = reader.read_all()
    return [
        {name: table[name][i].as_py() for name in table.column_names}
        for i in range(table.num_rows)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


MCQ_CHOICE_RE = re.compile(r"^\s*([A-H])[\.\)]\s*(.+?)\s*$")
LOWER_CHOICE_RE = re.compile(r"^\s*([a-h])[\.\)]\s*(.+?)\s*$")


def _extract_embedded_choices(question: str) -> tuple[str, list[str]]:
    lines = [line.rstrip() for line in question.splitlines()]
    choices: list[str] = []
    question_lines: list[str] = []
    seen_choices = False

    for line in lines:
        if line.strip().casefold() == "answer choices:":
            seen_choices = True
            continue
        upper_match = MCQ_CHOICE_RE.match(line)
        if upper_match:
            seen_choices = True
            choices.append(f"{upper_match.group(1)}. {upper_match.group(2).strip()}")
            continue
        if seen_choices and not line.strip():
            continue
        question_lines.append(line)

    cleaned_question = "\n".join(question_lines).strip()
    return cleaned_question, choices


def _extract_gpqa_choices(question: str) -> tuple[str, list[str]]:
    lines = [line.rstrip() for line in question.splitlines()]
    stem_lines: list[str] = []
    lower_choices: dict[str, str] = {}
    upper_map: list[tuple[str, str]] = []

    for line in lines:
        lower_match = LOWER_CHOICE_RE.match(line)
        if lower_match:
            lower_choices[lower_match.group(1).lower()] = lower_match.group(2).strip()
            continue
        upper_match = MCQ_CHOICE_RE.match(line)
        if upper_match and upper_match.group(2).strip().lower() in lower_choices:
            upper_map.append((upper_match.group(1), upper_match.group(2).strip().lower()))
            continue
        stem_lines.append(line)

    if lower_choices and upper_map:
        choices = [
            f"{letter}. {lower_choices[key]}"
            for letter, key in upper_map
            if key in lower_choices
        ]
        return "\n".join(stem_lines).strip(), choices

    return _extract_embedded_choices(question)


def build_gpqa() -> Path:
    arrow = HF_CACHE / "fingertap___gpqa-diamond/default/0.0.0/68be7564497676e07a77a042fdb587deb88c51c3/gpqa-diamond-test.arrow"
    rows = _read_arrow_rows(arrow)
    out = []
    for idx, row in enumerate(rows):
        cleaned_question, choices = _extract_gpqa_choices(row["question"])
        out.append(
            {
                "id": f"gpqa_test_{idx}",
                "task_type": "mcq",
                "question": cleaned_question,
                "choices": choices,
                "answer": str(row["answer"]).strip().upper(),
                "source_dataset": "fingertap/GPQA-Diamond",
                "split": "test",
            }
        )
    path = DATA_DIR / "gpqa_diamond_test.jsonl"
    _write_jsonl(path, out)
    return path


def _extract_gsm8k_final(answer: str) -> str:
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer)
    return nums[-1].replace(",", "") if nums else answer.strip()


def build_gsm8k(config_name: str) -> Path:
    arrow = HF_CACHE / f"openai___gsm8k/{config_name}/0.0.0/cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k-test.arrow"
    rows = _read_arrow_rows(arrow)
    out = []
    for idx, row in enumerate(rows):
        out.append(
            {
                "id": f"gsm8k_{config_name}_test_{idx}",
                "task_type": "numeric",
                "question": row["question"],
                "choices": [],
                "answer": _extract_gsm8k_final(str(row["answer"])),
                "source_dataset": "openai/gsm8k",
                "subset": config_name,
                "split": "test",
            }
        )
    path = DATA_DIR / f"gsm8k_{config_name}_test.jsonl"
    _write_jsonl(path, out)
    return path


def build_hle() -> Path:
    arrow = HF_CACHE / "cais___hle/default/0.0.0/5a81a4c7271a2a2a312b9a690f0c2fde837e4c29/hle-test.arrow"
    rows = _read_arrow_rows(arrow)
    out = []
    for row in rows:
        question = str(row["question"])
        cleaned_question, choices = _extract_embedded_choices(question)
        image = str(row.get("image", "") or "").strip() or None
        task_type = "mcq" if row.get("answer_type") == "multipleChoice" else "open"
        out.append(
            {
                "id": str(row["id"]),
                "task_type": task_type,
                "question": cleaned_question,
                "choices": choices if task_type == "mcq" else [],
                "answer": str(row["answer"]).strip() if row.get("answer") is not None else None,
                "modality": "multimodal" if image else "text",
                "image": image,
                "source_dataset": "cais/hle",
                "split": "test",
                "category": row.get("category", ""),
                "answer_type": row.get("answer_type", ""),
            }
        )
    path = DATA_DIR / "hle_test.jsonl"
    _write_jsonl(path, out)
    return path


def build_hle_mcq_text_only() -> Path:
    arrow = HF_CACHE / "cais___hle/default/0.0.0/5a81a4c7271a2a2a312b9a690f0c2fde837e4c29/hle-test.arrow"
    rows = _read_arrow_rows(arrow)
    out = []
    for row in rows:
        if row.get("answer_type") != "multipleChoice":
            continue
        if str(row.get("image", "")).strip():
            continue
        cleaned_question, choices = _extract_embedded_choices(row["question"])
        out.append(
            {
                "id": str(row["id"]),
                "task_type": "mcq",
                "question": cleaned_question,
                "choices": choices,
                "answer": str(row["answer"]).strip().upper(),
                "source_dataset": "cais/hle",
                "split": "test",
                "category": row.get("category", ""),
                "answer_type": row.get("answer_type", ""),
            }
        )
    path = DATA_DIR / "hle_mcq_text_only_test.jsonl"
    _write_jsonl(path, out)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cached HF Arrow datasets into RiskEval JSONL format")
    parser.add_argument(
        "--dataset",
        choices=["gpqa", "gsm8k-main", "gsm8k-socratic", "hle", "hle-mcq-text", "all"],
        default="all",
    )
    args = parser.parse_args()

    outputs = []
    if args.dataset in {"gpqa", "all"}:
        outputs.append(build_gpqa())
    if args.dataset in {"gsm8k-main", "all"}:
        outputs.append(build_gsm8k("main"))
    if args.dataset in {"gsm8k-socratic", "all"}:
        outputs.append(build_gsm8k("socratic"))
    if args.dataset in {"hle", "all"}:
        outputs.append(build_hle())
    if args.dataset in {"hle-mcq-text", "all"}:
        outputs.append(build_hle_mcq_text_only())

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
