from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import QAExample


def _infer_task_type(obj: dict) -> str:
    raw = str(obj.get("task_type", "") or "").strip().lower()
    if raw:
        return raw
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        return "mcq"
    return "open"


def _normalize_gold(obj: dict) -> tuple[str | None, bool]:
    if "answer" not in obj or obj["answer"] is None:
        return None, False
    answer = str(obj["answer"]).strip()
    if not answer:
        return None, False
    return answer, True


def load_jsonl(path: Path) -> list[QAExample]:
    rows: list[QAExample] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            answer, has_gold = _normalize_gold(obj)
            image = obj.get("image")
            image_value = str(image).strip() if image is not None and str(image).strip() else None
            modality = str(obj.get("modality", "") or "").strip().lower()
            if not modality:
                modality = "multimodal" if image_value else "text"
            rows.append(
                QAExample(
                    qid=str(obj["id"]),
                    task_type=_infer_task_type(obj),
                    question=str(obj["question"]),
                    choices=[str(c) for c in obj.get("choices", [])],
                    answer=answer,
                    has_gold=has_gold,
                    modality=modality,
                    image=image_value,
                )
            )
    return rows


def load_jsonl_dicts(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reset_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
