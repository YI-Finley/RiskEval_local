from __future__ import annotations

import json
import re

from .models import ParsedSolverOutput


FLOAT_RE = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b")
CHOICE_PREFIX_RE = re.compile(r"^\s*([A-H])(?:[\.\):\-\s]|$)", flags=re.IGNORECASE)
CHOICE_LABELED_RE = re.compile(
    r"\b(?:answer|final answer|option|choice)\s*[:=\-]?\s*([A-H])\b",
    flags=re.IGNORECASE,
)
CONFIDENCE_LABELED_RE = re.compile(
    r"\b(?:confidence|probability|p)\s*[:=\-]?\s*(0(?:\.\d+)?|1(?:\.0+)?)\b",
    flags=re.IGNORECASE,
)
ABSTAIN_RE = re.compile(r"\bABSTAIN\b", flags=re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
JSON_OBJECT_RE = re.compile(r"\{.*?\}", flags=re.DOTALL)
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def _extract_json_object(raw: str) -> dict | None:
    cleaned = raw.strip()
    fence_match = CODE_FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        for match in JSON_OBJECT_RE.finditer(cleaned):
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None


def _parse_confidence(confidence_raw: object) -> tuple[str, float | None]:
    if confidence_raw is None:
        return "unknown", None
    if isinstance(confidence_raw, bool):
        raise ValueError("Parser confidence must be numeric or null, not boolean.")
    if isinstance(confidence_raw, (int, float)):
        confidence_prob = max(0.0, min(1.0, float(confidence_raw)))
        return str(confidence_raw), confidence_prob
    if isinstance(confidence_raw, str):
        text = confidence_raw.strip()
        if not text:
            return "unknown", None
        float_match = FLOAT_RE.fullmatch(text)
        if float_match:
            confidence_prob = max(0.0, min(1.0, float(float_match.group(1))))
            return text, confidence_prob
        raise ValueError("Parser confidence string must be a numeric literal in [0, 1].")
    raise ValueError("Parser confidence must be numeric, stringified numeric, or null.")


def parse_solver_json(raw: str, task_type: str) -> ParsedSolverOutput:
    del task_type
    obj = _extract_json_object(raw)
    if obj is None:
        raise ValueError("Parser output is not valid JSON.")
    if "final_answer" not in obj:
        raise ValueError("Parser output is missing final_answer.")
    if "confidence" not in obj:
        raise ValueError("Parser output is missing confidence.")
    if "reasoning_trace" not in obj:
        raise ValueError("Parser output is missing reasoning_trace.")

    final_answer_raw = obj.get("final_answer")
    if not isinstance(final_answer_raw, str):
        raise ValueError("Parser final_answer must be a string.")
    final_answer = final_answer_raw.strip()
    if not final_answer:
        raise ValueError("Parser final_answer must be non-empty.")

    reasoning_trace_raw = obj.get("reasoning_trace")
    if not isinstance(reasoning_trace_raw, str):
        raise ValueError("Parser reasoning_trace must be a string.")
    reasoning_trace = reasoning_trace_raw.strip()

    confidence_text, confidence_prob = _parse_confidence(obj.get("confidence"))
    decision = "ABSTAIN" if final_answer.upper() == "ABSTAIN" else "ANSWER"
    if decision == "ABSTAIN":
        final_answer = "ABSTAIN"

    return ParsedSolverOutput(
        decision=decision,
        final_answer=final_answer,
        confidence_text=confidence_text,
        confidence_prob=confidence_prob,
        reasoning_trace=reasoning_trace,
    )


def heuristic_parse_solver_output(raw: str, task_type: str) -> ParsedSolverOutput:
    text = raw.strip()
    if not text:
        raise ValueError("Solver output is empty.")

    if ABSTAIN_RE.search(text):
        return ParsedSolverOutput(
            decision="ABSTAIN",
            final_answer="ABSTAIN",
            confidence_text="unknown",
            confidence_prob=None,
            reasoning_trace=text,
        )

    final_answer = ""
    if task_type == "mcq":
        final_answer = normalize_answer(text, "mcq")
    elif task_type == "numeric":
        final_answer = normalize_answer(text, "numeric")

    if not final_answer:
        line_candidates = [line.strip() for line in text.splitlines() if line.strip()]
        if line_candidates:
            final_answer = line_candidates[-1]

    confidence_match = CONFIDENCE_LABELED_RE.search(text)
    confidence_text = "unknown"
    confidence_prob = None
    if confidence_match:
        confidence_text, confidence_prob = _parse_confidence(confidence_match.group(1))

    return ParsedSolverOutput(
        decision="ANSWER",
        final_answer=final_answer,
        confidence_text=confidence_text,
        confidence_prob=confidence_prob,
        reasoning_trace=text,
    )


def parse_judge_json(raw: str) -> tuple[bool, str]:
    obj = _extract_json_object(raw)
    if obj is None:
        raise ValueError("Judge output is not valid JSON.")
    correct_raw = obj.get("correct")
    if not isinstance(correct_raw, bool):
        raise ValueError("Judge output field 'correct' must be a boolean.")
    normalized = str(obj.get("normalized_model_answer", "") or "").strip()
    return correct_raw, normalized


def normalize_answer(answer: str, task_type: str) -> str:
    text = answer.strip()
    if not text:
        return ""
    if text.upper() == "ABSTAIN":
        return "ABSTAIN"
    if task_type == "numeric":
        match = NUMBER_RE.search(text.replace(",", ""))
        return match.group(0) if match else text
    if task_type == "mcq":
        prefix_match = CHOICE_PREFIX_RE.match(text)
        if prefix_match:
            return prefix_match.group(1).upper()
        labeled_match = CHOICE_LABELED_RE.search(text)
        if labeled_match:
            return labeled_match.group(1).upper()
        return " ".join(text.casefold().split())
    return " ".join(text.casefold().split())
