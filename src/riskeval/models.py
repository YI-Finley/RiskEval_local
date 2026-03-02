from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class QAExample:
    qid: str
    task_type: str
    question: str
    choices: list[str]
    answer: Optional[str]
    has_gold: bool
    modality: str = "text"
    image: Optional[str] = None


@dataclass
class ParsedSolverOutput:
    decision: str
    final_answer: str
    confidence_text: str
    confidence_prob: float | None
    reasoning_trace: str


@dataclass
class ExampleRun:
    qid: str
    task_type: str
    penalty: float
    modality: str
    has_gold: bool
    gold: Optional[str]
    solver_answer: str
    solver_correct: Optional[bool]
    confidence_text: str
    confidence_prob: float | None
    model_decision: str
    judge_decision: str
    judge_applicable: bool
    used_judge: bool
    utility: float | None
    expected_utility_if_answer: float | None
    oracle_utility: float | None
    policy_consistent: bool | None
    regret: float | None
    normalized_regret: float | None
    normalized_utility: float | None

    def to_dict(self) -> dict:
        return asdict(self)
