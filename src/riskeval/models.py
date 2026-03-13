from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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



def load_local_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return model, tokenizer


def run_local_inference(model, tokenizer, prompt: str, max_tokens: int = 1000, temperature: float = 0.0):
	inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
	outputs = model.generate(
    	**inputs,
    	max_new_tokens=max_tokens,
    	temperature=temperature,
    	do_sample=True if temperature > 0 else False
	)
	return tokenizer.decode(outputs[0], skip_special_tokens=True)
