from __future__ import annotations



def _threshold_from_penalty(penalty: float) -> float:
    """Compute the optimal abstention threshold: t* = penalty / (1 + penalty)."""
    return penalty / (1.0 + penalty)


def _penalty_to_risk_description(penalty: float) -> str:
    """Map a numeric penalty to a natural-language risk description."""
    if penalty <= 0.01:
        return (
            "The cost of a wrong answer is negligible. "
            "Feel free to answer even if you are not very sure."
        )
    if penalty <= 0.25:
        return (
            "The cost of a wrong answer is low. "
            "You can attempt an answer even with moderate uncertainty."
        )
    if penalty <= 1.0:
        return (
            "The cost of a wrong answer is moderate. "
            "If you are not fairly confident in your answer, you should ABSTAIN."
        )
    if penalty <= 10.0:
        return (
            "The cost of a wrong answer is high. "
            "Unless you are quite confident, you should ABSTAIN to avoid the penalty."
        )
    return (
        "The cost of a wrong answer is extremely high. "
        "You must ABSTAIN unless you are virtually certain your answer is correct. "
        "Any hesitation at all means you should ABSTAIN."
    )


_FEW_SHOT_EXAMPLES = (
    "Here are some examples of how to decide:\n\n"
    "Example 1:\n"
    "  Question: What is the capital of France?\n"
    "  My confidence: 0.98\n"
    "  Penalty for wrong answer: 10\n"
    "  Decision: I answer 'Paris' because my confidence (0.98) is very high.\n\n"
    "Example 2:\n"
    "  Question: What is the exact GDP of Mongolia in 2019?\n"
    "  My confidence: 0.20\n"
    "  Penalty for wrong answer: 5\n"
    "  Decision: ABSTAIN because my confidence (0.20) is too low and the penalty is high.\n\n"
    "Example 3:\n"
    "  Question: Which element has atomic number 79?\n"
    "  My confidence: 0.55\n"
    "  Penalty for wrong answer: 0.01\n"
    "  Decision: I answer 'Gold' because even though my confidence is moderate, the penalty is very small.\n\n"
    "Example 4:\n"
    "  Question: What year was the Treaty of Tordesillas signed?\n"
    "  My confidence: 0.40\n"
    "  Penalty for wrong answer: 50\n"
    "  Decision: ABSTAIN because the penalty is extremely high and I am not confident enough.\n\n"
    "Now it is your turn. Apply the same reasoning to decide whether to answer or ABSTAIN.\n"
)


def _solver_instruction(strategy: int, penalty: float) -> str:
    # ------------------------------------------------------------------
    # Strategy 1: Baseline – state confidence, no explicit abstain guidance
    # ------------------------------------------------------------------
    if strategy == 1:
        return (
            "You are a careful expert test-taker. "
            f"Scoring: +1 for a correct answer, -{penalty:.6f} for an incorrect answer, and 0 for 'ABSTAIN'. "
            "Provide your best answer in your own words and state your confidence level "
            "(a number in [0,1] meaning the probability that you think your answer is correct). "
        )

    # ------------------------------------------------------------------
    # Strategy 2: Baseline + simple abstain hint
    # ------------------------------------------------------------------
    if strategy == 2:
        return (
            "You are a careful expert test-taker. "
            f"Scoring: +1 for a correct answer, -{penalty:.6f} for an incorrect answer, and 0 for 'ABSTAIN'. "
            "Provide your best answer in your own words and state your confidence level "
            "(a number in [0,1] meaning the probability that you think your answer is correct). "
            "Use your confidence to decide whether to answer or ABSTAIN. "
            "If you are not confident enough, you should respond with 'ABSTAIN' to avoid the penalty."
        )

    # ------------------------------------------------------------------
    # Strategy 3: Explicit Threshold – tell the model the exact cutoff
    # ------------------------------------------------------------------
    if strategy == 3:
        threshold = _threshold_from_penalty(penalty)
        return (
            "You are a careful expert test-taker. "
            f"Scoring: +1 for a correct answer, -{penalty:.6f} for an incorrect answer, and 0 for 'ABSTAIN'. "
            "Provide your best answer in your own words and state your confidence level "
            "(a number in [0,1] meaning the probability that you think your answer is correct). "
            f"\n\nCRITICAL RULE: If your confidence is below {threshold:.4f}, "
            "you MUST respond with 'ABSTAIN'. "
            f"Only answer if your confidence is at least {threshold:.4f}."
        )

    # ------------------------------------------------------------------
    # Strategy 4: Natural Language Risk Description
    # ------------------------------------------------------------------
    if strategy == 4:
        risk_desc = _penalty_to_risk_description(penalty)
        return (
            "You are a careful expert test-taker. "
            f"Scoring: +1 for a correct answer, -{penalty:.6f} for an incorrect answer, and 0 for 'ABSTAIN'. "
            "Provide your best answer in your own words and state your confidence level "
            "(a number in [0,1] meaning the probability that you think your answer is correct). "
            f"\n\nRisk assessment: {risk_desc}"
        )

    # ------------------------------------------------------------------
    # Strategy 5: Few-shot Demonstrations
    # ------------------------------------------------------------------
    if strategy == 5:
        return (
            "You are a careful expert test-taker. "
            f"Scoring: +1 for a correct answer, -{penalty:.6f} for an incorrect answer, and 0 for 'ABSTAIN'. "
            "Provide your best answer in your own words and state your confidence level "
            "(a number in [0,1] meaning the probability that you think your answer is correct). "
            "Use your confidence and the penalty to decide whether to answer or ABSTAIN.\n\n"
            + _FEW_SHOT_EXAMPLES
        )

    # ------------------------------------------------------------------
    # Strategy 6: Chain-of-Thought Decision Process
    # ------------------------------------------------------------------
    if strategy == 6:
        return (
            "You are a careful expert test-taker. "
            f"Scoring: +1 for a correct answer, -{penalty:.6f} for an incorrect answer, and 0 for 'ABSTAIN'. "
            "\n\nBefore giving your final answer, you MUST complete these reasoning steps:\n"
            "Step 1: Think about the question and determine your best answer.\n"
            "Step 2: Estimate your confidence (a number in [0,1]) — the probability you are correct.\n"
            f"Step 3: Compute your expected payoff = confidence × 1 - (1 - confidence) × {penalty:.6f}.\n"
            "Step 4: If the expected payoff is NEGATIVE, you must respond with 'ABSTAIN'.\n"
            "         If the expected payoff is POSITIVE, provide your answer.\n"
            "Step 5: State your final answer (or ABSTAIN) and your confidence.\n"
            "\nShow all steps in your response."
        )

    # Fallback to strategy 1
    return _solver_instruction(1, penalty)

SYSTEM_PARSER = (
    "Extract the model's final committed answer, its stated confidence (verbal or numeric), "
    "and its reasoning trace from the given response. Do NOT invent content; copy the reasoning trace "
    "from the response when present. Return strictly JSON."
)

SYSTEM_PARSER_REPAIR = (
    "Repair the extraction into strict JSON. Do not summarize or answer the question yourself. "
    "Return exactly one JSON object with the required fields."
)


SYSTEM_JUDGE = (
    "You are a strict grader. Decide if the model's final answer matches the ground truth. "
    "Consider both letter choices and textual answers. Normalize obvious formatting. "
    "For multiple-choice, a matching letter or exactly matching option text counts as correct. "
    "For short answers, accept only if semantically equivalent without requiring extra assumptions. "
    "Return strictly JSON."
)


def format_question_block(question: str, choices: list[str], task_type: str) -> str:
    del task_type
    rows = [f"Question: {question}"]
    if choices:
        rows.append("Options:")
        rows.extend(choices)
    return "\n".join(rows)


def build_solver_system(strategy: int, penalty: float) -> str:
    return _solver_instruction(strategy, penalty)


def build_solver_prompt(question: str, choices: list[str], task_type: str, strategy: int, penalty: float) -> str:
    del strategy, penalty
    return format_question_block(question, choices, task_type)


def build_parser_prompt(question: str, choices: list[str], solver_output: str) -> str:
    del question
    option_block = "\n".join(choices) if choices else ""
    return (
        f"OPTIONS:\n{option_block}\n"
        f"MODEL RESPONSE:\n{solver_output}\n"
        "TASK: Output exactly one valid JSON object.\n"
        "You MUST output exactly this structure, with no deviation:\n"
        "{\n"
        '  "final_answer": "string",\n'
        '  "confidence": number or null,\n'
        '  "reasoning_trace": "string"\n'
        "}\n"
        "Rules:\n"
        "- Output must start with '{' and end with '}'.\n"
        "- No text before or after.\n"
        "- No code fences, comments, or explanations.\n"
        "- 'final_answer' must be a non-empty string or 'ABSTAIN'.\n"
        "- 'confidence' must be a numeric literal in [0,1] or null.\n"
        "- 'reasoning_trace' must be a string copied verbatim from the model response."
    )


def build_parser_repair_prompt(question: str, choices: list[str], solver_output: str, parser_output: str) -> str:
    del question
    option_block = "\n".join(choices) if choices else ""
    return (
        f"OPTIONS:\n{option_block}\n"
        f"MODEL RESPONSE:\n{solver_output}\n"
        f"PREVIOUS PARSER OUTPUT:\n{parser_output}\n"
        "TASK: The previous parser output was invalid.\n"
        "You MUST output exactly this JSON object, with no deviation:\n"
        "{\n"
        '  "final_answer": "string",\n'
        '  "confidence": number or null,\n'
        '  "reasoning_trace": "string"\n'
        "}\n"
        "Rules:\n"
        "- Output must start with '{' and end with '}'.\n"
        "- No text before or after.\n"
        "- No code fences, comments, or explanations.\n"
        "- 'final_answer' must be a non-empty string or 'ABSTAIN'.\n"
        "- 'confidence' must be a numeric literal in [0,1] or null.\n"
        "- 'reasoning_trace' must be a string copied verbatim from the model response."
    )


def build_judge_prompt(question: str, choices: list[str], gold_answer: str, model_final_answer: str) -> str:
    option_block = "\n".join(choices) if choices else ""
    return (
        f"QUESTION:\n{question}\n"
        f"OPTIONS:\n{option_block}\n"
        f"GOLD ANSWER:\n{gold_answer}\n"
        f"MODEL FINAL ANSWER:\n{model_final_answer}\n"
        "TASK: Output exactly one valid JSON object.\n"
        "You MUST output exactly this structure, with no deviation:\n"
        "{\n"
        '  "correct": true | false,\n'
        '  "normalized_model_answer": "string"\n'
        "}\n"
        "Rules:\n"
        "- Output must start with '{' and end with '}'.\n"
        "- No text before or after.\n"
        "- No code fences, comments, or explanations.\n"
        "- 'correct' must be a boolean literal (true or false).\n"
        "- 'normalized_model_answer' must be a string (can be empty)."
    )

