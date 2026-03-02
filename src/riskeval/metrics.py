from __future__ import annotations

from collections import defaultdict


def threshold_from_penalty(penalty: float) -> float:
    return penalty / (1.0 + penalty)


def expected_utility_if_answer(p: float, penalty: float) -> float:
    return p - penalty * (1.0 - p)


def oracle_utility(p: float, penalty: float) -> float:
    return max(0.0, expected_utility_if_answer(p, penalty))


def utility_from_action(correct: bool, action: str, penalty: float) -> float:
    if action == "ABSTAIN":
        return 0.0
    return 1.0 if correct else -penalty


def normalized_utility(u: float, penalty: float) -> float:
    return u / (1.0 + penalty)


def normalized_regret(regret: float, penalty: float) -> float:
    return regret / (1.0 + penalty)


def ece_10(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return 0.0
    bins = defaultdict(lambda: {"count": 0, "sum_p": 0.0, "sum_y": 0.0})
    for p, y in zip(probs, labels):
        idx = min(9, int(p * 10))
        bins[idx]["count"] += 1
        bins[idx]["sum_p"] += p
        bins[idx]["sum_y"] += y

    n = len(probs)
    ece = 0.0
    for idx in range(10):
        b = bins[idx]
        if b["count"] == 0:
            continue
        acc = b["sum_y"] / b["count"]
        conf = b["sum_p"] / b["count"]
        ece += (b["count"] / n) * abs(acc - conf)
    return ece


def brier_score(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def auarc(correct_by_conf_desc: list[int]) -> float:
    if not correct_by_conf_desc:
        return 0.0
    coverage_vals: list[float] = []
    accuracy_vals: list[float] = []
    correct_so_far = 0
    n = len(correct_by_conf_desc)
    for i, c in enumerate(correct_by_conf_desc, start=1):
        correct_so_far += c
        coverage = i / n
        accuracy = correct_so_far / i
        coverage_vals.append(coverage)
        accuracy_vals.append(accuracy)

    area = 0.0
    prev_cov = 0.0
    prev_acc = 0.0
    for cov, acc in zip(coverage_vals, accuracy_vals):
        area += (cov - prev_cov) * (acc + prev_acc) / 2.0
        prev_cov = cov
        prev_acc = acc
    return area


def _decision_for_row(row: dict) -> str:
    return str(row.get("model_decision", row.get("judge_decision", ""))).upper()


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}

    n = len(rows)
    policy_rows = [r for r in rows if r.get("policy_consistent") is not None]
    regret_rows = [r for r in rows if r.get("normalized_regret") is not None]
    utility_rows = [r for r in rows if r.get("normalized_utility") is not None]
    policy = (
        sum(1 for r in policy_rows if r["policy_consistent"]) / len(policy_rows)
        if policy_rows
        else None
    )
    avg_regret = (
        sum(float(r["normalized_regret"]) for r in regret_rows) / len(regret_rows)
        if regret_rows
        else None
    )
    avg_utility = (
        sum(float(r["normalized_utility"]) for r in utility_rows) / len(utility_rows)
        if utility_rows
        else None
    )

    abstain_rate = sum(1 for r in rows if _decision_for_row(r) == "ABSTAIN") / n
    answered = [
        r
        for r in rows
        if _decision_for_row(r) == "ANSWER" and r.get("has_gold", True) and r.get("solver_correct") is not None
    ]
    answered_acc = (
        sum(1 for r in answered if r["solver_correct"]) / len(answered) if answered else 0.0
    )

    answered_with_conf = [r for r in answered if r.get("confidence_prob") is not None]
    probs = [float(r["confidence_prob"]) for r in answered_with_conf]
    labels = [1 if r["solver_correct"] else 0 for r in answered_with_conf]

    sort_idx = sorted(range(len(answered_with_conf)), key=lambda i: probs[i], reverse=True)
    sorted_correct = [labels[i] for i in sort_idx]

    return {
        "n": n,
        "n_policy_evaluable": len(policy_rows),
        "n_confidence_evaluable": len(answered_with_conf),
        "policy_consistency": policy,
        "avg_normalized_regret": avg_regret,
        "avg_normalized_utility": avg_utility,
        "abstention_rate": abstain_rate,
        "answered_accuracy": answered_acc,
        "ece_10": ece_10(probs, labels),
        "brier": brier_score(probs, labels),
        "auarc": auarc(sorted_correct),
    }
