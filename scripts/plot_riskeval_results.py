from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sorted_penalties(summary: dict) -> list[float]:
    return [float(x) for x in summary["penalties"]]


def _penalty_key(penalty: float) -> str:
    return f"{float(penalty):.6f}"


def _metric_series(summary: dict, metric_name: str) -> list[float]:
    penalties = _sorted_penalties(summary)
    return [float(summary["metrics_by_penalty"][_penalty_key(p)][metric_name]) for p in penalties]


def _rows_for_penalty(rows: list[dict], penalty: float) -> list[dict]:
    key = _penalty_key(penalty)
    return [r for r in rows if _penalty_key(float(r["penalty"])) == key]


def _avg_confidence_by_penalty(rows: list[dict], penalties: list[float]) -> list[float]:
    out = []
    for penalty in penalties:
        subset = [
            r
            for r in _rows_for_penalty(rows, penalty)
            if str(r.get("model_decision", "")).upper() == "ANSWER"
        ]
        confidences = [
            float(r["confidence_prob"])
            for r in subset
            if r.get("confidence_prob") is not None
        ]
        out.append(sum(confidences) / len(confidences) if confidences else float("nan"))
    return out


def _high_penalty_rows(rows: list[dict], threshold: float = 10.0) -> list[dict]:
    return [r for r in rows if float(r["penalty"]) >= threshold]


def _high_penalty_summary(summary: dict, metric_name: str, threshold: float = 10.0) -> float:
    penalties = [p for p in _sorted_penalties(summary) if p >= threshold]
    values = [float(summary["metrics_by_penalty"][_penalty_key(p)][metric_name]) for p in penalties]
    return sum(values) / len(values)


def _confidence_mean(rows: list[dict]) -> float:
    seen: dict[str, float] = {}
    for row in rows:
        if str(row.get("model_decision", "")).upper() != "ANSWER":
            continue
        if row.get("confidence_prob") is None:
            continue
        seen.setdefault(str(row["qid"]), float(row["confidence_prob"]))
    if not seen:
        return float("nan")
    return sum(seen.values()) / len(seen)


def _oracle_norm_utility_high_penalty(rows: list[dict], threshold: float = 10.0) -> float:
    subset = _high_penalty_rows(rows, threshold)
    vals = [
        float(r["oracle_utility"]) / (1.0 + float(r["penalty"]))
        for r in subset
        if r.get("oracle_utility") is not None
    ]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _write_table(path: Path, table_rows: list[dict]) -> None:
    if not table_rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)


def _penalty_ticklabels(penalties: list[float]) -> list[str]:
    labels = []
    for x in penalties:
        if float(x).is_integer():
            labels.append(str(int(x)))
        else:
            labels.append(str(x).rstrip("0").rstrip("."))
    return labels


def _fmt_metric(value: float) -> str:
    return "nan" if value != value else f"{value:.3f}"


def _abstentions_by_penalty(rows: list[dict], penalties: list[float]) -> list[int]:
    counts = []
    for penalty in penalties:
        counts.append(
            sum(
                1
                for r in _rows_for_penalty(rows, penalty)
                if str(r.get("model_decision", "")).upper() == "ABSTAIN"
            )
        )
    return counts


def _abstention_rate_by_penalty(rows: list[dict], penalties: list[float]) -> list[float]:
    rates = []
    for penalty in penalties:
        subset = _rows_for_penalty(rows, penalty)
        if not subset:
            rates.append(float("nan"))
            continue
        abstain_count = sum(1 for r in subset if str(r.get("model_decision", "")).upper() == "ABSTAIN")
        rates.append(abstain_count / len(subset))
    return rates


def _confidence_bin_label(start: float, end: float) -> str:
    if end >= 1.0:
        return f"[{start:.1f},1.0]"
    return f"[{start:.1f},{end:.1f})"


def _abstentions_by_confidence_bin(rows: list[dict], bin_width: float = 0.1) -> tuple[list[str], list[int]]:
    num_bins = int(round(1.0 / bin_width))
    counts = [0] * num_bins
    labels = [
        _confidence_bin_label(i * bin_width, min((i + 1) * bin_width, 1.0))
        for i in range(num_bins)
    ]
    for row in rows:
        if str(row.get("model_decision", "")).upper() != "ABSTAIN":
            continue
        if row.get("confidence_prob") is None:
            continue
        confidence = max(0.0, min(1.0, float(row["confidence_prob"])))
        idx = min(int(math.floor(confidence / bin_width)), num_bins - 1)
        counts[idx] += 1
    return labels, counts


def _answered_count_by_confidence_bin(rows: list[dict], bin_width: float = 0.1) -> tuple[list[str], list[int]]:
    num_bins = int(round(1.0 / bin_width))
    counts = [0] * num_bins
    labels = [
        _confidence_bin_label(i * bin_width, min((i + 1) * bin_width, 1.0))
        for i in range(num_bins)
    ]
    for row in rows:
        if str(row.get("model_decision", "")).upper() != "ANSWER":
            continue
        if row.get("confidence_prob") is None:
            continue
        confidence = max(0.0, min(1.0, float(row["confidence_prob"])))
        idx = min(int(math.floor(confidence / bin_width)), num_bins - 1)
        counts[idx] += 1
    return labels, counts


def _answered_accuracy_by_confidence_bin(rows: list[dict], bin_width: float = 0.1) -> tuple[list[str], list[float]]:
    num_bins = int(round(1.0 / bin_width))
    correct_counts = [0] * num_bins
    total_counts = [0] * num_bins
    labels = [
        _confidence_bin_label(i * bin_width, min((i + 1) * bin_width, 1.0))
        for i in range(num_bins)
    ]
    for row in rows:
        if str(row.get("model_decision", "")).upper() != "ANSWER":
            continue
        if row.get("confidence_prob") is None:
            continue
        if row.get("solver_correct") is None:
            continue
        confidence = max(0.0, min(1.0, float(row["confidence_prob"])))
        idx = min(int(math.floor(confidence / bin_width)), num_bins - 1)
        total_counts[idx] += 1
        if bool(row["solver_correct"]):
            correct_counts[idx] += 1
    accuracies = [
        (correct_counts[i] / total_counts[i]) if total_counts[i] else float("nan")
        for i in range(num_bins)
    ]
    return labels, accuracies


def _plot_metric_grid(out_path: Path, dataset_name: str, runs: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    metrics = [
        ("Average Confidence", "avg_confidence"),
        ("Normalized Average Utility", "avg_normalized_utility"),
        ("Policy Consistency", "policy_consistency"),
    ]

    for ax, (title, metric_key) in zip(axes, metrics):
        for run in runs:
            penalties = run["penalties"]
            x = list(range(len(penalties)))
            if metric_key == "avg_confidence":
                y = run[metric_key]
            else:
                y = _metric_series(run["summary"], metric_key)
            ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=run["label"])
        ax.set_title(title)
        ax.set_xlabel("Penalty")
        ax.set_xticks(list(range(len(run["penalties"]))))
        ax.set_xticklabels(_penalty_ticklabels(run["penalties"]), rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(dataset_name)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_grid(out_path: Path, dataset_name: str, runs: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    metrics = [
        ("AUARC", "auarc"),
        ("ECE-10", "ece_10"),
        ("Brier", "brier"),
    ]

    for ax, (title, metric_key) in zip(axes, metrics):
        for run in runs:
            penalties = run["penalties"]
            x = list(range(len(penalties)))
            y = _metric_series(run["summary"], metric_key)
            ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=run["label"])
        ax.set_title(title)
        ax.set_xlabel("Penalty")
        ax.set_xticks(list(range(len(run["penalties"]))))
        ax.set_xticklabels(_penalty_ticklabels(run["penalties"]), rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(dataset_name)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_abstention_penalty_bars(out_path: Path, dataset_name: str, runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    penalties = runs[0]["penalties"]
    x = list(range(len(penalties)))
    width = 0.8 / max(len(runs), 1)

    for idx, run in enumerate(runs):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        y = _abstentions_by_penalty(run["rows"], penalties)
        ax.bar(offsets, y, width=width, label=run["label"], alpha=0.85)

    ax.set_title(f"{dataset_name}: Abstentions by Penalty")
    ax.set_xlabel("Penalty")
    ax.set_ylabel("Abstain Count")
    ax.set_xticks(x)
    ax.set_xticklabels(_penalty_ticklabels(penalties), rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_answered_count_confidence_bars(out_path: Path, dataset_name: str, runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    labels, _ = _answered_count_by_confidence_bin(runs[0]["rows"])
    x = list(range(len(labels)))
    width = 0.8 / max(len(runs), 1)

    for idx, run in enumerate(runs):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        _, y = _answered_count_by_confidence_bin(run["rows"])
        ax.bar(offsets, y, width=width, label=run["label"], alpha=0.85)

    ax.set_title(f"{dataset_name}: Answered Count by Confidence Bin")
    ax.set_xlabel("Confidence Bin")
    ax.set_ylabel("Answered Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_abstention_rate_penalty_lines(out_path: Path, dataset_name: str, runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)

    for run in runs:
        penalties = run["penalties"]
        x = list(range(len(penalties)))
        y = _abstention_rate_by_penalty(run["rows"], penalties)
        ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=run["label"])

    ax.set_title(f"{dataset_name}: Abstention Rate by Penalty")
    ax.set_xlabel("Penalty")
    ax.set_ylabel("Abstention Rate")
    ax.set_xticks(list(range(len(runs[0]["penalties"]))))
    ax.set_xticklabels(_penalty_ticklabels(runs[0]["penalties"]), rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_answered_accuracy_confidence_lines(out_path: Path, dataset_name: str, runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    labels, _ = _answered_accuracy_by_confidence_bin(runs[0]["rows"])
    x = list(range(len(labels)))

    for run in runs:
        _, y = _answered_accuracy_by_confidence_bin(run["rows"])
        ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=run["label"])

    ax.set_title(f"{dataset_name}: Answered Accuracy by Confidence Bin")
    ax.set_xlabel("Confidence Bin")
    ax.set_ylabel("Answered Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_analysis_dashboard(out_path: Path, dataset_name: str, runs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    ax = axes[0][0]
    penalties = runs[0]["penalties"]
    x = list(range(len(penalties)))
    width = 0.8 / max(len(runs), 1)
    for idx, run in enumerate(runs):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        y = _abstentions_by_penalty(run["rows"], penalties)
        ax.bar(offsets, y, width=width, label=run["label"], alpha=0.85)
    ax.set_title("Abstentions by Penalty")
    ax.set_xlabel("Penalty")
    ax.set_ylabel("Abstain Count")
    ax.set_xticks(x)
    ax.set_xticklabels(_penalty_ticklabels(penalties), rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0][1]
    for run in runs:
        y = _abstention_rate_by_penalty(run["rows"], penalties)
        ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=run["label"])
    ax.set_title("Abstention Rate by Penalty")
    ax.set_xlabel("Penalty")
    ax.set_ylabel("Abstention Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(_penalty_ticklabels(penalties), rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    conf_labels, _ = _answered_count_by_confidence_bin(runs[0]["rows"])
    conf_x = list(range(len(conf_labels)))
    ax = axes[1][0]
    width = 0.8 / max(len(runs), 1)
    for idx, run in enumerate(runs):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in conf_x]
        _, y = _answered_count_by_confidence_bin(run["rows"])
        ax.bar(offsets, y, width=width, label=run["label"], alpha=0.85)
    ax.set_title("Answered Count by Confidence Bin")
    ax.set_xlabel("Confidence Bin")
    ax.set_ylabel("Answered Count")
    ax.set_xticks(conf_x)
    ax.set_xticklabels(conf_labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1][1]
    for run in runs:
        _, y = _answered_accuracy_by_confidence_bin(run["rows"])
        ax.plot(conf_x, y, marker="o", linewidth=2, markersize=4, label=run["label"])
    ax.set_title("Answered Accuracy by Confidence Bin")
    ax.set_xlabel("Confidence Bin")
    ax.set_ylabel("Answered Accuracy")
    ax.set_xticks(conf_x)
    ax.set_xticklabels(conf_labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))
    parser = argparse.ArgumentParser(description="Plot RiskEval experiment outputs")
    parser.add_argument("--dataset-name", required=True, help="Display name such as GPQA Diamond")
    parser.add_argument("--output-dir", required=True, help="Where to save figures and tables")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification in the form label=/abs/path/to/output_dir",
    )
    args = parser.parse_args()

    plot_runs = []
    table_rows = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Invalid --run value: {item}")
        label, run_dir_str = item.split("=", 1)
        run_dir = Path(run_dir_str)
        summary = _load_json(run_dir / "summary.json")
        rows = _load_jsonl(run_dir / "example_runs.jsonl")
        penalties = _sorted_penalties(summary)
        avg_conf = _avg_confidence_by_penalty(rows, penalties)

        plot_runs.append(
            {
                "label": label,
                "summary": summary,
                "rows": rows,
                "penalties": penalties,
                "avg_confidence": avg_conf,
            }
        )

        high_penalty_norm_u = _high_penalty_summary(summary, "avg_normalized_utility")
        high_penalty_pol = _high_penalty_summary(summary, "policy_consistency")
        high_penalty_reg = _high_penalty_summary(summary, "avg_normalized_regret")
        oracle_norm_u = _oracle_norm_utility_high_penalty(rows)

        table_rows.append(
            {
                "model": label,
                "auarc": _fmt_metric(_metric_series(summary, "auarc")[0]),
                "ece_10": _fmt_metric(_metric_series(summary, "ece_10")[0]),
                "brier": _fmt_metric(_metric_series(summary, "brier")[0]),
                "avg_confidence": _fmt_metric(_confidence_mean(rows)),
                "policy_consistency_lambda_ge_10": _fmt_metric(high_penalty_pol),
                "normalized_regret_lambda_ge_10": _fmt_metric(high_penalty_reg),
                "norm_utility_pi_M_lambda_ge_10": _fmt_metric(high_penalty_norm_u),
                "norm_utility_pi_star_lambda_ge_10": _fmt_metric(oracle_norm_u),
                "pi_star_gain": _fmt_metric(oracle_norm_u - high_penalty_norm_u),
            }
        )

    _plot_metric_grid(output_dir / "main_metrics.png", args.dataset_name, plot_runs)
    _plot_calibration_grid(output_dir / "calibration_metrics.png", args.dataset_name, plot_runs)
    _plot_abstention_penalty_bars(output_dir / "abstention_by_penalty.png", args.dataset_name, plot_runs)
    _plot_answered_count_confidence_bars(output_dir / "answered_count_by_confidence_bin.png", args.dataset_name, plot_runs)
    _plot_abstention_rate_penalty_lines(output_dir / "abstention_rate_by_penalty.png", args.dataset_name, plot_runs)
    _plot_answered_accuracy_confidence_lines(
        output_dir / "answered_accuracy_by_confidence_bin.png",
        args.dataset_name,
        plot_runs,
    )
    _plot_analysis_dashboard(output_dir / "analysis_dashboard.png", args.dataset_name, plot_runs)
    _write_table(output_dir / "metrics_table.csv", table_rows)
    (output_dir / "metrics_table.json").write_text(json.dumps(table_rows, indent=2), encoding="utf-8")

    print(output_dir / "main_metrics.png")
    print(output_dir / "calibration_metrics.png")
    print(output_dir / "abstention_by_penalty.png")
    print(output_dir / "answered_count_by_confidence_bin.png")
    print(output_dir / "abstention_rate_by_penalty.png")
    print(output_dir / "answered_accuracy_by_confidence_bin.png")
    print(output_dir / "analysis_dashboard.png")
    print(output_dir / "metrics_table.csv")


if __name__ == "__main__":
    main()
