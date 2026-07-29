from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from llmpidtuner.metrics import calculate_response_metrics

OVERSHOOT_BINS = [
    ("0-5%", 5.0),
    ("5-10%", 10.0),
    ("10-15%", 15.0),
    (">15%", None),
]
STEADY_STATE_ERROR_BINS = [
    ("<=0.001", 0.001),
    ("<=0.01", 0.01),
    ("<=0.1", 0.1),
    ("<=1", 1.0),
    (">1", None),
]
PUBLICATION_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
]
LLM_FAILURE_REASONS = {
    "context_length_exceeded",
    "pid_parse_failed",
    "llm_call_failed",
}


@dataclass(frozen=True)
class BatchCaseResult:
    batch_path: Path
    label: str
    case_id: str
    final_iteration: int
    convergence_iteration: int | None
    converged: bool
    final_overshoot: float
    final_steady_state_error: float
    demonstration_protocol: str = "legacy-unknown"
    stop_reason: str = ""
    analysis_horizon: int | None = None
    failed_next_llm_call: int | None = None
    max_iterations: int | None = None


@dataclass(frozen=True)
class BatchSummary:
    label: str
    batch_path: Path
    demonstration_protocol: str
    total: int
    converged: int
    unconverged: int
    llm_failed: int
    context_length_exceeded: int
    convergence_rate: float
    mean_convergence_iteration: float | None
    mean_final_overshoot: float
    median_final_overshoot: float
    mean_final_steady_state_error: float
    median_final_steady_state_error: float
    analysis_horizon: int | None = None


def analyze_batch_paths(
    paths: list[str | Path],
    labels: list[str] | None = None,
    success_overshoot: float = 15.0,
    success_steady_state_error: float = 1.0,
    allow_mixed_protocols: bool = False,
    horizon: int | None = None,
) -> tuple[list[BatchCaseResult], list[BatchSummary]]:
    batch_paths = [Path(path) for path in paths]
    batch_labels = labels or [_default_label(path) for path in batch_paths]
    batch_labels = _dedupe_labels(batch_labels, batch_paths)
    protocols = {
        protocol
        for path in batch_paths
        if (protocol := _read_demonstration_protocol(path)) != "not-applicable"
    }
    if len(protocols) > 1 and not allow_mixed_protocols:
        joined = ", ".join(sorted(protocols))
        raise ValueError(
            f"Refusing to combine batch results from different demonstration protocols: {joined}"
        )

    all_results: list[BatchCaseResult] = []
    for path, label in zip(batch_paths, batch_labels, strict=True):
        all_results.extend(
            analyze_batch_path(
                path,
                label,
                success_overshoot=success_overshoot,
                success_steady_state_error=success_steady_state_error,
                horizon=horizon,
            )
        )

    summaries = summarize_batch_results(all_results)
    return all_results, summaries


def analyze_batch_path(
    batch_path: str | Path,
    label: str | None = None,
    success_overshoot: float = 15.0,
    success_steady_state_error: float = 1.0,
    horizon: int | None = None,
) -> list[BatchCaseResult]:
    root = Path(batch_path)
    if horizon is not None and horizon <= 0:
        raise ValueError("Analysis horizon must be a positive iteration count.")
    if not root.exists():
        raise FileNotFoundError(f"Batch result path does not exist: {root}")

    result_dirs = _find_result_dirs(root)
    if not result_dirs:
        raise ValueError(f"No value_curve files found under: {root}")

    batch_label = label or _default_label(root)
    results: list[BatchCaseResult] = []
    demonstration_protocol = _read_demonstration_protocol(root)
    for result_dir in result_dirs:
        curve_files = _find_curve_files(result_dir)
        iteration_metrics = [
            (_iteration_index(path), *_calculate_curve_metrics(path)) for path in curve_files
        ]
        iteration_metrics.sort(key=lambda item: item[0])
        if horizon is not None:
            iteration_metrics = [item for item in iteration_metrics if item[0] <= horizon]
        if not iteration_metrics:
            raise ValueError(f"No curve is available within horizon={horizon}: {result_dir}")

        convergence_iteration: int | None = None
        for iteration, overshoot, steady_state_error, settled in iteration_metrics:
            if (
                overshoot < success_overshoot
                and steady_state_error < success_steady_state_error
                and settled
            ):
                convergence_iteration = iteration
                break

        final_iteration, final_overshoot, final_steady_state_error, _ = iteration_metrics[-1]
        run_status = _read_run_status(result_dir)
        configured_max = _optional_int(run_status.get("max_iterations"))
        effective_max = (
            min(configured_max, horizon) if configured_max is not None and horizon else horizon
        )
        results.append(
            BatchCaseResult(
                batch_path=root,
                label=batch_label,
                case_id=result_dir.name,
                final_iteration=final_iteration,
                convergence_iteration=convergence_iteration,
                converged=convergence_iteration is not None,
                final_overshoot=final_overshoot,
                final_steady_state_error=final_steady_state_error,
                stop_reason=str(run_status.get("stop_reason", "")),
                demonstration_protocol=demonstration_protocol,
                analysis_horizon=horizon,
                failed_next_llm_call=_optional_int(run_status.get("failed_next_llm_call")),
                max_iterations=effective_max if horizon is not None else configured_max,
            )
        )

    return results


def summarize_batch_results(results: list[BatchCaseResult]) -> list[BatchSummary]:
    summaries: list[BatchSummary] = []
    labels = list(dict.fromkeys(result.label for result in results))
    for label in labels:
        group = [result for result in results if result.label == label]
        converged = [result for result in group if result.converged]
        convergence_iterations = [
            result.convergence_iteration
            for result in converged
            if result.convergence_iteration is not None
        ]
        overshoots = [result.final_overshoot for result in group]
        steady_state_errors = [result.final_steady_state_error for result in group]

        summaries.append(
            BatchSummary(
                label=label,
                batch_path=group[0].batch_path,
                total=len(group),
                demonstration_protocol=group[0].demonstration_protocol,
                converged=len(converged),
                unconverged=len(group) - len(converged),
                llm_failed=sum(result.stop_reason in LLM_FAILURE_REASONS for result in group),
                context_length_exceeded=sum(
                    result.stop_reason == "context_length_exceeded" for result in group
                ),
                convergence_rate=len(converged) / len(group) if group else 0.0,
                mean_convergence_iteration=(
                    float(np.mean(convergence_iterations)) if convergence_iterations else None
                ),
                mean_final_overshoot=float(np.mean(overshoots)),
                median_final_overshoot=float(np.median(overshoots)),
                mean_final_steady_state_error=float(np.mean(steady_state_errors)),
                median_final_steady_state_error=float(np.median(steady_state_errors)),
                analysis_horizon=group[0].analysis_horizon,
            )
        )
    return summaries


def _read_demonstration_protocol(batch_path: str | Path) -> str:
    root = Path(batch_path)
    metadata_path = root / "demonstration_metadata.yaml"
    if metadata_path.is_file():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        return str(metadata.get("demonstration_protocol") or "legacy-unknown")
    if (root / "imc_metadata.txt").is_file():
        return "not-applicable"

    group_protocols = set()
    for group_metadata_path in root.glob("*/demonstration_metadata.yaml"):
        metadata = yaml.safe_load(group_metadata_path.read_text(encoding="utf-8")) or {}
        protocol = metadata.get("demonstration_protocol")
        if protocol:
            group_protocols.add(str(protocol))
    if len(group_protocols) > 1:
        joined = ", ".join(sorted(group_protocols))
        raise ValueError(f"Batch result contains multiple demonstration protocols: {joined}")
    if group_protocols:
        return next(iter(group_protocols))
    if any(root.glob("*/imc_metadata.txt")):
        return "not-applicable"
    return "legacy-unknown"


def write_batch_analysis(
    results: list[BatchCaseResult],
    summaries: list[BatchSummary],
    output_path: str | Path,
    csv_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_batch_analysis(results, summaries, figure_path)

    summary_csv_path = Path(csv_path) if csv_path else None
    if summary_csv_path is not None:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary.__dict__ for summary in summaries]).to_csv(
            summary_csv_path, index=False
        )
    return figure_path, summary_csv_path


def write_case_details(results: list[BatchCaseResult], csv_path: str | Path) -> Path:
    detail_path = Path(csv_path)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.__dict__ for result in results]).to_csv(detail_path, index=False)
    return detail_path


def _plot_batch_analysis(
    results: list[BatchCaseResult],
    summaries: list[BatchSummary],
    output_path: Path,
) -> None:
    labels = [summary.label for summary in summaries]
    iteration_labels = _publication_iteration_categories(results)

    publication_rc = {
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.1,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
    with plt.rc_context(publication_rc):
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
        _plot_convergence_percentages(axes[0], results, labels, iteration_labels)
        _plot_metric_percentages(
            axes[1],
            results,
            labels,
            metric_name="final_overshoot",
            bins=OVERSHOOT_BINS,
            xlabel="Overshoot Range (%)",
        )
        _plot_metric_percentages(
            axes[2],
            results,
            labels,
            metric_name="final_steady_state_error",
            bins=STEADY_STATE_ERROR_BINS,
            xlabel="Steady-State Error (%)",
        )
        fig.subplots_adjust(wspace=0.30)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _plot_convergence_percentages(
    axis: plt.Axes,
    results: list[BatchCaseResult],
    labels: list[str],
    iteration_labels: list[str],
) -> None:
    x = np.arange(len(iteration_labels))
    width = 0.72 / max(len(labels), 1)

    for index, label in enumerate(labels):
        group = [result for result in results if result.label == label]
        percentages = [
            100.0
            * sum(_publication_iteration_category(result) == category for result in group)
            / len(group)
            for category in iteration_labels
        ]
        offset = (index - (len(labels) - 1) / 2) * width
        bars = axis.bar(
            x + offset,
            percentages,
            width,
            label=label,
            color=PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)],
            edgecolor="#333333",
            linewidth=0.5,
        )
        _label_percentage_bars(axis, bars)

    axis.set_xlabel("Iterations")
    axis.set_ylabel("Percentage (%)")
    axis.set_xticks(x)
    axis.set_xticklabels(iteration_labels)
    _finish_publication_axis(axis, labels)


def _plot_metric_percentages(
    axis: plt.Axes,
    results: list[BatchCaseResult],
    labels: list[str],
    metric_name: str,
    bins: list[tuple[str, float | None]],
    xlabel: str,
) -> None:
    bin_labels = [label for label, _ in bins]
    x = np.arange(len(bin_labels))
    width = 0.72 / max(len(labels), 1)

    for index, label in enumerate(labels):
        values = [
            float(getattr(result, metric_name)) for result in results if result.label == label
        ]
        counts = _bucket_counts(values, bins)
        percentages = [100.0 * count / len(values) for count in counts]
        offset = (index - (len(labels) - 1) / 2) * width
        bars = axis.bar(
            x + offset,
            percentages,
            width,
            label=label,
            color=PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)],
            edgecolor="#333333",
            linewidth=0.5,
        )
        _label_percentage_bars(axis, bars)

    axis.set_xlabel(xlabel)
    axis.set_ylabel("Percentage (%)")
    axis.set_xticks(x)
    axis.set_xticklabels(bin_labels)
    _finish_publication_axis(axis, labels)


def _label_percentage_bars(axis: plt.Axes, bars: object) -> None:
    heights = [bar.get_height() for bar in bars]
    labels = [
        f"{height:.0f}" if abs(height - round(height)) < 1e-9 else f"{height:.1f}"
        for height in heights
    ]
    axis.bar_label(bars, labels=labels, padding=2, fontsize=7, fontweight="bold")


def _finish_publication_axis(axis: plt.Axes, labels: list[str]) -> None:
    ymax = max((patch.get_height() for patch in axis.patches), default=0.0)
    axis.set_ylim(0, max(5.0, ymax * 1.20))
    axis.set_xlim(-0.55, len(axis.get_xticks()) - 0.45)
    axis.grid(False)
    for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick.set_fontweight("bold")
        tick.set_fontfamily("Arial")
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)
    axis.tick_params(direction="out", width=1.0, length=3.5)
    if len(labels) > 1:
        axis.legend(frameon=False, loc="upper right")


def _bucket_counts(values: list[float], bins: list[tuple[str, float | None]]) -> list[int]:
    counts = [0] * len(bins)
    for value in values:
        counts[_bucket_index(value, bins)] += 1
    return counts


def _bucket_index(value: float, bins: list[tuple[str, float | None]]) -> int:
    normalized = max(value, 0.0) if np.isfinite(value) else float("inf")
    for index, (_, upper_bound) in enumerate(bins):
        if upper_bound is None or normalized <= upper_bound:
            return index
    return len(bins) - 1


def _find_result_dirs(root: Path) -> list[Path]:
    if _find_curve_files(root):
        return [root]
    return sorted(child for child in root.iterdir() if child.is_dir() and _find_curve_files(child))


def _find_curve_files(result_dir: Path) -> list[Path]:
    files = list(result_dir.glob("value_curve.txt"))
    files.extend(result_dir.glob("value_curve_iteration_*.txt"))
    return sorted(files, key=_iteration_index)


def _iteration_index(path: Path) -> int:
    if path.name == "value_curve.txt":
        return 0
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError as error:
        raise ValueError(f"Cannot parse iteration index from {path}") from error


def _calculate_curve_metrics(curve_path: Path) -> tuple[float, float, bool]:
    data = np.loadtxt(curve_path, skiprows=2)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    time = data[:, 0]
    setpoint = float(data[-1, 1])
    output = data[:, 2]
    errors = setpoint - output
    iae = float(np.trapz(np.abs(errors), time)) if len(time) > 1 else 0.0
    metrics = calculate_response_metrics(
        time,
        output,
        np.zeros_like(output),
        errors,
        iae,
        setpoint=setpoint,
        time_delay=0.0,
        finite=bool(np.all(np.isfinite(output))),
    )
    return metrics.overshoot_pct, metrics.steady_state_error_pct, metrics.settled


def _publication_iteration_categories(results: list[BatchCaseResult]) -> list[str]:
    observed = [
        result.convergence_iteration
        for result in results
        if result.convergence_iteration is not None
    ]
    configured_max = max(
        (result.max_iterations or 0 for result in results),
        default=0,
    )
    completed = [result.final_iteration for result in results]
    maximum = max([configured_max, *observed, *completed], default=0)
    start = 0 if 0 in observed else 1
    categories = [str(iteration) for iteration in range(start, maximum + 1)]
    categories.append("Not")
    return categories


def _publication_iteration_category(result: BatchCaseResult) -> str:
    if result.converged and result.convergence_iteration is not None:
        return str(result.convergence_iteration)
    return "Not"


def _read_run_status(result_dir: Path) -> dict[str, object]:
    status_path = result_dir / "run_status.yaml"
    if not status_path.exists():
        return {}
    data = yaml.safe_load(status_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_label(path: Path) -> str:
    metadata_path = path / "llm_metadata.txt"
    if metadata_path.exists():
        metadata = _read_metadata(metadata_path)
        if metadata.get("profile"):
            return metadata["profile"]
        if metadata.get("model"):
            return metadata["model"]
    return path.name


def _read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _dedupe_labels(labels: list[str], paths: list[Path]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for label, path in zip(labels, paths, strict=True):
        counts[label] = counts.get(label, 0) + 1
        if counts[label] == 1 and labels.count(label) == 1:
            output.append(label)
        else:
            output.append(f"{label} ({path.name})")
    return output
