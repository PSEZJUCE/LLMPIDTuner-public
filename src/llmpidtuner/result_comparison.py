from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmpidtuner.metrics import calculate_performance_metrics


@dataclass(frozen=True)
class ResponseCurve:
    label: str
    path: Path
    time: np.ndarray
    setpoint: np.ndarray
    output: np.ndarray
    source_file: Path


@dataclass(frozen=True)
class ResponseSummary:
    label: str
    result_path: Path
    source_file: Path
    overshoot: float
    steady_state_error: float


def resolve_batch_group_paths(
    batch_paths: list[str | Path],
    group: int,
) -> list[Path]:
    if group < 1:
        raise ValueError("group must be a positive integer.")

    prefix = f"group_{group:03d}_"
    resolved: list[Path] = []
    for batch_path in batch_paths:
        root = Path(batch_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Batch result path does not exist: {root}")
        matches = sorted(
            child
            for child in root.iterdir()
            if child.is_dir()
            and child.name.startswith(prefix)
            and (child / "value_curve.txt").is_file()
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {prefix}* result under {root}, found {len(matches)}."
            )
        resolved.append(matches[0])
    return resolved


def compare_result_paths(
    result_paths: list[str | Path],
    labels: list[str] | None = None,
    output_path: str | Path = "runs/result_comparison.png",
    origin_label: str = "Origin",
    title: str | None = None,
    metrics_csv: str | Path | None = None,
    x_break: tuple[float, float] | None = None,
) -> tuple[Path, Path | None, list[ResponseSummary]]:
    paths = [Path(path) for path in result_paths]
    if not paths:
        raise ValueError("At least one result path is required.")

    display_labels = labels or [_default_label(path) for path in paths]
    if len(display_labels) != len(paths):
        raise ValueError("labels count must match result_paths count.")

    origin_curve = _load_curve(paths[0] / "value_curve.txt", origin_label, paths[0])
    method_curves = [
        _load_curve(_latest_response_curve(path), label, path)
        for path, label in zip(paths, display_labels, strict=True)
    ]

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    if x_break is None:
        _plot_response_comparison(origin_curve, method_curves, figure_path, title=title)
    else:
        _plot_broken_response_comparison(
            origin_curve,
            method_curves,
            figure_path,
            x_break=x_break,
            title=title,
        )

    summaries = [_summarize_curve(curve) for curve in method_curves]
    csv_path = Path(metrics_csv) if metrics_csv else None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary.__dict__ for summary in summaries]).to_csv(csv_path, index=False)

    return figure_path, csv_path, summaries


def _latest_response_curve(result_path: Path) -> Path:
    if not result_path.exists():
        raise FileNotFoundError(f"Result path does not exist: {result_path}")
    iteration_files = sorted(
        result_path.glob("value_curve_iteration_*.txt"),
        key=_iteration_index,
    )
    if iteration_files:
        return iteration_files[-1]
    fallback = result_path / "value_curve.txt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No value_curve files found under: {result_path}")


def _load_curve(curve_path: Path, label: str, result_path: Path) -> ResponseCurve:
    if not curve_path.exists():
        raise FileNotFoundError(f"Curve file does not exist: {curve_path}")
    data = np.loadtxt(curve_path, skiprows=2)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return ResponseCurve(
        label=label,
        path=result_path,
        time=data[:, 0],
        setpoint=data[:, 1],
        output=data[:, 2],
        source_file=curve_path,
    )


def _plot_response_comparison(
    origin_curve: ResponseCurve,
    method_curves: list[ResponseCurve],
    output_path: Path,
    title: str | None = None,
) -> None:
    publication_rc = {
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.1,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
    colors = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#D55E00"]
    with plt.rc_context(publication_rc):
        fig, axis = plt.subplots(figsize=(6.3, 4.5))
        axis.plot(
            origin_curve.time,
            origin_curve.setpoint,
            color="#222222",
            linewidth=1.1,
            label="Setpoint",
        )
        axis.plot(
            origin_curve.time,
            origin_curve.output,
            color="#D55E00",
            linewidth=1.1,
            label=origin_curve.label,
        )

        for index, curve in enumerate(method_curves):
            axis.plot(
                curve.time,
                curve.output,
                color=colors[index % len(colors)],
                linewidth=1.1,
                label=curve.label,
            )

        axis.set_xlabel("Time")
        axis.set_ylabel("Output")
        if title:
            axis.set_title(title)
        xmax = max(
            float(origin_curve.time[-1]),
            *(float(curve.time[-1]) for curve in method_curves),
        )
        axis.set_xlim(0, xmax)
        axis.set_ylim(bottom=0)
        axis.margins(x=0, y=0)
        axis.grid(False)
        axis.legend(
            frameon=False,
            loc="best",
            handlelength=2.2,
            handletextpad=0.4,
        )
        for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick.set_fontweight("bold")
            tick.set_fontfamily("Arial")
        for spine in axis.spines.values():
            spine.set_linewidth(1.1)
        axis.tick_params(direction="out", width=1.0, length=3.5)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _plot_broken_response_comparison(
    origin_curve: ResponseCurve,
    method_curves: list[ResponseCurve],
    output_path: Path,
    x_break: tuple[float, float],
    title: str | None = None,
) -> None:
    publication_rc = {
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.1,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
    colors = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#D55E00"]
    xmax = max(
        float(origin_curve.time[-1]),
        *(float(curve.time[-1]) for curve in method_curves),
    )
    break_start, break_end = x_break
    if not 0.0 < break_start < break_end < xmax:
        raise ValueError(
            "x_break must satisfy 0 < start < end < maximum response time "
            f"({xmax:g})."
        )

    with plt.rc_context(publication_rc):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(6.3, 4.5),
            sharey=True,
            gridspec_kw={
                "width_ratios": (break_start, xmax - break_end),
                "wspace": 0.18,
            },
        )
        for axis in axes:
            axis.plot(
                origin_curve.time,
                origin_curve.setpoint,
                color="#222222",
                linewidth=1.1,
                label="Setpoint",
            )
            axis.plot(
                origin_curve.time,
                origin_curve.output,
                color="#D55E00",
                linewidth=1.1,
                label=origin_curve.label,
            )
            for index, curve in enumerate(method_curves):
                axis.plot(
                    curve.time,
                    curve.output,
                    color=colors[index % len(colors)],
                    linewidth=1.1,
                    label=curve.label,
                )
            axis.margins(x=0, y=0)
            axis.grid(False)
            for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
                tick.set_fontweight("bold")
                tick.set_fontfamily("Arial")
            for spine in axis.spines.values():
                spine.set_linewidth(1.1)
            axis.tick_params(direction="out", width=1.0, length=3.5)

        axes[0].set_xlim(0, break_start)
        axes[1].set_xlim(break_end, xmax)
        ymax = max(
            float(np.max(origin_curve.setpoint)),
            float(np.max(origin_curve.output)),
            *(float(np.max(curve.output)) for curve in method_curves),
        )
        y_padding = max(0.02, 0.05 * ymax)
        axes[0].set_ylim(0, ymax + y_padding)
        axes[0].set_ylabel("Output")
        figure.supxlabel("Time", fontweight="bold")
        if title:
            figure.suptitle(title)

        axes[0].spines["right"].set_visible(False)
        axes[1].spines["left"].set_visible(False)
        axes[1].tick_params(axis="y", left=False, labelleft=False)
        _draw_break_marks(axes[0], axes[1])
        axes[0].legend(
            frameon=False,
            loc="lower right",
            handlelength=2.2,
            handletextpad=0.4,
        )
        figure.subplots_adjust(
            left=0.12,
            right=0.98,
            bottom=0.14,
            top=0.90 if title else 0.98,
            wspace=0.18,
        )
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def _draw_break_marks(left_axis: plt.Axes, right_axis: plt.Axes) -> None:
    diagonal = 0.012
    mark_style = {"color": "black", "clip_on": False, "linewidth": 1.1}
    left_axis.plot(
        (1 - diagonal, 1 + diagonal),
        (-diagonal, diagonal),
        transform=left_axis.transAxes,
        **mark_style,
    )
    left_axis.plot(
        (1 - diagonal, 1 + diagonal),
        (1 - diagonal, 1 + diagonal),
        transform=left_axis.transAxes,
        **mark_style,
    )
    right_axis.plot(
        (-diagonal, diagonal),
        (-diagonal, diagonal),
        transform=right_axis.transAxes,
        **mark_style,
    )
    right_axis.plot(
        (-diagonal, diagonal),
        (1 - diagonal, 1 + diagonal),
        transform=right_axis.transAxes,
        **mark_style,
    )


def _summarize_curve(curve: ResponseCurve) -> ResponseSummary:
    setpoint = float(curve.setpoint[-1])
    metrics = calculate_performance_metrics(curve.output, setpoint)
    return ResponseSummary(
        label=curve.label,
        result_path=curve.path,
        source_file=curve.source_file,
        overshoot=metrics.overshoot,
        steady_state_error=metrics.steady_state_error,
    )


def _iteration_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError as error:
        raise ValueError(f"Cannot parse iteration index from {path}") from error


def _default_label(path: Path) -> str:
    metadata_path = path / "llm_metadata.txt"
    if metadata_path.exists():
        metadata = _read_metadata(metadata_path)
        if metadata.get("profile"):
            return metadata["profile"]
        if metadata.get("model"):
            return metadata["model"]
    if (path / "imc_metadata.txt").exists():
        return "IMC"
    return path.name


def _read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata
