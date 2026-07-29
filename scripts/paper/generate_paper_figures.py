from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from llmpidtuner.batch_analysis import analyze_batch_paths, write_batch_analysis, write_case_details
from llmpidtuner.frequency_analysis import (
    FrequencyPlant,
    _adaptive_frequency_grid,
    _frequency_response,
    _summarize_frequency_response,
)
from llmpidtuner.models import PIDParams
from llmpidtuner.result_comparison import compare_result_paths, resolve_batch_group_paths

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
DEFAULT_OUTPUT = RUNS / "paper_figures"
RC = {
    "font.family": "Arial",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 1.1,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
}
TYPICAL_CASES = (
    ("fopdt_group018_delay_dominant", "first_order", 18),
    ("fopdt_group055_medium_delay", "first_order", 55),
    ("sopdt_group014_delay_dominant", "second_order", 14),
    ("sopdt_group052_medium_delay", "second_order", 52),
)

LABELS = {
    "imc": "IMC",
    "deepseek": "DeepSeek-V4-Flash",
    "qwen": "Qwen3.7-Plus",
    "base": "Base-0.6B",
    "sft": "SFT-0.6B",
    "grpo": "GRPO-0.6B",
}


def batch_root(system: str, method: str, variant: str = "full") -> Path:
    suffix = "" if variant == "full" else f"_{variant}"
    names = {
        "imc": f"{system}_100_imc",
        "deepseek": f"{system}_100_deepseek_v4_flash{suffix}",
        "qwen": f"{system}_100_qwen3_7_plus{suffix}",
        "base": f"{system}_100_base_qwen3_0p6b{suffix}",
        "sft": f"{system}_100_sft_qwen3_0p6b",
        "grpo": f"{system}_100_grpo_qwen3_0p6b",
    }
    if method not in names:
        raise ValueError(f"Unknown result method: {method}")
    path = RUNS / names[method]
    if not path.is_dir():
        raise FileNotFoundError(f"Required result directory does not exist: {path}")
    return path


def single_case_paths(
    system: str, group: int, methods: Iterable[tuple[str, str]]
) -> tuple[list[Path], list[str]]:
    selected = list(methods)
    roots = [batch_root(system, method, variant) for method, variant in selected]
    return resolve_batch_group_paths(roots, group), [LABELS[method] for method, _ in selected]


def response_figure(
    output_dir: Path,
    stem: str,
    system: str,
    group: int,
    methods: list[tuple[str, str]],
    x_break: tuple[float, float] | None = None,
) -> None:
    paths, labels = single_case_paths(system, group, methods)
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_result_paths(
        paths,
        labels=labels,
        output_path=output_dir / f"{stem}.png",
        origin_label="Initial",
        title=None,
        metrics_csv=output_dir / f"{stem}_metrics.csv",
        x_break=x_break,
    )


def batch_figure(
    output_dir: Path,
    stem: str,
    roots: list[Path],
    labels: list[str],
    *,
    horizon: int | None = None,
    allow_mixed_protocols: bool = False,
) -> None:
    results, summaries = analyze_batch_paths(
        roots,
        labels=labels,
        horizon=horizon,
        allow_mixed_protocols=allow_mixed_protocols,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_batch_analysis(
        results,
        summaries,
        output_dir / f"{stem}.png",
        output_dir / f"{stem}_summary.csv",
    )
    write_case_details(results, output_dir / f"{stem}_cases.csv")


def section_31(output: Path) -> None:
    target = output / "3_1_case_coverage"
    target.mkdir(parents=True, exist_ok=True)
    source_root = ROOT / "cases" / "protocol" / "perturbed_imc_delay_stratified" / "sources"
    frames: dict[str, pd.DataFrame] = {}
    for system, filename in (
        ("FOPDT", "evaluation_first_order.yaml"),
        ("SOPDT", "evaluation_second_order.yaml"),
    ):
        data = yaml.safe_load((source_root / filename).read_text(encoding="utf-8")) or {}
        rows: list[dict[str, Any]] = []
        for case in data.get("cases", []):
            plant = case["plant"]
            row = {
                "system": system,
                "group": int(case["group"]),
                "K": float(plant["k"]),
                "L": float(case["time_delay"]),
                "rho": float(case["rho"]),
                "fault_type": str(case["fault_type"]),
                "severity": str(case["severity"]),
            }
            if system == "FOPDT":
                row["T"] = float(plant["t"])
            else:
                row["tau1"] = float(plant["tau1"])
                row["tau2"] = float(plant["tau2"])
            rows.append(row)
        frame = pd.DataFrame(rows)
        if len(frame) != 100:
            raise ValueError(f"Expected 100 {system} evaluation cases, found {len(frame)}.")
        frames[system] = frame

    _write_case_coverage_tables(target, frames)
    _plot_case_coverage(target / "case_coverage.png", frames)


def _write_case_coverage_tables(output_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    parameter_specs = {
        "FOPDT": (("K", "K", "-"), ("T", "T", "s"), ("L", "L", "s"), ("rho", "rho", "-")),
        "SOPDT": (
            ("K", "K", "-"),
            ("tau1", "tau1", "s"),
            ("tau2", "tau2", "s"),
            ("L", "L", "s"),
            ("rho", "rho", "-"),
        ),
    }
    summary_rows: list[dict[str, Any]] = []
    for system, specs in parameter_specs.items():
        frame = frames[system]
        for column, symbol, unit in specs:
            values = frame[column]
            summary_rows.append(
                {
                    "system": system,
                    "parameter": symbol,
                    "unit": unit,
                    "minimum": float(values.min()),
                    "median": float(values.median()),
                    "maximum": float(values.max()),
                }
            )
    pd.DataFrame(summary_rows).to_csv(output_dir / "process_parameter_summary.csv", index=False)

    rho_edges = (0.0, 0.1, 0.3, 0.5, np.inf)
    rho_labels = ("<0.1", "0.1-0.3", "0.3-0.5", ">=0.5")
    rho_rows: list[dict[str, Any]] = []
    for system, frame in frames.items():
        counts, _ = np.histogram(frame["rho"], bins=rho_edges)
        rho_rows.extend(
            {"system": system, "rho_range": label, "cases": int(count)}
            for label, count in zip(rho_labels, counts, strict=True)
        )
    pd.DataFrame(rho_rows).to_csv(output_dir / "relative_delay_distribution.csv", index=False)

    combined = pd.concat(frames.values(), ignore_index=True)
    fault_summary = (
        combined.groupby(["fault_type", "severity"], sort=False)
        .size()
        .rename("cases")
        .reset_index()
    )
    fault_summary.to_csv(output_dir / "pid_fault_coverage.csv", index=False)


def _plot_case_coverage(output_path: Path, frames: dict[str, pd.DataFrame]) -> None:
    fopdt = frames["FOPDT"]
    sopdt = frames["SOPDT"]
    rho_max = max(float(fopdt["rho"].max()), float(sopdt["rho"].max()))
    k_min, k_median, k_max = (
        float(sopdt["K"].min()),
        float(sopdt["K"].median()),
        float(sopdt["K"].max()),
    )

    def marker_size(value: float | pd.Series) -> float | pd.Series:
        return 24.0 + 76.0 * (value - k_min) / max(k_max - k_min, 1e-12)

    with plt.rc_context(RC):
        figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.5))
        fopdt_scatter = axes[0, 0].scatter(
            fopdt["T"],
            fopdt["K"],
            c=fopdt["rho"],
            cmap="viridis",
            vmin=0,
            vmax=rho_max,
            s=36,
            edgecolors="black",
            linewidths=0.25,
        )
        axes[0, 0].set_xlabel(r"Time Constant, $T$ (s)")
        axes[0, 0].set_ylabel(r"Process Gain, $K$")
        fopdt_colorbar = figure.colorbar(fopdt_scatter, ax=axes[0, 0], pad=0.02)
        fopdt_colorbar.set_label(r"Relative Delay, $\rho$")

        sopdt_scatter = axes[0, 1].scatter(
            sopdt["tau1"],
            sopdt["tau2"],
            c=sopdt["rho"],
            cmap="viridis",
            vmin=0,
            vmax=rho_max,
            s=marker_size(sopdt["K"]),
            edgecolors="black",
            linewidths=0.25,
        )
        axes[0, 1].set_xlabel(r"Time Constant, $\tau_1$ (s)")
        axes[0, 1].set_ylabel(r"Time Constant, $\tau_2$ (s)")
        sopdt_colorbar = figure.colorbar(sopdt_scatter, ax=axes[0, 1], pad=0.02)
        sopdt_colorbar.set_label(r"Relative Delay, $\rho$")
        for value in (k_min, k_median, k_max):
            axes[0, 1].scatter(
                [], [], s=marker_size(value), color="#777777", alpha=0.75, label=f"{value:.2f}"
            )
        axes[0, 1].legend(title=r"$K$", frameon=False, loc="best")

        rho_edges = (0.0, 0.1, 0.3, 0.5, np.inf)
        rho_labels = ("<0.1", "0.1-0.3", "0.3-0.5", ">=0.5")
        positions = np.arange(len(rho_labels))
        width = 0.36
        for offset, (system, color) in zip(
            (-width / 2, width / 2),
            (("FOPDT", "#0072B2"), ("SOPDT", "#E69F00")),
            strict=True,
        ):
            counts, _ = np.histogram(frames[system]["rho"], bins=rho_edges)
            axes[1, 0].bar(positions + offset, counts, width=width, color=color, label=system)
        axes[1, 0].set_xticks(positions, rho_labels)
        axes[1, 0].set_xlabel(r"Relative Delay, $\rho$")
        axes[1, 0].set_ylabel("Cases")
        axes[1, 0].legend(frameon=False)

        combined = pd.concat(frames.values(), ignore_index=True)
        fault_order = list(dict.fromkeys(combined["fault_type"]))
        severity_order = ("mild", "moderate", "severe")
        heatmap = np.array(
            [
                [
                    int(
                        (
                            (combined["fault_type"] == fault) & (combined["severity"] == severity)
                        ).sum()
                    )
                    for severity in severity_order
                ]
                for fault in fault_order
            ]
        )
        image = axes[1, 1].imshow(heatmap, cmap="Blues", aspect="auto", vmin=0)
        axes[1, 1].set_xticks(
            np.arange(len(severity_order)), [item.title() for item in severity_order]
        )
        axes[1, 1].set_yticks(
            np.arange(len(fault_order)),
            [fault.replace("_", " ").title() for fault in fault_order],
            fontsize=7,
        )
        axes[1, 1].set_xlabel("Severity")
        axes[1, 1].set_ylabel("PID Fault Type")
        threshold = float(heatmap.max()) / 2.0
        for row_index in range(heatmap.shape[0]):
            for column_index in range(heatmap.shape[1]):
                value = int(heatmap[row_index, column_index])
                axes[1, 1].text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "black",
                )
        figure.colorbar(image, ax=axes[1, 1], pad=0.02, label="Cases")

        for axis in axes.flat:
            axis.grid(False)
        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def section_411(output: Path) -> None:
    target = output / "4_1_1_full_prompt"
    methods = [
        ("imc", "full"),
        ("deepseek", "full"),
        ("qwen", "full"),
        ("base", "full"),
    ]
    for stem, system, group in TYPICAL_CASES:
        response_figure(
            target / "typical",
            stem,
            system,
            group,
            methods,
            x_break=(500.0, 3800.0) if stem == "sopdt_group052_medium_delay" else None,
        )
    for system in ("first_order", "second_order"):
        batch_figure(
            target / "statistics",
            f"{system}_deepseek_vs_qwen_vs_base",
            [batch_root(system, method) for method in ("deepseek", "qwen", "base")],
            [LABELS[method] for method in ("deepseek", "qwen", "base")],
        )


def section_412(output: Path) -> None:
    target = output / "4_1_2_prompt_ablation"
    variants = ("full", "kpi3", "numeric8")
    labels = ["Full", "KPI-3", "Numeric-8"]
    for method in ("deepseek", "qwen"):
        destination = target / ("main_deepseek" if method == "deepseek" else "appendix_qwen")
        for system in ("first_order", "second_order"):
            batch_figure(
                destination,
                f"{system}_{method}_prompt_ablation",
                [batch_root(system, method, variant) for variant in variants],
                labels,
                allow_mixed_protocols=True,
            )
        ablation_iae_figure(destination, method, variants, labels)


def ablation_iae_figure(
    output_dir: Path,
    method: str,
    variants: tuple[str, ...],
    labels: list[str],
) -> None:
    rows: list[dict[str, Any]] = []
    for system in ("first_order", "second_order"):
        for variant, label in zip(variants, labels, strict=True):
            for result_dir in sorted(
                path
                for path in batch_root(system, method, variant).iterdir()
                if path.is_dir() and (path / "value_curve.txt").is_file()
            ):
                initial_iae = read_initial_iae(result_dir)
                final_iae = read_final_iae(result_dir)
                rows.append(
                    {
                        "model": LABELS[method],
                        "system": system,
                        "prompt": label,
                        "case_id": result_dir.name,
                        "initial_iae": initial_iae,
                        "final_iae": final_iae,
                        "iae_improvement_pct": 100.0
                        * (initial_iae - final_iae)
                        / max(initial_iae, 1e-12),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"{method}_prompt_ablation_iae_cases.csv", index=False)
    summary = (
        frame.groupby(["system", "prompt"], sort=False)["iae_improvement_pct"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(output_dir / f"{method}_prompt_ablation_iae_summary.csv", index=False)

    with plt.rc_context(RC):
        figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), sharey=True)
        for axis, system in zip(axes, ("first_order", "second_order"), strict=True):
            values = [
                frame.loc[
                    (frame["system"] == system) & (frame["prompt"] == label),
                    "iae_improvement_pct",
                ].to_numpy()
                for label in labels
            ]
            boxes = axis.boxplot(
                values,
                tick_labels=labels,
                patch_artist=True,
                widths=0.58,
                showmeans=True,
                showfliers=False,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 4,
                },
                medianprops={"color": "#222222", "linewidth": 1.2},
            )
            for patch, color in zip(boxes["boxes"], ("#0072B2", "#E69F00", "#009E73"), strict=True):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
            axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
            axis.set_xlabel("FOPDT" if system == "first_order" else "SOPDT")
            axis.grid(False)
        axes[0].set_ylabel("IAE Improvement (%)")
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{method}_prompt_ablation_iae.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def section_413(output: Path) -> None:
    target = output / "4_1_3_control_styles"
    styles = [("conservative", "Conservative"), ("full", "Balanced"), ("aggressive", "Aggressive")]
    for method in ("deepseek", "qwen"):
        destination = target / ("main_deepseek" if method == "deepseek" else "appendix_qwen")
        for stem, system, group in TYPICAL_CASES:
            roots = [batch_root(system, method, variant) for variant, _ in styles]
            paths = resolve_batch_group_paths(roots, group)
            destination.mkdir(parents=True, exist_ok=True)
            compare_result_paths(
                paths,
                labels=[label for _, label in styles],
                output_path=destination / f"{stem}_{method}_styles.png",
                origin_label="Initial",
                title=None,
                metrics_csv=destination / f"{stem}_{method}_styles_metrics.csv",
                x_break=(500.0, 3800.0) if stem == "sopdt_group052_medium_delay" else None,
            )


def section_42(output: Path, sft_dir: Path | None, grpo_dir: Path | None) -> None:
    target = output / "4_2_small_model_training"
    methods = ("deepseek", "qwen", "base", "sft", "grpo")
    for system in ("first_order", "second_order"):
        batch_figure(
            target / "one_step_comparison",
            f"{system}_one_step_model_comparison",
            [batch_root(system, method) for method in methods],
            [LABELS[method] for method in methods],
            horizon=1,
        )
    pass_at_one_figure(target / "pass_at_one", methods)
    if sft_dir is None or grpo_dir is None:
        print(
            "Skipped SFT/GRPO training curves because --sft-dir and --grpo-dir were not both supplied."
        )
        return
    sft_training_curve(target / "training_curves", sft_dir)
    grpo_training_curve(target / "training_curves", grpo_dir)


def pass_at_one_figure(output_dir: Path, methods: tuple[str, ...]) -> None:
    display_labels = {
        "deepseek": "DeepSeek\nV4",
        "qwen": "Qwen\n3.7",
        "base": "Base\n0.6B",
        "sft": "SFT\n0.6B",
        "grpo": "GRPO\n0.6B",
    }
    colors = ("#0072B2", "#E69F00", "#999999", "#009E73", "#CC79A7")
    rows: list[dict[str, Any]] = []
    counts_by_system: dict[str, list[int]] = {}
    for system in ("first_order", "second_order"):
        _, summaries = analyze_batch_paths(
            [batch_root(system, method) for method in methods],
            labels=[LABELS[method] for method in methods],
            horizon=1,
        )
        counts = [summary.converged for summary in summaries]
        counts_by_system[system] = counts
        for method, summary in zip(methods, summaries, strict=True):
            rows.append(
                {
                    "system": "FOPDT" if system == "first_order" else "SOPDT",
                    "model": LABELS[method],
                    "pass_at_1": summary.converged,
                    "total_cases": summary.total,
                    "pass_at_1_rate_pct": round(100.0 * summary.convergence_rate, 2),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "pass_at_one_model_comparison.csv", index=False)
    positions = np.arange(len(methods))
    with plt.rc_context(RC):
        figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.35), sharey=True)
        for axis, system in zip(axes, ("first_order", "second_order"), strict=True):
            counts = counts_by_system[system]
            bars = axis.bar(positions, counts, color=colors, width=0.68)
            axis.set_xticks(positions, [display_labels[method] for method in methods])
            axis.set_xlabel("Model")
            axis.set_ylim(0, 105)
            axis.grid(False)
            axis.bar_label(bars, labels=[str(count) for count in counts], padding=2, fontsize=8)
            for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
                tick.set_fontweight("bold")
                tick.set_fontfamily("Arial")
        axes[0].set_ylabel("Pass@1 Cases")
        figure.tight_layout()
        figure.savefig(
            output_dir / "pass_at_one_model_comparison.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)

    combined_rows: list[dict[str, Any]] = []
    combined_rates: list[float] = []
    for method_index, method in enumerate(methods):
        passed = sum(counts_by_system[system][method_index] for system in counts_by_system)
        total = sum(row["total_cases"] for row in rows if row["model"] == LABELS[method])
        rate = 100.0 * passed / total
        combined_rates.append(rate)
        combined_rows.append(
            {
                "model": LABELS[method],
                "pass_at_1": passed,
                "total_cases": total,
                "pass_at_1_rate_pct": round(rate, 2),
            }
        )
    pd.DataFrame(combined_rows).to_csv(output_dir / "pass_at_one_combined_rate.csv", index=False)

    with plt.rc_context(RC):
        figure, axis = plt.subplots(figsize=(5.2, 3.5))
        bars = axis.bar(positions, combined_rates, color=colors, width=0.68)
        axis.set_xticks(positions, [display_labels[method] for method in methods])
        axis.set_xlabel("Model")
        axis.set_ylabel("Pass@1 Rate (%)")
        axis.set_ylim(0, 105)
        axis.grid(False)
        axis.bar_label(
            bars,
            labels=[f"{rate:.1f}%" for rate in combined_rates],
            padding=2,
            fontsize=8,
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / "pass_at_one_combined_rate.png",
            dpi=300,
            facecolor="white",
        )
        plt.close(figure)


def sft_training_curve(output_dir: Path, source: Path) -> None:
    state_path = source if source.is_file() else source / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"SFT trainer_state.json does not exist: {state_path}")
    history = json.loads(state_path.read_text(encoding="utf-8")).get("log_history") or []
    train = pd.DataFrame(
        {"step": row["step"], "loss": row["loss"]}
        for row in history
        if "step" in row and "loss" in row and "eval_loss" not in row
    ).drop_duplicates("step", keep="last")
    validation = pd.DataFrame(
        {"step": row["step"], "eval_loss": row["eval_loss"]}
        for row in history
        if "step" in row and "eval_loss" in row
    ).drop_duplicates("step", keep="last")
    if train.empty or validation.empty:
        raise ValueError(f"Incomplete SFT loss history: {state_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "sft_train_loss.csv", index=False)
    validation.to_csv(output_dir / "sft_validation_loss.csv", index=False)
    _plot_sft_training_curve(output_dir, train, validation)


def _plot_sft_training_curve(
    output_dir: Path, train: pd.DataFrame, validation: pd.DataFrame
) -> None:
    with plt.rc_context(RC):
        figure, axis = plt.subplots(figsize=(5.2, 3.5))
        axis.plot(train["step"], train["loss"], color="#0072B2", linewidth=1.0, label="Train")
        axis.plot(
            validation["step"],
            validation["eval_loss"],
            color="#D55E00",
            linewidth=1.2,
            marker="o",
            markersize=3,
            label="Validation",
        )
        axis.set_xlabel("Optimizer Step")
        axis.set_ylabel("Cross-Entropy Loss")
        axis.set_xlim(left=0)
        axis.grid(False)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / "sft_training_curve.png", dpi=300, facecolor="white")
        plt.close(figure)


def grpo_training_curve(output_dir: Path, source: Path) -> None:
    if source.is_file():
        raise ValueError("GRPO input must be the output directory containing all GRPO logs.")
    train_path = source / "trainer_log.jsonl"
    validation_path = source / "validation_log.jsonl"
    manifest_path = source / "training_manifest.json"
    for path in (train_path, validation_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required GRPO artifact does not exist: {path}")
    train = (
        pd.DataFrame(read_jsonl(train_path))
        .sort_values("step")
        .drop_duplicates("step", keep="last")
    )
    validation = (
        pd.DataFrame(read_jsonl(validation_path))
        .sort_values("step")
        .drop_duplicates("step", keep="last")
    )
    selected_step = int(
        json.loads(manifest_path.read_text(encoding="utf-8")).get("best_validation_step", 0)
    )
    if train.empty or validation.empty:
        raise ValueError(f"GRPO logs are empty under: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "grpo_training_log.csv", index=False)
    validation.to_csv(output_dir / "grpo_validation_log.csv", index=False)
    _plot_grpo_training_curves(output_dir, train, validation, selected_step)


def _plot_grpo_training_curves(
    output_dir: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selected_step: int,
) -> None:
    smooth_reward = train["reward_mean"].rolling(25, min_periods=1, center=True).mean()

    with plt.rc_context(RC):
        reward_figure, reward_axis = plt.subplots(figsize=(5.2, 3.5))
        reward_axis.plot(
            train["step"], train["reward_mean"], color="#0072B2", alpha=0.25, linewidth=0.7
        )
        reward_axis.plot(
            train["step"], smooth_reward, color="#0072B2", linewidth=1.4, label="25-step mean"
        )
        reward_axis.set_xlabel("GRPO Step")
        reward_axis.set_ylabel("Mean Reward")
        reward_axis.legend(frameon=False)
        _format_training_axis(reward_axis, selected_step)
        reward_figure.tight_layout()
        reward_figure.savefig(output_dir / "grpo_reward_curve.png", dpi=300, facecolor="white")
        plt.close(reward_figure)

        kl_figure, kl_axis = plt.subplots(figsize=(5.2, 3.5))
        kl_axis.plot(
            train["step"],
            train["approx_kl"],
            color="#D55E00",
            alpha=0.35,
            linewidth=0.7,
            label="Approx. KL",
        )
        if "kl_ema" in train:
            kl_axis.plot(
                train["step"], train["kl_ema"], color="#D55E00", linewidth=1.3, label="KL EMA"
            )
        target_kl = float(train["target_kl"].dropna().iloc[-1]) if "target_kl" in train else 0.0
        kl_axis.axhline(target_kl, color="#333333", linestyle="--", linewidth=0.8, label="Target")
        kl_axis.set_xlabel("GRPO Step")
        kl_axis.set_ylabel("KL Divergence")
        kl_axis.legend(frameon=False)
        _format_training_axis(kl_axis, selected_step)
        kl_figure.tight_layout()
        kl_figure.savefig(output_dir / "grpo_kl_curve.png", dpi=300, facecolor="white")
        plt.close(kl_figure)

        validation_figure, pass_axis = plt.subplots(figsize=(5.2, 3.5))
        iae_axis = pass_axis.twinx()
        pass_axis.plot(
            validation["step"],
            100.0 * validation["success_rate"],
            color="#009E73",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="Pass@1",
        )
        iae_axis.plot(
            validation["step"],
            validation["iae_improvement_mean"],
            color="#CC79A7",
            marker="s",
            markersize=3,
            linewidth=1.2,
            label="IAE reward",
        )
        pass_axis.set_xlabel("Validation Step")
        pass_axis.set_ylabel("Pass@1 (%)", color="#009E73")
        iae_axis.set_ylabel("Mean IAE Reward", color="#CC79A7")
        pass_axis.tick_params(axis="y", colors="#009E73")
        iae_axis.tick_params(axis="y", colors="#CC79A7")
        if selected_step > 0:
            pass_axis.axvline(selected_step, color="#555555", linestyle=":", linewidth=0.9)
        pass_axis.set_xlim(left=0)
        pass_axis.grid(False)
        validation_figure.tight_layout()
        validation_figure.savefig(
            output_dir / "grpo_validation_curve.png", dpi=300, facecolor="white"
        )
        plt.close(validation_figure)


def _format_training_axis(axis: plt.Axes, selected_step: int) -> None:
    if selected_step > 0:
        axis.axvline(selected_step, color="#555555", linestyle=":", linewidth=0.9)
    axis.set_xlim(left=0)
    axis.grid(False)


def section_43(output: Path) -> None:
    target = output / "4_3_grpo_analysis"
    methods = [
        ("imc", "full"),
        ("deepseek", "full"),
        ("qwen", "full"),
        ("sft", "full"),
        ("grpo", "full"),
    ]
    for stem, system, group in TYPICAL_CASES:
        response_figure(
            target / "typical",
            stem,
            system,
            group,
            methods,
            x_break=(500.0, 3800.0) if stem == "sopdt_group052_medium_delay" else None,
        )
    for stem, system, group in (
        ("fopdt_group018_initial_sft_grpo", "first_order", 18),
        ("sopdt_group014_initial_sft_grpo", "second_order", 14),
    ):
        stability_figure(target / "stability", stem, system, group)


def stability_figure(output_dir: Path, stem: str, system: str, group: int) -> None:
    source_path = (
        ROOT
        / "cases"
        / "protocol"
        / "perturbed_imc_delay_stratified"
        / "sources"
        / f"evaluation_{system}.yaml"
    )
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    case = next(row for row in source["cases"] if int(row["group"]) == group)
    data = case["plant"]
    plant = FrequencyPlant(
        system=system,
        k=float(data["k"]),
        t=float(data["t"]) if system == "first_order" else None,
        tau1=float(data["tau1"]) if system == "second_order" else None,
        tau2=float(data["tau2"]) if system == "second_order" else None,
        time_delay=float(case["time_delay"]),
    )
    initial_data = case["initial_pid"]
    initial_pid = PIDParams(
        kp=float(initial_data["kp"]),
        ki=float(initial_data["ki"]),
        kd=float(initial_data["kd"]),
    )
    sft_path, grpo_path = resolve_batch_group_paths(
        [batch_root(system, "sft"), batch_root(system, "grpo")], group
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_three_controller_stability(
        plant,
        initial_pid,
        read_final_pid(sft_path),
        read_final_pid(grpo_path),
        output_dir / f"{stem}.png",
        output_dir / f"{stem}.json",
    )


def _plot_three_controller_stability(
    plant: FrequencyPlant,
    initial_pid: PIDParams,
    sft_pid: PIDParams,
    grpo_pid: PIDParams,
    output_path: Path,
    summary_path: Path,
) -> None:
    controllers = [
        ("Initial", initial_pid, "#D55E00"),
        (LABELS["sft"], sft_pid, "#009E73"),
        (LABELS["grpo"], grpo_pid, "#CC79A7"),
    ]
    omega = _adaptive_frequency_grid(plant, 4096)
    responses = [_frequency_response(plant, pid, omega) for _, pid, _ in controllers]
    summaries = [
        _summarize_frequency_response(plant, label, pid, omega, response)
        for (label, pid, _), response in zip(controllers, responses, strict=True)
    ]
    crossover_frequencies = [
        summary.critical_crossover_frequency
        for summary in summaries
        if summary.critical_crossover_frequency is not None
    ]
    if crossover_frequencies:
        bode_min = max(float(omega[0]), min(crossover_frequencies) / 20.0)
        bode_max = min(float(omega[-1]), max(crossover_frequencies) * 20.0)
    else:
        bode_min, bode_max = float(omega[0]), float(omega[-1])
    bode_mask = (omega >= bode_min) & (omega <= bode_max)

    with plt.rc_context(RC):
        figure = plt.figure(figsize=(12, 7.2))
        grid = figure.add_gridspec(2, 2, width_ratios=(1.15, 1.0), hspace=0.08, wspace=0.30)
        magnitude_axis = figure.add_subplot(grid[0, 0])
        phase_axis = figure.add_subplot(grid[1, 0], sharex=magnitude_axis)
        nyquist_axis = figure.add_subplot(grid[:, 1])

        for (label, _, color), response, summary in zip(
            controllers, responses, summaries, strict=True
        ):
            magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))
            phase_deg = np.degrees(np.unwrap(np.angle(response)))
            magnitude_axis.semilogx(
                omega[bode_mask],
                magnitude_db[bode_mask],
                color=color,
                linewidth=1.6,
                label=label,
            )
            phase_axis.semilogx(
                omega[bode_mask],
                phase_deg[bode_mask],
                color=color,
                linewidth=1.6,
                label=label,
            )
            nyquist_axis.plot(
                response.real, response.imag, color=color, linewidth=1.6, label=f"{label} (+omega)"
            )
            nyquist_axis.plot(
                response.real,
                -response.imag,
                color=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
            )
            if summary.critical_crossover_frequency is not None:
                for axis in (magnitude_axis, phase_axis):
                    axis.axvline(
                        summary.critical_crossover_frequency,
                        color=color,
                        linewidth=0.8,
                        linestyle=":",
                        alpha=0.9,
                    )

        magnitude_axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        phase_axis.axhline(-180.0, color="#555555", linewidth=0.8, linestyle="--")
        magnitude_axis.set_ylabel("Magnitude (dB)")
        phase_axis.set_ylabel("Phase (deg)")
        phase_axis.set_xlabel("Frequency (rad/s)")
        magnitude_axis.tick_params(labelbottom=False)
        for axis in (magnitude_axis, phase_axis):
            axis.grid(True, which="both", alpha=0.25)
            axis.margins(x=0)
            axis.set_xlim(bode_min, bode_max)

        nyquist_axis.plot(
            -1.0, 0.0, "o", color="#222222", markersize=5, label="Critical point (-1, 0)"
        )
        nyquist_axis.axhline(0.0, color="#888888", linewidth=0.7)
        nyquist_axis.axvline(0.0, color="#888888", linewidth=0.7)
        nyquist_axis.set_xlabel("Real")
        nyquist_axis.set_ylabel("Imaginary")
        nyquist_axis.grid(True, alpha=0.25)
        nyquist_axis.set_xlim(-2.5, 1.5)
        nyquist_axis.set_ylim(-2.0, 2.0)
        nyquist_axis.set_aspect("equal", adjustable="box")

        margin_labels = []
        for summary in summaries:
            phase_margin = (
                "N/A" if summary.phase_margin_deg is None else f"{summary.phase_margin_deg:.1f} deg"
            )
            margin_labels.append(f"{summary.label}: PM={phase_margin}")
        magnitude_axis.legend(frameon=False, loc="best", title="\n".join(margin_labels))
        nyquist_axis.legend(frameon=False, loc="best")
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)

    summary_path.write_text(
        json.dumps(
            {
                "plant": dict(vars(plant)),
                "delay_evaluation": "exact_exp_minus_j_omega_theta",
                "frequency_range_rad_per_s": [float(omega[0]), float(omega[-1])],
                "frequency_points": int(len(omega)),
                "controllers": [summary.as_dict() for summary in summaries],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def read_initial_iae(result_dir: Path) -> float:
    path = result_dir / "parameter_PID_IAE.txt"
    match = re.search(r"IAE\s*=\s*([-+0-9.eE]+)", path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Cannot parse initial IAE from: {path}")
    return float(match.group(1))


def read_final_iae(result_dir: Path) -> float:
    status = yaml.safe_load((result_dir / "run_status.yaml").read_text(encoding="utf-8")) or {}
    if status.get("final_iae") is not None:
        return float(status["final_iae"])
    curves = sorted(result_dir.glob("value_curve_iteration_*.txt"), key=iteration_number)
    data = np.loadtxt(curves[-1] if curves else result_dir / "value_curve.txt", skiprows=2)
    return float(np.trapz(np.abs(data[:, 1] - data[:, 2]), data[:, 0]))


def read_final_pid(result_dir: Path) -> PIDParams:
    status = yaml.safe_load((result_dir / "run_status.yaml").read_text(encoding="utf-8")) or {}
    pid = status["final_pid"]
    return PIDParams(kp=float(pid["kp"]), ki=float(pid["ki"]), kd=float(pid["kd"]))


def iteration_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures from completed LLMPIDTuner artifacts."
    )
    parser.add_argument("section", choices=("3.1", "4.1.1", "4.1.2", "4.1.3", "4.2", "4.3", "all"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sft-dir")
    parser.add_argument("--grpo-dir")
    args = parser.parse_args()
    output = resolve_path(args.output_root)
    assert output is not None
    selected = (
        ("3.1", "4.1.1", "4.1.2", "4.1.3", "4.2", "4.3")
        if args.section == "all"
        else (args.section,)
    )
    actions = {
        "3.1": lambda: section_31(output),
        "4.1.1": lambda: section_411(output),
        "4.1.2": lambda: section_412(output),
        "4.1.3": lambda: section_413(output),
        "4.2": lambda: section_42(output, resolve_path(args.sft_dir), resolve_path(args.grpo_dir)),
        "4.3": lambda: section_43(output),
    }
    for section in selected:
        actions[section]()
        print(f"Generated manuscript section {section} under {output}")


if __name__ == "__main__":
    main()
