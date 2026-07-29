from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from llmpidtuner.batch_analysis import analyze_batch_paths, write_batch_analysis, write_case_details
from llmpidtuner.config import load_case_config
from llmpidtuner.demonstrations import (
    demonstration_protocol_id,
    generate_demonstration_from_spec,
)
from llmpidtuner.result_comparison import (
    compare_result_paths,
    resolve_batch_group_paths,
)
from llmpidtuner.frequency_analysis import compare_case_frequency_response
from llmpidtuner.models import PIDParams
from llmpidtuner.runner import (
    run_case,
    write_batch_plants_from_config,
    write_batch_results_from_config,
)


def _parse_group_spec(value: str) -> set[int]:
    groups: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 1 or end < start:
                raise ValueError(f"Invalid group range: {token}")
            groups.update(range(start, end + 1))
        else:
            group = int(token)
            if group < 1:
                raise ValueError(f"Invalid group number: {token}")
            groups.add(group)
    if not groups:
        raise ValueError("--groups must contain at least one positive group number.")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLMPIDTuner cases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a YAML case file.")
    run_parser.add_argument("case", help="Path to a case YAML file.")
    run_parser.add_argument(
        "--mode",
        choices=["dry_run", "llm", "imc"],
        help="Override the case mode. dry_run and imc do not call any LLM API.",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="For batch LLM runs, skip completed group directories.",
    )
    run_parser.add_argument(
        "--groups",
        help="Run only selected batch groups, for example 1-20 or 1-10,31-40.",
    )
    run_parser.add_argument(
        "--no-batch-excel",
        action="store_true",
        help="Do not write experiment_results.xlsx; required for parallel batch workers.",
    )

    parallel_api_parser = subparsers.add_parser(
        "run-api-parallel",
        help="Run one or more batch API cases with independent worker processes.",
    )
    parallel_api_parser.add_argument(
        "cases",
        nargs="+",
        help="Batch case YAML files. Cases run sequentially; groups run concurrently.",
    )
    parallel_api_parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Maximum concurrent API workers per case (default: 10).",
    )
    parallel_api_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip group directories that already contain a terminal run result.",
    )
    parallel_api_parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=0.5,
        help="Delay successive worker starts to avoid a request burst (default: 0.5).",
    )

    collect_batch_parser = subparsers.add_parser(
        "collect-batch",
        help="Rebuild experiment_results.xlsx from completed batch group result files.",
    )
    collect_batch_parser.add_argument("case", help="Path to a batch case YAML file.")

    build_demo_parser = subparsers.add_parser(
        "build-demonstrations",
        help="Build or verify versioned frozen demonstration artifacts.",
    )
    build_demo_parser.add_argument("config", help="Path to a demonstration protocol YAML file.")
    build_demo_parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate in memory and fail if committed artifacts differ.",
    )
    build_demo_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing artifacts. Cannot be combined with --check.",
    )

    protocol_parser = subparsers.add_parser(
        "build-protocol-assets",
        help="Build or verify the perturbed-IMC demonstration and benchmark sources.",
    )
    protocol_parser.add_argument(
        "--protocol-root",
        default="cases/protocol/perturbed_imc_delay_stratified",
    )
    protocol_parser.add_argument(
        "--demonstration-root",
        default="cases/demonstrations/perturbed_imc_delay_stratified",
    )
    protocol_parser.add_argument("--check", action="store_true")
    protocol_parser.add_argument("--force", action="store_true")

    export_batch_parser = subparsers.add_parser(
        "export-batch-plants",
        help="Freeze a seeded batch case into an explicit plant-list YAML file.",
    )
    export_batch_parser.add_argument("case", help="Path to a batch case YAML file.")
    export_batch_parser.add_argument("--output", required=True, help="Output plant-list YAML path.")

    analyze_parser = subparsers.add_parser(
        "analyze-batches",
        help="Summarize one or more batch result directories.",
    )
    analyze_parser.add_argument(
        "paths",
        nargs="+",
        help="Batch result directories, for example runs/first_order_batch_ds_deepseek-v4-flash.",
    )
    analyze_parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional display labels. Count must match the number of paths.",
    )
    analyze_parser.add_argument(
        "--output",
        default="runs/batch_analysis.png",
        help="Output PNG path. Default: runs/batch_analysis.png.",
    )
    analyze_parser.add_argument(
        "--summary-csv",
        default="runs/batch_analysis_summary.csv",
        help="Output summary CSV path. Default: runs/batch_analysis_summary.csv.",
    )
    analyze_parser.add_argument(
        "--case-csv",
        default="runs/batch_analysis_cases.csv",
        help="Output per-case CSV path. Default: runs/batch_analysis_cases.csv.",
    )
    analyze_parser.add_argument(
        "--success-overshoot",
        type=float,
        default=15.0,
        help="Overshoot threshold for convergence. Default: 15.0.",
    )
    analyze_parser.add_argument(
        "--success-steady-state-error",
        type=float,
        default=1.0,
        help="Steady-state error threshold for convergence. Default: 1.0.",
    )
    analyze_parser.add_argument(
        "--horizon",
        type=int,
        help="Analyze results using only iterations up to this positive limit.",
    )
    analyze_parser.add_argument(
        "--allow-mixed-protocols",
        action="store_true",
        help="Allow intentional comparison of different frozen prompt protocols.",
    )

    compare_parser = subparsers.add_parser(
        "compare-results",
        help="Overlay final response curves from single-case result directories.",
    )
    compare_parser.add_argument(
        "paths",
        nargs="+",
        help="Single-case result directories, for example runs/first_order_c1_imc.",
    )
    compare_parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional display labels. Count must match the number of paths.",
    )
    compare_parser.add_argument(
        "--group",
        type=int,
        help="Treat paths as batch roots and compare the matching group number.",
    )
    compare_parser.add_argument(
        "--output",
        default="runs/result_comparison.png",
        help="Output PNG path. Default: runs/result_comparison.png.",
    )
    compare_parser.add_argument(
        "--origin-label",
        default="Origin",
        help="Label for the initial PID curve read from the first result path. Default: Origin.",
    )
    compare_parser.add_argument("--title", help="Optional figure title.")
    compare_parser.add_argument(
        "--metrics-csv",
        help="Optional CSV path for final overshoot and steady-state error.",
    )

    frequency_parser = subparsers.add_parser(
        "compare-frequency",
        help="Compare exact-delay Bode and Nyquist plots for base and GRPO PID gains.",
    )
    frequency_parser.add_argument(
        "case",
        help="Single first_order or second_order case YAML defining the transfer function.",
    )
    frequency_parser.add_argument(
        "--base-pid",
        nargs=3,
        type=float,
        metavar=("KP", "KI", "KD"),
        required=True,
        help="Base-model PID gains: Kp Ki Kd.",
    )
    frequency_parser.add_argument(
        "--grpo-pid",
        nargs=3,
        type=float,
        metavar=("KP", "KI", "KD"),
        required=True,
        help="GRPO-model PID gains: Kp Ki Kd.",
    )
    frequency_parser.add_argument("--base-label", default="Base model")
    frequency_parser.add_argument("--grpo-label", default="GRPO model")
    frequency_parser.add_argument(
        "--output",
        default="runs/frequency_comparison.png",
        help="Output PNG path. Default: runs/frequency_comparison.png.",
    )
    frequency_parser.add_argument(
        "--summary",
        help="Optional JSON path for phase-margin and crossover summaries.",
    )
    frequency_parser.add_argument(
        "--frequency-points",
        type=int,
        default=4096,
        help="Number of adaptive logarithmic frequency samples. Default: 4096.",
    )

    frequency_parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit plot titles for manuscript-ready figure panels.",
    )

    sft_data_parser = subparsers.add_parser(
        "make-sft-data",
        help="Generate PID tuning prompt data for SFT or GRPO.",
    )
    sft_data_parser.add_argument("config", help="Path to a training data YAML file.")

    validate_sft_parser = subparsers.add_parser(
        "validate-sft-data",
        help="Tokenize every SFT row and reject sequences above max_length.",
    )
    validate_sft_parser.add_argument("config", help="Path to an SFT training YAML file.")

    train_sft_parser = subparsers.add_parser(
        "train-sft",
        help="Run supervised fine-tuning. Intended for the GPU server.",
    )
    train_sft_parser.add_argument("config", help="Path to an SFT training YAML file.")

    train_grpo_parser = subparsers.add_parser(
        "train-grpo",
        help="Run GRPO training. Intended for the GPU server.",
    )
    train_grpo_parser.add_argument("config", help="Path to a GRPO training YAML file.")

    sbatch_parser = subparsers.add_parser(
        "render-sbatch",
        help="Render an sbatch script for a server training job.",
    )
    sbatch_parser.add_argument("config", help="Path to a server job YAML file.")
    sbatch_parser.add_argument("--output", required=True, help="Output sbatch script path.")

    args = parser.parse_args()
    if args.command == "run":
        config = load_case_config(args.case)
        if args.mode:
            config = config.__class__(**{**config.__dict__, "mode": args.mode})
        if args.resume:
            config = config.__class__(**{**config.__dict__, "resume": True})
        groups = _parse_group_spec(args.groups) if args.groups else None
        run_case(config, batch_groups=groups, write_batch_excel=not args.no_batch_excel)
    elif args.command == "run-api-parallel":
        from llmpidtuner.parallel_api import run_api_cases_parallel

        run_api_cases_parallel(
            args.cases,
            workers=args.workers,
            resume=args.resume,
            stagger_seconds=args.stagger_seconds,
        )
    elif args.command == "collect-batch":
        output_path = write_batch_results_from_config(load_case_config(args.case))
        print(f"Batch workbook: {output_path}")
    elif args.command == "build-demonstrations":
        if args.check and args.force:
            raise SystemExit("--check and --force cannot be combined.")
        from llmpidtuner.demonstration_protocol import build_demonstration_protocol

        paths = build_demonstration_protocol(args.config, check=args.check, force=args.force)
        action = "Verified" if args.check else "Built"
        print(f"{action} {len(paths)} demonstration artifacts.")
    elif args.command == "build-protocol-assets":
        if args.check and args.force:
            raise SystemExit("--check and --force cannot be combined.")
        from llmpidtuner.protocol_artifacts import build_protocol_assets

        paths = build_protocol_assets(
            protocol_root=args.protocol_root,
            demonstration_root=args.demonstration_root,
            check=args.check,
            force=args.force,
        )
        print(f"{'Verified' if args.check else 'Built'} {len(paths)} protocol artifacts.")
    elif args.command == "export-batch-plants":
        config = load_case_config(args.case)
        output_path = write_batch_plants_from_config(config, args.output)
        print(f"Batch plant list written to {output_path}")
    elif args.command == "analyze-batches":
        if args.labels and len(args.labels) != len(args.paths):
            raise SystemExit("--labels count must match the number of paths.")
        results, summaries = analyze_batch_paths(
            args.paths,
            labels=args.labels,
            success_overshoot=args.success_overshoot,
            success_steady_state_error=args.success_steady_state_error,
            allow_mixed_protocols=args.allow_mixed_protocols,
            horizon=args.horizon,
        )
        figure_path, summary_csv_path = write_batch_analysis(
            results, summaries, args.output, args.summary_csv
        )
        case_csv_path = write_case_details(results, args.case_csv)
        print(f"Batch analysis figure: {figure_path}")
        if summary_csv_path is not None:
            print(f"Summary CSV: {summary_csv_path}")
        print(f"Case detail CSV: {case_csv_path}")
    elif args.command == "compare-results":
        if args.labels and len(args.labels) != len(args.paths):
            raise SystemExit("--labels count must match the number of paths.")
        comparison_paths = (
            resolve_batch_group_paths(args.paths, args.group) if args.group else args.paths
        )
        figure_path, metrics_csv_path, summaries = compare_result_paths(
            comparison_paths,
            labels=args.labels,
            output_path=args.output,
            origin_label=args.origin_label,
            title=args.title,
            metrics_csv=args.metrics_csv,
        )
        print(f"Comparison figure: {figure_path}")
        if metrics_csv_path is not None:
            print(f"Metrics CSV: {metrics_csv_path}")
        for summary in summaries:
            print(
                f"{summary.label}: source={summary.source_file}, "
                f"Overshoot={summary.overshoot:.2f}%, "
                f"SSE={summary.steady_state_error:.2f}%"
            )

    elif args.command == "compare-frequency":
        case = load_case_config(args.case)
        base_pid = PIDParams(*args.base_pid)
        grpo_pid = PIDParams(*args.grpo_pid)
        figure_path, summary_path, summaries = compare_case_frequency_response(
            case,
            base_pid,
            grpo_pid,
            args.output,
            base_label=args.base_label,
            grpo_label=args.grpo_label,
            summary_path=args.summary,
            points=args.frequency_points,
            show_titles=not args.no_title,
        )
        print(f"Frequency comparison figure: {figure_path}")
        if summary_path is not None:
            print(f"Frequency summary: {summary_path}")
        for summary in summaries:
            phase_margin = (
                "N/A" if summary.phase_margin_deg is None else f"{summary.phase_margin_deg:.2f} deg"
            )
            crossover = (
                "N/A"
                if summary.critical_crossover_frequency is None
                else f"{summary.critical_crossover_frequency:.6g} rad/s"
            )
            print(f"{summary.label}: PM={phase_margin}, critical crossover={crossover}")
    elif args.command == "make-sft-data":
        from collections import Counter

        from llmpidtuner.training.config import load_training_data_config
        from llmpidtuner.training.artifacts import runtime_metadata, sha256_file, write_json_atomic
        from llmpidtuner.training.data import (
            generate_prompt_samples,
            generate_protocol_prompt_samples_by_type,
            write_prompt_samples,
            write_sft_messages_dataset,
        )

        config = load_training_data_config(args.config)
        demonstrations: dict[str, str] = {}
        if config.demonstration and config.demonstration.get("method") != "frozen":
            raise ValueError(
                "SFT data generation requires a versioned frozen demonstration protocol."
            )
        if config.demonstration:
            for system in ("first_order", "second_order"):
                spec = {**config.demonstration, "system": system}
                text = generate_demonstration_from_spec(
                    spec,
                    initial_pid=config.initial_pid,
                    simulation=config.simulation,
                )
                if text:
                    demonstrations[system] = text
        if config.first_order_count is not None and config.second_order_count is not None:
            samples = generate_protocol_prompt_samples_by_type(
                first_order_count=config.first_order_count,
                second_order_count=config.second_order_count,
                seed=config.seed,
                simulation=config.simulation,
                demonstrations=demonstrations,
                excluded_case_paths=config.excluded_plants_paths,
                workers=config.workers,
                control_style=config.control_style,
            )
        else:
            samples = generate_prompt_samples(
                count=config.count,
                seed=config.seed,
                simulation=config.simulation,
                second_order_prob=config.second_order_prob,
                initial_pid=config.initial_pid,
                demonstrations=demonstrations,
                control_style=config.control_style,
            )
        if config.format == "prompt_samples":
            count = write_prompt_samples(samples, config.output_path)
        elif config.format == "openai_messages":
            count = write_sft_messages_dataset(
                samples,
                config.output_path,
                lambda_value=config.lambda_value,
                simulation=config.simulation,
                feedback_sample_probability=config.feedback_sample_probability,
                seed=config.seed,
                include_target_metrics=config.include_target_metrics,
            )
        else:
            raise SystemExit("format must be either 'openai_messages' or 'prompt_samples'.")
        output_path = Path(config.output_path)
        control_style_counts = Counter(sample.control_style for sample in samples)
        plant_type_counts = Counter(sample.plant.plant_type for sample in samples)
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        protocol_id = demonstration_protocol_id(config.demonstration)
        write_json_atomic(
            manifest_path,
            {
                "schema_version": 1,
                "artifact_type": "sft_dataset",
                "demonstration_protocol": protocol_id,
                "source_config": str(args.config),
                "generator_config": asdict(config),
                "dataset": {
                    "path": str(output_path),
                    "rows": count,
                    "bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                    "control_styles": dict(sorted(control_style_counts.items())),
                    "plant_types": dict(sorted(plant_type_counts.items())),
                },
                **runtime_metadata(("llmpidtuner", "numpy", "scipy", "PyYAML")),
            },
        )
        print(f"Training-data manifest: {manifest_path}")
        print(f"Wrote {count} training samples to {config.output_path}")
    elif args.command == "train-sft":
        from llmpidtuner.training.config import load_sft_train_config
        from llmpidtuner.training.sft import train_sft

        train_sft(load_sft_train_config(args.config))
    elif args.command == "validate-sft-data":
        from llmpidtuner.training.config import load_sft_train_config
        from llmpidtuner.training.sft import validate_sft_dataset

        validate_sft_dataset(load_sft_train_config(args.config))
    elif args.command == "train-grpo":
        from llmpidtuner.training.config import load_grpo_train_config
        from llmpidtuner.training.grpo import train_grpo

        train_grpo(load_grpo_train_config(args.config))
    elif args.command == "render-sbatch":
        from llmpidtuner.training.config import load_server_job_config
        from llmpidtuner.training.server import write_sbatch

        output_path = write_sbatch(load_server_job_config(args.config), args.output)
        print(f"SBATCH script written to {output_path}")
