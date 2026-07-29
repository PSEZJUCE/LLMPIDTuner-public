from __future__ import annotations

import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from llmpidtuner.config import CaseConfig, load_case_config
from llmpidtuner.runner import _batch_plants, run_case, write_batch_results_from_config


def partition_groups(groups: Sequence[int], workers: int) -> list[tuple[int, ...]]:
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    ordered = tuple(sorted(set(groups)))
    if not ordered:
        raise ValueError("At least one batch group is required.")
    chunk_size = math.ceil(len(ordered) / min(workers, len(ordered)))
    return [ordered[index : index + chunk_size] for index in range(0, len(ordered), chunk_size)]


def run_api_cases_parallel(
    case_paths: Sequence[str | Path],
    *,
    workers: int = 10,
    resume: bool = False,
    stagger_seconds: float = 0.5,
) -> list[Path]:
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if stagger_seconds < 0:
        raise ValueError("stagger_seconds must be non-negative.")

    workbooks: list[Path] = []
    for case_path in case_paths:
        path = Path(case_path)
        config = load_case_config(path)
        plant_kind = _batch_plant_kind(config)
        if not config.llm_profile:
            raise ValueError(f"Parallel API case requires llm_profile: {path}")

        groups = [group for group, _ in _batch_plants(config.batch or {}, plant_kind)]
        shards = partition_groups(groups, workers)
        output_root = Path(config.output_dir) / config.name
        log_dir = output_root / "parallel_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Parallel API evaluation: {path} with {len(shards)} workers for {len(groups)} groups"
        )
        context = multiprocessing.get_context("spawn")
        failures: list[str] = []
        with ProcessPoolExecutor(max_workers=len(shards), mp_context=context) as executor:
            futures = {
                executor.submit(
                    _run_api_shard,
                    str(path),
                    shard,
                    resume,
                    worker_index,
                    stagger_seconds,
                ): (worker_index, shard)
                for worker_index, shard in enumerate(shards)
            }
            for future in as_completed(futures):
                worker_index, shard = futures[future]
                try:
                    log_path = future.result()
                except Exception as error:
                    failures.append(
                        f"worker {worker_index + 1} groups {_format_group_span(shard)}: {error}"
                    )
                else:
                    print(
                        f"Completed worker {worker_index + 1}: groups "
                        f"{_format_group_span(shard)} ({log_path})"
                    )

        if failures:
            details = "\n".join(failures)
            raise RuntimeError(f"Parallel API evaluation failed for {path}:\n{details}")

        workbook = write_batch_results_from_config(replace(config, mode="llm"))
        workbooks.append(workbook)
        print(f"Parallel API case complete: {workbook}")
    return workbooks


def _run_api_shard(
    case_path: str,
    groups: tuple[int, ...],
    resume: bool,
    worker_index: int,
    stagger_seconds: float,
) -> Path:
    if stagger_seconds:
        time.sleep(worker_index * stagger_seconds)
    config = replace(load_case_config(case_path), mode="llm", resume=resume)
    output_root = Path(config.output_dir) / config.name
    log_dir = output_root / "parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    span = _format_group_span(groups).replace("-", "_")
    log_path = log_dir / f"worker_{worker_index + 1:02d}_groups_{span}.log"
    with log_path.open("w", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            run_case(config, batch_groups=set(groups), write_batch_excel=False)
    return log_path


def _batch_plant_kind(config: CaseConfig) -> str:
    if config.system == "first_order_batch":
        return "first_order"
    if config.system == "second_order_batch":
        return "second_order"
    raise ValueError(f"Parallel API evaluation requires a batch case, got {config.system}.")


def _format_group_span(groups: Sequence[int]) -> str:
    if len(groups) == 1:
        return str(groups[0])
    return f"{groups[0]}-{groups[-1]}"
