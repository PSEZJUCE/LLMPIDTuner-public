from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from llmpidtuner.experiment_protocol import (
    PROTOCOL_ID,
    ProtocolCase,
    generate_protocol_cases,
    imc_pid_for_style,
    protocol_manifest,
    simulate_protocol_case,
)
from llmpidtuner.metrics import ControlSystemAnalysis
from llmpidtuner.prompting import format_pid_response


DEFAULT_ROOT = Path("cases/protocol/perturbed_imc_delay_stratified")
DEMONSTRATION_ROOT = Path("cases/demonstrations/perturbed_imc_delay_stratified")
SEEDS = {
    "demonstration_first_order": 51001,
    "demonstration_second_order": 52001,
    "evaluation_first_order": 61001,
    "evaluation_second_order": 62001,
    "grpo_validation_first_order": 71001,
    "grpo_validation_second_order": 72001,
}
DEMONSTRATION_ARTIFACTS = (
    ("balanced", "full"),
    ("balanced", "kpi3"),
    ("balanced", "numeric8"),
    ("aggressive", "full"),
    ("conservative", "full"),
)


def build_protocol_assets(
    *,
    protocol_root: str | Path = DEFAULT_ROOT,
    demonstration_root: str | Path = DEMONSTRATION_ROOT,
    check: bool = False,
    force: bool = False,
) -> list[Path]:
    protocol_root = Path(protocol_root)
    demonstration_root = Path(demonstration_root)
    desired: dict[Path, bytes] = {}
    all_excluded: set[str] = set()

    for system in ("first_order", "second_order"):
        demo_seed = SEEDS[f"demonstration_{system}"]
        demo_cases = generate_protocol_cases(
            system,
            10,
            demo_seed,
            purpose="frozen_demonstration",
            require_all_style_targets=True,
            excluded_hashes=all_excluded,
        )
        all_excluded.update(case.case_hash for case in demo_cases)
        source_path = protocol_root / "sources" / f"demonstration_{system}.yaml"
        source_bytes = _yaml_bytes(
            {
                **protocol_manifest(
                    demo_cases,
                    seed=demo_seed,
                    purpose="frozen_demonstration",
                ),
                "cases": [_case_row(index, case) for index, case in enumerate(demo_cases, 1)],
            }
        )
        desired[source_path] = source_bytes

        for style, variant in DEMONSTRATION_ARTIFACTS:
            prompt_path = demonstration_root / style / variant / f"{system}.txt"
            manifest_path = prompt_path.with_suffix(".manifest.yaml")
            prompt = render_demonstration(demo_cases, style=style, prompt_variant=variant)
            prompt_bytes = _text_bytes(prompt)
            desired[prompt_path] = prompt_bytes
            desired[manifest_path] = _yaml_bytes(
                {
                    "schema_version": 1,
                    "artifact_type": "frozen_demonstration",
                    "demonstration_protocol": PROTOCOL_ID,
                    "system": system,
                    "control_style": style,
                    "prompt_variant": variant,
                    "source": {
                        "path": source_path.as_posix(),
                        "sha256": _sha256(source_bytes),
                        "count": len(demo_cases),
                    },
                    "prompt": {
                        "path": prompt_path.as_posix(),
                        "sha256": _sha256(prompt_bytes),
                    },
                }
            )

    evaluation_hashes: set[str] = set()
    for system in ("first_order", "second_order"):
        seed = SEEDS[f"evaluation_{system}"]
        cases = generate_protocol_cases(
            system,
            100,
            seed,
            purpose="paper_evaluation",
            excluded_hashes=all_excluded,
        )
        evaluation_hashes.update(case.case_hash for case in cases)
        path = protocol_root / "sources" / f"evaluation_{system}.yaml"
        desired[path] = _yaml_bytes(
            {
                **protocol_manifest(cases, seed=seed, purpose="paper_evaluation"),
                "cases": [_case_row(index, case) for index, case in enumerate(cases, 1)],
            }
        )

    validation_exclusions = all_excluded | evaluation_hashes
    for system in ("first_order", "second_order"):
        seed = SEEDS[f"grpo_validation_{system}"]
        cases = generate_protocol_cases(
            system,
            100,
            seed,
            purpose="grpo_validation",
            excluded_hashes=validation_exclusions,
        )
        path = protocol_root / "sources" / f"grpo_validation_{system}.yaml"
        desired[path] = _yaml_bytes(
            {
                **protocol_manifest(cases, seed=seed, purpose="grpo_validation"),
                "cases": [_case_row(index, case) for index, case in enumerate(cases, 1)],
            }
        )

    if check:
        _check(desired)
    else:
        _write(desired, force=force)
    return sorted(desired)


def render_demonstration(
    cases: list[ProtocolCase],
    *,
    style: str,
    prompt_variant: str,
) -> str:
    sections: list[str] = []
    for index, case in enumerate(cases, start=1):
        result, _ = simulate_protocol_case(
            case.plant,
            case.initial_pid,
            case.time_delay,
        )
        description = ControlSystemAnalysis.from_arrays(
            result.time,
            result.time * 0.0 + 1.0,
            result.output,
            filename=f"{case.system}_demo_{index}",
            time_delay=case.time_delay,
        ).generate_description(prompt_variant)
        target = imc_pid_for_style(case.plant, case.time_delay, style)
        sections.append(
            f"""Experiment {index}:
Initial PID parameters and IAE: Kp={case.initial_pid.kp:.6g}, Ki={case.initial_pid.ki:.6g}, Kd={case.initial_pid.kd:.6g}, IAE={case.initial_metrics.iae:.6g}
Observed process dead time: {case.time_delay:.4g} seconds.
{description.strip()}
Recommended control style: {style}.
Suggested PID parameters: {format_pid_response(target)}"""
        )
    return "\n\n".join(sections) + "\n"


def _case_row(group: int, case: ProtocolCase) -> dict[str, Any]:
    row = case.as_dict()
    row["group"] = group
    return row


def _text_bytes(text: str) -> bytes:
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _yaml_bytes(data: dict[str, Any]) -> bytes:
    return _text_bytes(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _check(desired: dict[Path, bytes]) -> None:
    failures = [
        f"{'missing' if not path.exists() else 'stale'}: {path}"
        for path, content in desired.items()
        if not path.exists() or path.read_bytes() != content
    ]
    if failures:
        raise ValueError("Protocol asset check failed:\n" + "\n".join(failures))


def _write(desired: dict[Path, bytes], *, force: bool) -> None:
    existing = [path for path in desired if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite protocol assets without --force: "
            + ", ".join(str(path) for path in existing)
        )
    for path, content in desired.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
