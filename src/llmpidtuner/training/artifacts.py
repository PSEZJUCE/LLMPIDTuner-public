from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


_SOURCE_PATHS = (
    "src",
    "configs",
    "cases",
    "scripts",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: str | Path = ".") -> str:
    root_path = Path(root).resolve()
    files: list[Path] = []
    for relative in _SOURCE_PATHS:
        candidate = root_path / relative
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and "generated" not in path.parts
                and not any(part.endswith(".egg-info") for part in path.parts)
                and path.suffix != ".pyc"
            )

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root_path).as_posix()):
        relative = path.relative_to(root_path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_metadata(package_names: Iterable[str] = ()) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return {
        "python_version": platform.python_version(),
        "package_versions": versions,
        "git_commit": current_git_commit(),
        "source_tree_sha256": source_tree_sha256(),
    }


def current_git_commit() -> str | None:
    configured = os.environ.get("LLMPIDTUNER_GIT_COMMIT", "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path
