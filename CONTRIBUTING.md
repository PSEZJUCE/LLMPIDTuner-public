# Contributing

Contributions should preserve numerical compatibility unless a protocol change is explicitly proposed.

1. Create a focused branch and keep unrelated refactors separate.
2. Use `uv sync --extra dev`.
3. Run `uv run ruff check src tests scripts/paper`.
4. Run `uv run pytest -q`.
5. Run `uv run llmpidtuner build-protocol-assets --check`.
6. Update documentation when commands, cases, configuration, metrics, or protocol behavior changes.

Any change to simulation, dead-time handling, PID implementation, IAE, overshoot, SSE, convergence, prompt rendering, or reward computation must receive a new protocol identifier and must not silently overwrite frozen assets.

Do not include generated `runs/`, training `outputs/`, secrets, private server configuration, or third-party model weights in a pull request.