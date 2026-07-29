# Paper Artifacts

This directory contains the compact manuscript-facing artifact snapshot:

- `figures/`: generated PNG figures and companion CSV/JSON tables;
- `training_logs/`: the SFT trainer state and GRPO logs used to draw training curves;
- `experiment_manifest.yaml`: release provenance and external-deposit identifiers;
- `SHA256SUMS`: checksums for every file in this directory except the checksum file itself.

Machine-specific paths in companion CSV files have been converted to repository-relative `runs/` paths. The CSV files document source-run identities but do not replace the full raw runs.

The complete case-level run archive is intentionally distributed separately because it contains more than 100,000 files and is unsuitable for normal Git history. It is available with the generated SFT dataset and checksums in the [Zenodo reproducibility dataset](https://doi.org/10.5281/zenodo.21669697).
The figures, companion tables, and training-curve inputs in this directory are licensed under CC BY-NC 4.0; see `../LICENSE-DATA`.
