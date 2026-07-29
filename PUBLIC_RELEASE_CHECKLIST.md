# Public Release Checklist

Complete these items after the new GitHub repository URL is known and before changing repository visibility to public.

- [ ] Confirm all relevant authors and institutions approve both the public release and the selected licenses.
- [x] Select and add source-code and data licenses: PolyForm Noncommercial 1.0.0 and CC BY-NC 4.0.
- [ ] Add the public repository URL to `CITATION.cff` and README.
- [ ] Create a clean public Git history; do not import the private repository's `.git`.
- [ ] Create a tagged GitHub release and archive it with a software DOI.
- [ ] Deposit full raw results separately and record its DOI/checksums.
- [ ] Decide whether fine-tuned weights will receive a later model deposit.
- [ ] Replace every `PENDING` value in `paper_artifacts/experiment_manifest.yaml`.
- [ ] Run secret/path scans, tests, protocol checks, and CI.
- [ ] Verify that no `.env`, private host, account name, local absolute path, or model credential is tracked.