# Security Policy

## Secrets

Never commit `.env`, API keys, access tokens, private model URLs, SSH credentials, or internal hostnames. Use `.env.example` only to document variable names.

If a secret is accidentally committed, revoke it before rewriting history; deleting the file in a later commit is not sufficient.

## Reporting

Before the public repository URL is assigned, report security-sensitive issues privately to the corresponding authors. After publication, enable GitHub private vulnerability reporting and update this file with the repository security-advisory link.

## Operational Safety

LLMPIDTuner is research software. Generated PID gains must be independently checked in simulation and by a qualified control engineer before use. Do not connect generated controllers directly to safety-critical or production equipment.