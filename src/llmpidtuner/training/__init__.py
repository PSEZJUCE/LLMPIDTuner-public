"""Training utilities for SFT and GRPO-style small-model PID tuning.

The package is intentionally import-light. Heavy dependencies such as torch,
transformers, accelerate, and peft are imported only inside training entry
points so the default LLMPIDTuner workflow remains usable on a local machine.
"""
