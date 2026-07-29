from __future__ import annotations

import os
import re
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

from llmpidtuner.models import PIDParams


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "openai-compatible"
    profile: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str = "Meta-Llama-3.1-8B-Instruct"
    temperature: float = 0.1
    top_p: float = 0.1
    enable_thinking: bool | None = None
    seed: int | None = None
    max_tokens: int | None = None
    max_retries: int = 2

    def with_overrides(self, values: dict[str, object] | None) -> "LLMSettings":
        if not values:
            return self
        allowed = {
            "temperature",
            "top_p",
            "enable_thinking",
            "seed",
            "max_retries",
            "max_tokens",
        }
        unknown = set(values) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported llm setting(s): {names}")
        max_tokens = (
            int(values["max_tokens"])
            if values.get("max_tokens") is not None
            else self.max_tokens
        )
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        return LLMSettings(
            provider=self.provider,
            profile=self.profile,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            temperature=float(values.get("temperature", self.temperature)),
            top_p=float(values.get("top_p", self.top_p)),
            enable_thinking=(
                _coerce_optional_bool(values["enable_thinking"])
                if "enable_thinking" in values
                else self.enable_thinking
            ),
            seed=(int(values["seed"]) if values.get("seed") is not None else self.seed),
            max_retries=int(values.get("max_retries", self.max_retries)),
            max_tokens=max_tokens,
        )

    @classmethod
    def from_env(cls, env_path: str = ".env", profile: str | None = None) -> "LLMSettings":
        load_dotenv(env_path)
        if profile:
            return cls._from_profile(profile)

        enable_thinking = _optional_bool(os.getenv("LLM_ENABLE_THINKING"))
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openai-compatible"),
            profile=None,
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            model=os.getenv("LLM_MODEL", "Meta-Llama-3.1-8B-Instruct"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            top_p=float(os.getenv("LLM_TOP_P", "0.1")),
            enable_thinking=enable_thinking,
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        )

    @classmethod
    def _from_profile(cls, profile: str) -> "LLMSettings":
        provider_key, model_alias = split_profile(profile)
        env_prefix = provider_env_prefix(provider_key)
        profile_prefix = normalize_env_key(profile)
        model_key_prefix = f"{env_prefix}_{normalize_env_key(model_alias)}"

        model = (
            infer_model_from_alias(model_alias)
            or os.getenv(f"{model_key_prefix}_MODEL")
            or os.getenv(f"{profile_prefix}_MODEL")
            or os.getenv(f"{env_prefix}_MODEL")
        )
        api_key = (
            os.getenv(f"{model_key_prefix}_API_KEY")
            or os.getenv(f"{profile_prefix}_API_KEY")
            or os.getenv(f"{env_prefix}_API_KEY")
        )
        if env_prefix == "VLLM" and not api_key:
            api_key = "EMPTY"

        return cls(
            provider=env_prefix.lower(),
            profile=profile,
            base_url=(
                os.getenv(f"{model_key_prefix}_BASE_URL")
                or os.getenv(f"{profile_prefix}_BASE_URL")
                or os.getenv(f"{env_prefix}_BASE_URL")
            ),
            api_key=api_key,
            model=model,
            temperature=float(
                os.getenv(f"{model_key_prefix}_TEMPERATURE")
                or os.getenv(f"{profile_prefix}_TEMPERATURE")
                or os.getenv(f"{env_prefix}_TEMPERATURE")
                or os.getenv("LLM_TEMPERATURE", "0.1")
            ),
            top_p=float(
                os.getenv(f"{model_key_prefix}_TOP_P")
                or os.getenv(f"{profile_prefix}_TOP_P")
                or os.getenv(f"{env_prefix}_TOP_P")
                or os.getenv("LLM_TOP_P", "0.1")
            ),
            enable_thinking=_optional_bool(
                os.getenv(f"{model_key_prefix}_ENABLE_THINKING")
                or os.getenv(f"{profile_prefix}_ENABLE_THINKING")
                or os.getenv(f"{env_prefix}_ENABLE_THINKING")
            ),
            max_retries=int(
                os.getenv(f"{model_key_prefix}_MAX_RETRIES")
                or os.getenv(f"{profile_prefix}_MAX_RETRIES")
                or os.getenv(f"{env_prefix}_MAX_RETRIES")
                or os.getenv("LLM_MAX_RETRIES", "2")
            ),
        )


class PIDControllerClient:
    def __init__(self, settings: LLMSettings) -> None:
        if not settings.api_key:
            detail = f" for profile {settings.profile}" if settings.profile else ""
            raise ValueError(f"API key is required for llm mode{detail}.")
        self.settings = settings
        if settings.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        ensure_no_proxy_for_base_url(settings.base_url)
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=settings.max_retries,
        )
        self.conversation_history: list[dict[str, str]] = []

    def call_pid_parameters(
        self,
        system_content: str,
        user_content: str,
        use_initial_model: bool = True,
    ) -> PIDParams:
        if use_initial_model:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            self.conversation_history = messages.copy()
        else:
            self.conversation_history.append({"role": "user", "content": user_content})
            messages = self.conversation_history

        kwargs: dict[str, object] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
        }
        if self.settings.seed is not None:
            kwargs["seed"] = self.settings.seed
        if self.settings.max_tokens is not None:
            kwargs["max_tokens"] = self.settings.max_tokens
        thinking_body = _thinking_request_body(self.settings)
        if thinking_body is not None:
            kwargs["extra_body"] = thinking_body
            kwargs["stream"] = False

        response = self.client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content or ""
        self.conversation_history.append({"role": "assistant", "content": response_text})
        print("Response from API:", response_text)
        return parse_pid_parameters(response_text)


def parse_pid_parameters(response_text: str) -> PIDParams:
    normalized_text = (
        response_text.replace("；", ";")
        .replace("：", ":")
        .replace("，", ",")
        .replace("P =", "P:")
        .replace("I =", "I:")
        .replace("D =", "D:")
    )
    normalized_text = normalized_text.replace("；", ";").replace("：", ":").replace("，", ",")
    number_pattern = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    pid_values = re.search(
        rf"K?P\s*[:=]\s*{number_pattern}\s*[,;]\s*"
        rf"K?I\s*[:=]\s*{number_pattern}\s*[,;]\s*"
        rf"K?D\s*[:=]\s*{number_pattern}",
        normalized_text,
        re.IGNORECASE,
    )
    if pid_values:
        kp, ki, kd = map(float, pid_values.groups())
        return PIDParams(kp, ki, kd)

    values: dict[str, float] = {}
    for line in normalized_text.splitlines():
        for key, attr in (("K?P", "kp"), ("K?I", "ki"), ("K?D", "kd")):
            match = re.search(rf"\b{key}\s*[:=]\s*{number_pattern}", line, re.IGNORECASE)
            if match:
                values[attr] = float(match.group(1))

    if len(values) == 3:
        return PIDParams(values["kp"], values["ki"], values["kd"])
    raise ValueError("PID values not found in the response.")


def _optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _optional_bool(value)
    raise ValueError(f"Boolean LLM setting expected, got {value!r}")


def _thinking_request_body(settings: LLMSettings) -> dict[str, object] | None:
    if settings.enable_thinking is None:
        return None
    if settings.provider == "deepseek":
        mode = "enabled" if settings.enable_thinking else "disabled"
        return {"thinking": {"type": mode}}
    if settings.provider == "vllm":
        return {"chat_template_kwargs": {"enable_thinking": settings.enable_thinking}}
    return {"enable_thinking": settings.enable_thinking}


def split_profile(profile: str) -> tuple[str, str]:
    if "_" not in profile:
        raise ValueError(
            f"LLM profile must use SERVICE_MODEL format, for example DS_Deepseek-V4-Flash: {profile}"
        )
    provider_key, model_alias = profile.split("_", 1)
    if not provider_key or not model_alias:
        raise ValueError(f"Invalid LLM profile: {profile}")
    return provider_key, model_alias


def provider_env_prefix(provider_key: str) -> str:
    normalized = normalize_env_key(provider_key)
    aliases = {
        "DS": "DEEPSEEK",
        "DEEPSEEK": "DEEPSEEK",
        "VLLM": "VLLM",
        "QWEN": "QWEN",
        "DASHSCOPE": "QWEN",
        "SAMBANOVA": "SAMBANOVA",
        "LLAMA": "SAMBANOVA",
    }
    return aliases.get(normalized, normalized)


def normalize_env_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def infer_model_from_alias(model_alias: str) -> str:
    return re.sub(r"[\s_]+", "-", model_alias.strip().lower())


def ensure_no_proxy_for_base_url(base_url: str | None) -> None:
    if not base_url:
        return
    host = urlparse(base_url).hostname
    if not host or not _should_bypass_proxy(host):
        return

    for key in ("NO_PROXY", "no_proxy"):
        entries = _split_no_proxy(os.environ.get(key, ""))
        if host not in entries:
            entries.append(host)
            os.environ[key] = ",".join(entries)


def _split_no_proxy(value: str) -> list[str]:
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def _should_bypass_proxy(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized in {"localhost", "127.0.0.1", "::1"} or normalized.endswith(".local"):
        return True
    try:
        address = ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback or address.is_private
