import os
from types import SimpleNamespace

from llmpidtuner.llm import (
    LLMSettings,
    PIDControllerClient,
    _thinking_request_body,
    ensure_no_proxy_for_base_url,
    normalize_env_key,
    provider_env_prefix,
    split_profile,
)


def test_split_profile():
    assert split_profile("DS_Deepseek-V4-Flash") == ("DS", "Deepseek-V4-Flash")
    assert split_profile("vLLM_llama3.1-8b") == ("vLLM", "llama3.1-8b")


def test_provider_aliases():
    assert provider_env_prefix("DS") == "DEEPSEEK"
    assert provider_env_prefix("vLLM") == "VLLM"
    assert provider_env_prefix("llama") == "SAMBANOVA"


def test_normalize_env_key():
    assert normalize_env_key("DS_Deepseek-V4-Flash") == "DS_DEEPSEEK_V4_FLASH"
    assert normalize_env_key("llama3.1-8b") == "LLAMA3_1_8B"


def test_deepseek_profile_reads_provider_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("DEEPSEEK_MODEL", "wrong-env-model")

    settings = LLMSettings.from_env(profile="DS_Deepseek-V4-Flash")

    assert settings.profile == "DS_Deepseek-V4-Flash"
    assert settings.provider == "deepseek"
    assert settings.api_key == "secret"
    assert settings.base_url == "https://api.deepseek.com/v1"
    assert settings.model == "deepseek-v4-flash"


def test_vllm_profile_uses_empty_key_default(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL", "wrong-env-model")

    settings = LLMSettings.from_env(profile="vLLM_llama3.1-8b")

    assert settings.provider == "vllm"
    assert settings.api_key == "EMPTY"
    assert settings.base_url == "http://localhost:8000/v1"
    assert settings.model == "llama3.1-8b"
    assert settings.enable_thinking is None


def test_profile_does_not_inherit_global_enable_thinking(monkeypatch):
    monkeypatch.setenv("LLM_ENABLE_THINKING", "false")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.delenv("DEEPSEEK_ENABLE_THINKING", raising=False)

    settings = LLMSettings.from_env(profile="DS_Deepseek-V4-Flash")

    assert settings.enable_thinking is None


def test_yaml_overrides_sampling_and_thinking():
    settings = LLMSettings(provider="qwen").with_overrides(
        {
            "temperature": 0.1,
            "top_p": 0.1,
            "enable_thinking": False,
            "seed": 42,
            "max_retries": 5,
            "max_tokens": 64,
        }
    )

    assert settings.temperature == 0.1
    assert settings.top_p == 0.1
    assert settings.enable_thinking is False
    assert settings.seed == 42
    assert settings.max_retries == 5
    assert settings.max_tokens == 64


def test_unknown_yaml_llm_setting_is_rejected():
    try:
        LLMSettings().with_overrides({"top_k": 20})
    except ValueError as error:
        assert "top_k" in str(error)
    else:
        raise AssertionError("Unsupported cross-provider top_k must be rejected")


def test_provider_specific_non_thinking_request_bodies():
    deepseek = LLMSettings(provider="deepseek", enable_thinking=False)
    qwen = LLMSettings(provider="qwen", enable_thinking=False)
    vllm = LLMSettings(provider="vllm", enable_thinking=False)

    assert _thinking_request_body(deepseek) == {"thinking": {"type": "disabled"}}
    assert _thinking_request_body(qwen) == {"enable_thinking": False}
    assert _thinking_request_body(vllm) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_client_passes_configured_retries_to_openai(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("llmpidtuner.llm.OpenAI", FakeOpenAI)
    PIDControllerClient(
        LLMSettings(api_key="secret", base_url="https://example.test/v1", max_retries=5)
    )

    assert captured["max_retries"] == 5


def test_client_passes_configured_seed_to_chat_completion(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="P:1.0; I:0.1; D:0.01")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("llmpidtuner.llm.OpenAI", FakeOpenAI)
    client = PIDControllerClient(
        LLMSettings(api_key="secret", base_url="https://example.test/v1", seed=42, max_tokens=64)
    )

    client.call_pid_parameters("system", "prompt")

    assert captured["seed"] == 42

    assert captured["max_tokens"] == 64

def test_private_base_url_is_added_to_no_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")

    ensure_no_proxy_for_base_url("http://192.0.2.10:11434/v1")

    assert "192.0.2.10" in os.environ["NO_PROXY"]
    assert "192.0.2.10" in os.environ["no_proxy"]


def test_public_base_url_is_not_added_to_no_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")

    ensure_no_proxy_for_base_url("https://api.deepseek.com/v1")

    assert "api.deepseek.com" not in os.environ["NO_PROXY"]
