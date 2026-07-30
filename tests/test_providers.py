"""Provider tests — mocked by default, with an opt-in live smoke test."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from codegraft.config import Config
from codegraft.errors import PlanValidationError, ProviderError
from codegraft.models.plan import ImplementationPlan
from codegraft.models.repo import RepoSummary
from codegraft.planning.service import generate_plan, get_provider
from codegraft.providers.anthropic_provider import AnthropicProvider
from codegraft.providers.base import (
    PlanningRequest,
    rejects_temperature,
    supports_temperature,
)
from codegraft.providers.openai_provider import OpenAIProvider
from codegraft.providers.prompt import build_planning_prompt
from tests.conftest import write

# --- Fakes -----------------------------------------------------------------

class _FakeMessages:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response: object) -> None:
        self.messages = _FakeMessages(response)


def _content_response(plan_or_text, stop_reason: str = "end_turn"):
    """Mimic an Anthropic messages.create() response carrying JSON text."""

    text = plan_or_text if isinstance(plan_or_text, str) else plan_or_text.model_dump_json()
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
    )


def _sample_plan() -> ImplementationPlan:
    return ImplementationPlan(title="Sample", feature_summary="A sample plan.")


def _request() -> PlanningRequest:
    return PlanningRequest(
        feature_request="add caching to the planner",
        repo_root="/tmp/x",
        summary=RepoSummary(
            root="/tmp/x", file_count=3, primary_language="Python",
            directory_tree="src/\n  app.py",
        ),
    )


def _config(model: str = "claude-sonnet-4-6") -> Config:
    config = Config()
    config.provider.model = model
    config.secrets.anthropic_api_key = "test-key"
    return config


# --- Prompt assembly -------------------------------------------------------

def test_prompt_includes_evidence() -> None:
    system, user = build_planning_prompt(_request())
    assert "implementation-planning assistant" in system.lower()
    assert "anti-hallucination" in system.lower()
    assert "add caching to the planner" in user
    assert "<directory_tree>" in user and "app.py" in user


def test_prompt_favours_goal_over_structure_and_reuse() -> None:
    # Guards the planning-prompt change: plans should describe goals and prefer
    # reuse, not dictate internal structure (see the MeshiMate A/B finding).
    system, _ = build_planning_prompt(_request())
    lowered = system.lower()
    assert "goal" in lowered and "internal structure" in lowered
    assert "unifying with it" in lowered or "prefer extending or unifying" in lowered


# --- Anthropic provider (mocked) ------------------------------------------

def test_provider_returns_parsed_plan() -> None:
    client = _FakeClient(_content_response(_sample_plan()))
    provider = AnthropicProvider(_config(), client=client)

    plan = provider.generate_plan(_request())
    assert isinstance(plan, ImplementationPlan)
    assert plan.title == "Sample"

    # The call embeds the JSON schema in the prompt (instructed-JSON path, not
    # grammar-constrained structured output).
    call = client.messages.calls[0]
    assert "output_format" not in call
    assert call["model"] == "claude-sonnet-4-6"
    assert call["messages"][0]["role"] == "user"
    assert "json_schema" in call["messages"][0]["content"]
    assert "feature_summary" in call["messages"][0]["content"]  # schema field embedded


def test_json_extracted_from_surrounding_text() -> None:
    # Model wraps the JSON in fences / prose — we still extract and validate it.
    payload = "```json\n" + _sample_plan().model_dump_json() + "\n```\nDone."
    client = _FakeClient(_content_response(payload))
    plan = AnthropicProvider(_config(), client=client).generate_plan(_request())
    assert plan.title == "Sample"


def test_temperature_sent_only_when_supported() -> None:
    # sonnet-4-6 accepts temperature
    c1 = _FakeClient(_content_response(_sample_plan()))
    AnthropicProvider(_config("claude-sonnet-4-6"), client=c1).generate_plan(_request())
    assert c1.messages.calls[0].get("temperature") == 0.2

    # opus-4-8 rejects sampling params -> must be omitted
    c2 = _FakeClient(_content_response(_sample_plan()))
    AnthropicProvider(_config("claude-opus-4-8"), client=c2).generate_plan(_request())
    assert "temperature" not in c2.messages.calls[0]


def test_missing_key_raises_provider_error() -> None:
    config = _config()
    config.secrets.anthropic_api_key = None
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(config).generate_plan(_request())


def test_non_json_response_raises() -> None:
    client = _FakeClient(_content_response("Sorry, I cannot help with that."))
    provider = AnthropicProvider(_config(), client=client)
    with pytest.raises(PlanValidationError, match="no JSON object"):
        provider.generate_plan(_request())


def test_truncated_response_raises() -> None:
    client = _FakeClient(_content_response(_sample_plan(), stop_reason="max_tokens"))
    provider = AnthropicProvider(_config(), client=client)
    with pytest.raises(PlanValidationError, match="truncated"):
        provider.generate_plan(_request())


def test_sdk_error_wrapped_as_provider_error() -> None:
    client = _FakeClient(_content_response(_sample_plan()))

    def _raise(**kwargs):
        raise RuntimeError("network down")

    client.messages.create = _raise  # type: ignore[assignment]
    with pytest.raises(ProviderError, match="network down"):
        AnthropicProvider(_config(), client=client).generate_plan(_request())


def test_supports_temperature_helper() -> None:
    assert supports_temperature("claude-sonnet-4-6")
    assert not supports_temperature("claude-opus-4-8")
    assert not supports_temperature("claude-fable-5")
    # OpenAI reasoning models reject sampling params the same way.
    assert not supports_temperature("gpt-5")
    assert not supports_temperature("o3-mini")
    assert supports_temperature("gpt-4o")


def test_rejects_temperature_recognises_refusal() -> None:
    assert rejects_temperature(RuntimeError(
        "400 Unsupported parameter: 'temperature' is not supported with this model"
    ))
    assert rejects_temperature(RuntimeError("`temperature` cannot be specified"))
    # Unrelated failures must not be mistaken for a parameter refusal.
    assert not rejects_temperature(RuntimeError("429 rate limited"))
    assert not rejects_temperature(RuntimeError("invalid api key"))


def test_anthropic_retries_once_without_rejected_temperature() -> None:
    """A model missing from the denylist must degrade, not fail: on a temperature
    refusal we retry once without it (a rejected request costs no tokens)."""

    client = _FakeClient(_content_response(_sample_plan()))
    real_create = client.messages.create
    seen: list[dict] = []

    def flaky(**kwargs):
        seen.append(kwargs)
        if "temperature" in kwargs:
            raise RuntimeError("400 unsupported parameter: temperature")
        return real_create(**kwargs)

    client.messages.create = flaky  # type: ignore[assignment]
    # sonnet-4-6 is on the "supported" side, so temperature is sent first...
    plan = AnthropicProvider(_config("claude-sonnet-4-6"), client=client).generate_plan(
        _request()
    )

    assert plan.title == "Sample"
    assert len(seen) == 2
    assert seen[0]["temperature"] == 0.2
    assert "temperature" not in seen[1]


def test_anthropic_unrelated_error_is_not_retried() -> None:
    client = _FakeClient(_content_response(_sample_plan()))
    calls: list[dict] = []

    def always_fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("network down")

    client.messages.create = always_fail  # type: ignore[assignment]
    with pytest.raises(ProviderError, match="network down"):
        AnthropicProvider(_config(), client=client).generate_plan(_request())
    assert len(calls) == 1  # no pointless second call


# --- OpenAI provider (mocked) ---------------------------------------------

def _oa_config(model: str = "gpt-4o") -> Config:
    config = Config()
    config.provider.name = "openai"
    config.provider.model = model
    config.secrets.openai_api_key = "test-key"
    return config


def _fake_openai_client(parsed, refusal=None):
    parser = SimpleNamespace(calls=[])

    def parse(**kwargs):
        parser.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=refusal))]
        )

    parser.parse = parse
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=parser)))
    return client, parser


def test_openai_returns_parsed_plan() -> None:
    client, parser = _fake_openai_client(_sample_plan())
    plan = OpenAIProvider(_oa_config(), client=client).generate_plan(_request())

    assert isinstance(plan, ImplementationPlan)
    call = parser.calls[0]
    assert call["response_format"] is ImplementationPlan
    assert call["model"] == "gpt-4o"
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["system", "user"]


def test_openai_missing_key_raises() -> None:
    config = _oa_config()
    config.secrets.openai_api_key = None
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAIProvider(config).generate_plan(_request())


def test_openai_unparsable_raises() -> None:
    client, _ = _fake_openai_client(parsed=None, refusal="cannot comply")
    with pytest.raises(PlanValidationError, match="refusal"):
        OpenAIProvider(_oa_config(), client=client).generate_plan(_request())


def test_openai_temperature_sent_only_when_supported() -> None:
    """Regression: the OpenAI adapter sent temperature unconditionally while the
    Anthropic one gated it, so a reasoning model turned a valid config into a 400."""

    client, parser = _fake_openai_client(_sample_plan())
    OpenAIProvider(_oa_config("gpt-4o"), client=client).generate_plan(_request())
    assert parser.calls[0].get("temperature") == 0.2

    client, parser = _fake_openai_client(_sample_plan())
    OpenAIProvider(_oa_config("gpt-5"), client=client).generate_plan(_request())
    assert "temperature" not in parser.calls[0]


def test_openai_retries_once_without_rejected_temperature() -> None:
    parser = SimpleNamespace(calls=[])

    def parse(**kwargs):
        parser.calls.append(kwargs)
        if "temperature" in kwargs:
            raise RuntimeError("400 Unsupported parameter: 'temperature'")
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(parsed=_sample_plan(), refusal=None)
            )]
        )

    parser.parse = parse
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=parser)))
    plan = OpenAIProvider(_oa_config("gpt-4o"), client=client).generate_plan(_request())

    assert plan.title == "Sample"
    assert len(parser.calls) == 2
    assert "temperature" not in parser.calls[1]


def test_openai_sdk_error_wrapped() -> None:
    def parse(**kwargs):
        raise RuntimeError("rate limited")

    parser = SimpleNamespace(parse=parse)
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=parser)))
    with pytest.raises(ProviderError, match="rate limited"):
        OpenAIProvider(_oa_config(), client=client).generate_plan(_request())


# --- Service orchestration -------------------------------------------------

class _FakeProvider:
    def __init__(self) -> None:
        self.seen: PlanningRequest | None = None

    def generate_plan(self, request: PlanningRequest) -> ImplementationPlan:
        self.seen = request
        return _sample_plan()


def test_generate_plan_stamps_metadata(tmp_path: Path) -> None:
    write(tmp_path, "app/routes/admin.py", "def admin(role):\n    return role\n")
    config = Config()
    config.repo.prefer_git_ls_files = False
    fake = _FakeProvider()

    plan = generate_plan(
        "add role-based access to admin", tmp_path, config, provider=fake
    )

    # Metadata is authored by codegraft, not the provider.
    assert plan.metadata.provider == "anthropic"
    assert plan.metadata.model == config.provider.model
    assert plan.metadata.timestamp
    # The provider actually received the analyzed evidence bundle.
    assert fake.seen is not None
    assert fake.seen.feature_request == "add role-based access to admin"
    assert fake.seen.summary.primary_language == "Python"


def test_get_provider_openai_returns_provider() -> None:
    config = Config()
    config.provider.name = "openai"
    assert isinstance(get_provider(config), OpenAIProvider)


def test_get_provider_unknown() -> None:
    config = Config()
    config.provider.name = "bogus"
    with pytest.raises(ProviderError, match="unknown provider"):
        get_provider(config)


# --- Live smoke test (opt-in) ---------------------------------------------

@pytest.mark.live_anthropic
def test_live_anthropic_smoke(tmp_path: Path) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    write(tmp_path, "pyproject.toml", '[project]\ndependencies = ["fastapi"]\n')
    write(
        tmp_path, "app/routes/admin.py",
        "from fastapi import APIRouter\nrouter = APIRouter()\n"
        "@router.get('/admin')\ndef admin():\n    return {}\n",
    )
    config = Config.load(tmp_path)
    plan = generate_plan("add role-based access control to admin routes", tmp_path, config)

    assert isinstance(plan, ImplementationPlan)
    assert plan.title
    assert plan.implementation_phases
