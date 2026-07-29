"""Anthropic provider: instructed-JSON plan generation.

We deliberately do **not** use grammar-constrained structured outputs
(``messages.parse`` / ``output_config.format``). Our ``ImplementationPlan``
schema is large and deeply nested, and the structured-output engine compiles the
schema into a constrained-decoding grammar up front — which times out on a schema
this size ("Grammar compilation timed out"). So instead we hand the model the
JSON Schema in the prompt, ask for a single JSON object, and validate the result
with the same Pydantic model. This keeps the typed contract (the real value)
without depending on grammar compilation.

The SDK is an optional dependency (``codegraft[anthropic]``), imported lazily;
tests inject a fake client.
"""

from __future__ import annotations

import json
from typing import Any

from codegraft.config import Config
from codegraft.errors import PlanValidationError, ProviderError
from codegraft.models.plan import ImplementationPlan
from codegraft.providers.base import (
    PlanningRequest,
    rejects_temperature,
    supports_temperature,
)
from codegraft.providers.prompt import build_planning_prompt


def _plan_schema_json() -> str:
    """The plan JSON Schema, minus `metadata` (codegraft stamps that itself)."""

    schema = ImplementationPlan.model_json_schema()
    schema.get("properties", {}).pop("metadata", None)
    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r != "metadata"]
    return json.dumps(schema, separators=(",", ":"))


_JSON_INSTRUCTION = (
    "\n\n---\n"
    "Respond with a SINGLE JSON object conforming to the JSON Schema below. "
    "Output ONLY the JSON object — no prose, no explanation, no markdown code "
    "fences. Do not include a `metadata` field.\n\n"
    "<json_schema>\n{schema}\n</json_schema>"
)


def _response_text(response: Any) -> str:
    """Concatenate the text blocks of an Anthropic message response."""

    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _parse_plan(text: str) -> ImplementationPlan:
    """Extract and validate the JSON object from the model's text response."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise PlanValidationError("Anthropic response contained no JSON object")
    try:
        return ImplementationPlan.model_validate_json(text[start : end + 1])
    except Exception as exc:
        raise PlanValidationError(f"could not parse plan JSON: {exc}") from exc


class AnthropicProvider:
    """Generates plans via the Anthropic Messages API (instructed JSON)."""

    def __init__(self, config: Config, client: Any | None = None) -> None:
        self._config = config
        self._client = client  # injectable for tests; lazily created otherwise

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        key = self._config.secrets.anthropic_api_key
        if not key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Add it to your environment or a "
                ".env file (see `codegraft init`)."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ProviderError(
                "The anthropic SDK is not installed. Install it with: "
                'pip install "codegraft[anthropic]"'
            ) from exc

        self._client = anthropic.Anthropic(api_key=key)
        return self._client

    @staticmethod
    def _create(client: Any, kwargs: dict[str, Any]) -> Any:
        """Call the Messages API, retrying once without a rejected ``temperature``
        so a model missing from ``_NO_SAMPLING_PREFIXES`` degrades instead of
        failing. A rejected request consumes no tokens, so the retry is free."""

        try:
            return client.messages.create(**kwargs)
        except Exception as exc:
            if "temperature" not in kwargs or not rejects_temperature(exc):
                raise
            return client.messages.create(
                **{k: v for k, v in kwargs.items() if k != "temperature"}
            )

    def generate_plan(self, request: PlanningRequest) -> ImplementationPlan:
        client = self._ensure_client()
        system, user = build_planning_prompt(request)
        user = user + _JSON_INSTRUCTION.format(schema=_plan_schema_json())

        model = self._config.provider.model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self._config.provider.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        temperature = self._config.provider.temperature
        if temperature is not None and supports_temperature(model):
            kwargs["temperature"] = temperature

        try:
            response = self._create(client, kwargs)
        except Exception as exc:  # wrap any SDK/transport error with context
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        if getattr(response, "stop_reason", None) == "max_tokens":
            raise PlanValidationError(
                "the plan was truncated (hit max_tokens before the JSON closed); "
                "increase provider.max_output_tokens in codegraft.toml"
            )
        return _parse_plan(_response_text(response))
