"""OpenAI provider: structured-output plan generation.

Mirrors the Anthropic adapter and reuses the *exact same* prompt assembly — the
whole point of the provider abstraction is that only the SDK call differs. Uses
``client.beta.chat.completions.parse(response_format=ImplementationPlan)`` so the
model is constrained to our schema and the SDK validates into a Pydantic object.

The SDK is an optional dependency (``codegraft[openai]``), imported lazily; tests
inject a fake client. Use an OpenAI model id via config/``--model`` (the default
model is a Claude model, so the openai provider needs an explicit one).
"""

from __future__ import annotations

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


class OpenAIProvider:
    """Generates plans via the OpenAI Chat Completions API with structured outputs."""

    def __init__(self, config: Config, client: Any | None = None) -> None:
        self._config = config
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        key = self._config.secrets.openai_api_key
        if not key:
            raise ProviderError(
                "OPENAI_API_KEY is not set. Add it to your environment or a "
                ".env file (see `codegraft init`)."
            )
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ProviderError(
                "The openai SDK is not installed. Install it with: "
                'pip install "codegraft[openai]"'
            ) from exc

        self._client = openai.OpenAI(api_key=key)
        return self._client

    @staticmethod
    def _parse(client: Any, kwargs: dict[str, Any]) -> Any:
        """Call the structured-output endpoint, retrying once without a rejected
        ``temperature`` so an unknown reasoning model degrades instead of failing."""

        try:
            return client.beta.chat.completions.parse(**kwargs)
        except Exception as exc:
            if "temperature" not in kwargs or not rejects_temperature(exc):
                raise
            return client.beta.chat.completions.parse(
                **{k: v for k, v in kwargs.items() if k != "temperature"}
            )

    def generate_plan(self, request: PlanningRequest) -> ImplementationPlan:
        client = self._ensure_client()
        system, user = build_planning_prompt(request)

        model = self._config.provider.model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": ImplementationPlan,
        }
        # Same gate the Anthropic adapter applies: reasoning models reject
        # sampling params, and sending one turns a valid config into a 400.
        temperature = self._config.provider.temperature
        if temperature is not None and supports_temperature(model):
            kwargs["temperature"] = temperature

        try:
            response = self._parse(client, kwargs)
        except Exception as exc:  # wrap any SDK/transport error with context
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        message = response.choices[0].message
        plan = getattr(message, "parsed", None)
        if not isinstance(plan, ImplementationPlan):
            refusal = getattr(message, "refusal", None)
            raise PlanValidationError(
                "OpenAI response did not parse into an ImplementationPlan"
                + (f" (refusal: {refusal})" if refusal else "")
            )
        return plan
