"""Provider contract.

A provider's only job is to turn a typed :class:`PlanningRequest` (the
deterministic context bundle) into a validated :class:`ImplementationPlan`.
Everything provider-specific — SDK calls, prompt-to-API mapping, error handling
— lives behind this boundary; the rest of codegraft never imports an SDK.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from codegraft.models.plan import ImplementationPlan
from codegraft.models.repo import RankedFile, RepoSummary, Snippet


class PlanningRequest(BaseModel):
    """The deterministic evidence bundle handed to a provider.

    Built by the planning service from a :class:`~codegraft.repo.analyze.RepoAnalysis`.
    Holds only what the prompt needs — not the full file scan.
    """

    feature_request: str = Field(description="The user's feature request.")
    repo_root: str = Field(description="Absolute repo root (for context only).")
    summary: RepoSummary = Field(description="Compact repo characterization.")
    ranked: list[RankedFile] = Field(
        default_factory=list, description="Relevance-ranked candidate files."
    )
    snippets: list[Snippet] = Field(
        default_factory=list, description="Bounded code evidence for the model."
    )


class PlanProvider(Protocol):
    """Structural type every provider adapter satisfies."""

    def generate_plan(self, request: PlanningRequest) -> ImplementationPlan:
        ...


# Models known to reject sampling parameters (temperature/top_p/top_k) — the
# provider must not send `temperature` to these or the API returns 400. This is a
# fast path only: providers keep shipping models with this restriction, so the
# list goes stale by default and `rejects_temperature` below is the real net.
_NO_SAMPLING_PREFIXES = (
    "claude-opus-4-7", "claude-opus-4-8", "claude-fable-5",
    # OpenAI reasoning models reject sampling params the same way.
    "o1", "o3", "o4", "gpt-5",
)


def supports_temperature(model: str) -> bool:
    """True if *model* is not known to reject a temperature parameter."""

    return not any(model.startswith(p) for p in _NO_SAMPLING_PREFIXES)


# Substrings that mark an API error as "you sent temperature and I don't take it".
_TEMPERATURE_REFUSAL_MARKERS = (
    "unsupported", "not supported", "does not support", "unexpected", "invalid",
    "unrecognized", "cannot be specified",
)


def rejects_temperature(exc: Exception) -> bool:
    """True if *exc* looks like an API complaint about the ``temperature`` param.

    Recognising the refusal is what keeps a stale ``_NO_SAMPLING_PREFIXES`` from
    becoming a hard failure: a provider that rejects the parameter costs no tokens
    (the request never ran), so the caller can safely retry once without it.
    """

    message = str(exc).lower()
    return "temperature" in message and any(
        marker in message for marker in _TEMPERATURE_REFUSAL_MARKERS
    )
