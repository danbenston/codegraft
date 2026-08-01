"""Typed configuration for codegraft.

Two sources, kept deliberately separate:

- Behaviour config lives in a repo-local ``codegraft.toml`` (provider choice,
  scan limits, budgets). It is safe to commit and easy to read.
- Secrets (API keys) come only from the environment / ``.env`` via
  ``pydantic-settings``. They are never written to config files or rendered.

For Phase 1 nothing here calls a provider yet; this just establishes the typed
shape so later phases have a stable contract to read from.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from codegraft.errors import ConfigError

CONFIG_FILENAME = "codegraft.toml"


class ProviderConfig(BaseModel):
    name: str = "anthropic"
    # Intentionally not pinned to a model that may be deprecated; overridable in
    # codegraft.toml and via --model. Balanced default for planning work.
    model: str = "claude-sonnet-4-6"
    # Optional: newer models (Opus 4.7+/Fable) reject sampling params, so the
    # provider only sends this to models that accept it. None = never send.
    temperature: float | None = 0.2
    # A full ImplementationPlan as JSON (phases + handoff prompts) is sizeable;
    # keep generous headroom so the JSON isn't truncated mid-object.
    max_output_tokens: int = 12000


class RepoConfig(BaseModel):
    respect_gitignore: bool = True
    prefer_git_ls_files: bool = True
    max_file_bytes: int = 180_000
    max_file_lines: int = 4_000
    max_candidate_files: int = 2_500
    # Vendor / build / noise directories. Secrets and binaries are handled
    # authoritatively by repo.safety, so they are intentionally NOT duplicated
    # here (that keeps skip reasons accurate, e.g. ".env" -> "secret").
    extra_excludes: list[str] = Field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.venv/**",
            "**/dist/**",
            "**/build/**",
            "**/__pycache__/**",
        ]
    )


class AnalysisConfig(BaseModel):
    max_ranked_files: int = 12
    max_snippet_lines_per_file: int = 160
    always_include_manifests: bool = True
    include_tests_when_relevant: bool = True
    context_char_budget: int = 50_000
    # Boost files imported by already-relevant files (architectural centrality).
    # Exposed mainly so `codegraft eval --ablation` can measure its contribution.
    use_import_edge: bool = True
    # Modulate architectural role weights by the request's frontend/backend lean,
    # so an explicitly frontend request stops losing to backend files that earn a
    # structural role bump frontend files cannot. No-op for requests with no clear
    # lean. Toggleable so `eval` can measure it on a full-stack repo.
    use_intent_roles: bool = True
    # Treat a file the request names outright ("port static/css/style.css") as a
    # near-direct hit, extracted from the full request so it survives the
    # ranking-signal focus that truncates long requests. Toggleable for ablation.
    use_named_file_boost: bool = True


class OutputConfig(BaseModel):
    output_dir: str = "plans"
    write_debug_json: bool = True
    write_debug_context: bool = False


class Secrets(BaseSettings):
    """API keys, read only from the environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None


class Config(BaseModel):
    """The full resolved configuration object."""

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    repo: RepoConfig = Field(default_factory=RepoConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    secrets: Secrets = Field(default_factory=Secrets)

    @classmethod
    def load(cls, repo_root: Path | None = None) -> Config:
        """Load config from ``codegraft.toml`` in *repo_root* (if present) plus env.

        Missing file is fine: every field has a default, so an unconfigured repo
        still produces a valid Config.
        """

        data: dict = {}
        config_path = repo_root / CONFIG_FILENAME if repo_root is not None else None
        if config_path is not None and config_path.is_file():
            try:
                with config_path.open("rb") as fh:
                    data = tomllib.load(fh)
            except tomllib.TOMLDecodeError as exc:
                raise ConfigError(f"{config_path} is not valid TOML: {exc}") from exc
            except OSError as exc:
                raise ConfigError(f"could not read {config_path}: {exc}") from exc

        # A typo'd value (e.g. max_ranked_files = "twelve") used to surface as a
        # raw Pydantic ValidationError traceback, because the CLI only catches
        # CodegraftError. Name the file and the offending field instead.
        try:
            return cls(
                provider=ProviderConfig(**data.get("provider", {})),
                repo=RepoConfig(**data.get("repo", {})),
                analysis=AnalysisConfig(**data.get("analysis", {})),
                output=OutputConfig(**data.get("output", {})),
                secrets=Secrets(),
            )
        except ValidationError as exc:
            where = str(config_path) if config_path is not None else "configuration"
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            raise ConfigError(f"invalid config in {where} — {problems}") from exc


# Default TOML content written by `codegraft init`.
DEFAULT_CONFIG_TOML = """\
[provider]
name = "anthropic"
model = "claude-sonnet-4-6"
temperature = 0.2
max_output_tokens = 12000

[repo]
respect_gitignore = true
prefer_git_ls_files = true
max_file_bytes = 180000
max_file_lines = 4000
max_candidate_files = 2500
# Secrets/binaries are handled by codegraft's safety filter, not listed here.
extra_excludes = [
  "**/node_modules/**",
  "**/.venv/**",
  "**/dist/**",
  "**/build/**",
  "**/__pycache__/**",
]

[analysis]
max_ranked_files = 12
max_snippet_lines_per_file = 160
always_include_manifests = true
include_tests_when_relevant = true
context_char_budget = 50000
use_import_edge = true
use_intent_roles = true
use_named_file_boost = true

[output]
output_dir = "plans"
write_debug_json = true
write_debug_context = false
"""
