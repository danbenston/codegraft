"""Tests for the ranking-signal focusing of long requests."""

from __future__ import annotations

from codegraft.utils.text import (
    extract_keywords,
    extract_named_files,
    ranking_signal,
    request_intent,
)


def _intent(request: str) -> str:
    return request_intent(extract_keywords(request))


def test_intent_frontend_lean() -> None:
    assert _intent("migrate the frontend to react and typescript") == "frontend"
    assert _intent("restyle the dashboard template and css") == "frontend"


def test_intent_backend_lean() -> None:
    assert _intent("add a migration and a new schema for users") == "backend"
    assert _intent("secure the auth endpoints") == "backend"


def test_intent_none_when_ambiguous_or_absent() -> None:
    # Neither lexicon.
    assert _intent("improve logging and error handling") == "none"
    # Both lexicons hit -> deliberately unsteered.
    assert _intent("wire the react components to the api endpoints") == "none"


def test_named_files_basename_and_path() -> None:
    # A bare filename and a path-qualified mention both resolve; a path mention
    # also yields its basename so either form of candidate can match.
    assert extract_named_files("port the layout from style.css") == {"style.css"}
    assert extract_named_files("port static/css/style.css now") == {
        "static/css/style.css",
        "style.css",
    }


def test_named_files_only_known_extensions() -> None:
    # Prose that merely contains dots must not be mistaken for filenames.
    assert extract_named_files("bump to version 2.0, e.g. soon") == set()
    assert extract_named_files("speed up by 3.5x overall") == set()


def test_named_files_multiple_and_mixed() -> None:
    found = extract_named_files(
        "wire app.js and templates/board.html to core/demo/scenarios/board.json"
    )
    assert found == {
        "app.js",
        "templates/board.html",
        "board.html",
        "core/demo/scenarios/board.json",
        "board.json",
    }


def test_short_request_passes_through() -> None:
    req = "add an openai provider"
    assert ranking_signal(req) == req


def test_markdown_heading_is_used() -> None:
    req = (
        "# Add role-based access control\n\n"
        "We need to restrict admin endpoints. " * 10
    )
    assert ranking_signal(req) == "Add role-based access control"


def test_long_paragraph_uses_first_sentence() -> None:
    # Must exceed the focus threshold (a single long sentence-laden paragraph).
    req = (
        "We need to add a new OpenAI provider to the system. "
        "It should implement the same provider protocol as the existing Anthropic "
        "provider, handle errors gracefully, support configuration via the toml file "
        "and environment variables, include proper tests with mocks, and follow the "
        "existing code conventions throughout the entire codebase."
    )
    assert len(req) > 240  # guard: the focusing path is actually exercised
    assert ranking_signal(req) == "We need to add a new OpenAI provider to the system."


def test_later_section_heading_is_not_the_signal() -> None:
    # Regression: the heading scan used to walk *every* line, so a trailing
    # "## Constraints" section beat the prose goal and ranking ran on the single
    # word "Constraints". Only the opening line may supply a heading.
    req = (
        "Please add role-based access control to the admin routes so operators can "
        "restrict who sees the audit log. This needs to cover the middleware layer "
        "and the templates as well, and should be backwards compatible with the "
        "existing session cookie handling used by the legacy portal today.\n"
        "\n## Constraints\n- must not break existing sessions\n"
    )
    assert len(req.strip()) > 240  # guard: the focusing path is exercised
    signal = ranking_signal(req)
    assert signal.startswith("Please add role-based access control")
    assert "Constraints" not in signal


def test_fenced_comment_does_not_hijack_the_signal() -> None:
    # A "#" comment inside the request body is not a heading either.
    req = (
        "Cache the planner results between runs so repeated inspect calls on an "
        "unchanged repository do not redo the whole deterministic scan every time, "
        "which currently dominates the runtime of the evaluation harness.\n"
        "```python\n# TODO: pick an eviction policy\n```\n"
    )
    assert len(req.strip()) > 240
    assert ranking_signal(req).startswith("Cache the planner results")


def test_bare_hash_rule_is_skipped() -> None:
    req = (
        "###\n"
        "Rework the ignore engine so nested gitignore negation is handled the way "
        "git itself handles it, including re-inclusion from a deeper file, because "
        "the current fallback treats any matching spec as authoritative and that "
        "diverges from git whenever a deeper file re-includes a path.\n"
    )
    assert len(req.strip()) > 240
    assert ranking_signal(req).startswith("Rework the ignore engine")


def test_named_files_keeps_leading_dot_of_dotfiles() -> None:
    # Regression: the `\b` anchor refused to consume a mention's leading dot, so
    # ".eslintrc.json" was captured as "eslintrc.json" and matched no candidate.
    assert extract_named_files("port the .eslintrc.json rules") == {".eslintrc.json"}
    assert extract_named_files("update .github/workflows/ci.yml too") == {
        ".github/workflows/ci.yml",
        "ci.yml",
    }


def test_named_files_strips_only_a_relative_prefix() -> None:
    # "./" is a relative-path prefix and goes; a dotfile's own dot stays.
    assert extract_named_files("see ./src/app.py") == {"src/app.py", "app.py"}


def test_long_multiline_without_heading_uses_first_line() -> None:
    req = (
        "Implement caching for the planner\n"
        "- use an LRU cache\n"
        "- invalidate on file changes\n"
        "- add metrics and logging and tests and docs and benchmarks " * 5
    )
    assert ranking_signal(req) == "Implement caching for the planner"
