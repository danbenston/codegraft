"""Heuristic symbol-granularity fetch — one definition, not a whole-file read.

The bounded snippet extractor (``repo/snippets.py``) at finer granularity: given a
symbol *name*, locate where it is **defined** and return just that definition —
signature, body, and a line-numbered snippet — so an agent reads one function /
class / selector instead of the whole file. The single biggest read-side token
win observed in the field.

Location and block-extent are **heuristic**, not compiler-accurate: this reuses
the same definition regexes as the ranking symbol signal (``SYMBOL_RE`` for code,
``TEMPLATE_SYMBOL_RE`` / ``STYLE_SYMBOL_RE`` for HTML/CSS) and a small,
dependency-free block-extent tier per language. It is **not** ``find_references``
or ``go_to_definition`` — those need real cross-file resolution (an LSP). Precision
target: "good enough to avoid a full-file read."
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from codegraft.config import Config
from codegraft.models.repo import RepoScan
from codegraft.repo import detect
from codegraft.repo.graph import resolve_target
from codegraft.repo.patterns import (
    STYLE_SYMBOL_RE,
    SYMBOL_RE,
    TEMPLATE_SYMBOL_RE,
    is_code,
)
from codegraft.repo.snippets import _render
from codegraft.utils.text import split_identifier

# Languages whose block extent is found by balancing `{ }` from the definition
# line (CSS selector blocks balance the same way). Everything else falls back to
# Python dedent, HTML block matching, or a fixed window.
_BRACE_LANGS = {
    "JavaScript", "TypeScript", "Go", "Rust", "Java", "Kotlin", "Scala",
    "Swift", "PHP", "C", "C++", "C#", "C/C++ header", "C++ header", "Vue",
    "Svelte", "CSS",
}
_FALLBACK_WINDOW = 12   # lines, when the language's extent is unknown
_BRACE_LOOKAHEAD = 4    # give a brace-language def this many lines to open a block


@dataclass
class SymbolHit:
    """One located definition of a symbol."""

    path: str
    name: str                       # the identifier as it appears in the source
    signature: str                  # the def/class/selector line, stripped
    span: tuple[int, int]           # 1-based [start, end], inclusive
    snippet: str                    # line-numbered render of the span
    truncated: bool = False         # body exceeded the per-file line cap
    resolution: str = "heuristic"


def _regex_for(path: str) -> re.Pattern[str] | None:
    """The definition regex appropriate to *path*'s language, or None if the file
    type carries no request-nameable symbols (docs, data, plain text)."""

    if is_code(path):
        return SYMBOL_RE
    lang = detect.language_of(path)
    if lang == "HTML":
        return TEMPLATE_SYMBOL_RE
    if lang == "CSS":
        return STYLE_SYMBOL_RE
    return None


def _captured_name(match: re.Match[str]) -> str | None:
    """The identifier a definition regex captured (the first non-empty group)."""

    for group in match.groups():
        if group:
            return group
    return None


def _iter_symbol_defs(path: str, lines: list[str]) -> Iterator[tuple[int, str]]:
    """Yield ``(line_index, name)`` for each symbol *defined* in *path*.

    Cheap: matches the definition regex per line and yields the captured name
    without computing extents — extent/render is done only for the symbols a
    caller actually keeps.

    *Every* match on a line is yielded, not just the first. Stopping at the first
    made any symbol sharing a line unreachable — a multi-selector CSS rule
    (``.card, .panel {``) exposed only ``.card``, and ``const a = 1, b = 2``
    only ``a``. Duplicate names on one line are de-duplicated so a repeated
    identifier cannot produce two identical hits.
    """

    regex = _regex_for(path)
    if regex is None:
        return
    for i, line in enumerate(lines):
        seen: set[str] = set()
        for match in regex.finditer(line):
            name = _captured_name(match)
            if name and name not in seen:
                seen.add(name)
                yield i, name


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _python_end(lines: list[str], start: int) -> int:
    """End of a Python ``def``/``class`` block: the last line before indentation
    returns to ≤ the definition's own indent (blank lines don't terminate it).

    A multi-line signature is handled first — its continuation lines and the
    closing ``)`` often sit at column 0, which would otherwise look like a dedent
    and cut the body off. We walk to the header's terminating ``:`` (the first
    line at bracket-depth 0 that contains a colon) before applying the dedent rule.
    """

    base = _indent(lines[start])

    # 1) Advance past the (possibly multi-line) def/class header.
    depth = 0
    header_end = start
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
        header_end = i
        if depth <= 0 and ":" in lines[i]:
            break

    # 2) Consume the body until indentation returns to ≤ the definition's indent.
    end = header_end
    for i in range(header_end + 1, len(lines)):
        if not lines[i].strip():
            continue
        if _indent(lines[i]) <= base:
            break
        end = i
    return end


def _brace_end(lines: list[str], start: int) -> int:
    """End of a brace-delimited block by balancing ``{``/``}`` from *start*.

    A definition that opens no brace within a short lookahead (an expression /
    one-liner like ``const x = 5``) collapses to its own line. String/comment
    braces are not special-cased — heuristic, by design.
    """

    depth = 0
    seen_open = False
    for i in range(start, len(lines)):
        line = lines[i]
        opens, closes = line.count("{"), line.count("}")
        if opens:
            seen_open = True
        depth += opens - closes
        if seen_open and depth <= 0:
            return i
        if not seen_open and i - start >= _BRACE_LOOKAHEAD:
            return start
    return len(lines) - 1  # unbalanced — take to EOF; the caller caps it


def _html_end(lines: list[str], start: int) -> int:
    """End of an HTML template symbol: a ``{% block %}`` runs to its
    ``{% endblock %}``; a component tag falls back to a fixed window."""

    if "block" in lines[start]:
        for i in range(start, len(lines)):
            if "endblock" in lines[i]:
                return i
    return min(start + _FALLBACK_WINDOW, len(lines) - 1)


def _natural_end(lines: list[str], start: int, lang: str | None) -> int:
    if lang == "Python":
        return _python_end(lines, start)
    if lang == "HTML":
        return _html_end(lines, start)
    if lang in _BRACE_LANGS:
        return _brace_end(lines, start)
    return min(start + _FALLBACK_WINDOW, len(lines) - 1)


def symbol_span(
    lines: list[str], start: int, lang: str | None, max_lines: int
) -> tuple[int, bool]:
    """Return ``(end_index, truncated)`` for the definition starting at *start*,
    capped at *max_lines* (``truncated`` set when the natural body was longer)."""

    natural = _natural_end(lines, start, lang)
    length = natural - start + 1
    if length > max_lines:
        return min(start + max_lines - 1, len(lines) - 1), True
    return min(natural, len(lines) - 1), False


def _make_hit(path: str, lines: list[str], start: int, name: str, max_lines: int) -> SymbolHit:
    lang = detect.language_of(path)
    end, truncated = symbol_span(lines, start, lang, max_lines)
    indices = list(range(start, end + 1))
    return SymbolHit(
        path=path,
        name=name,
        signature=lines[start].strip(),
        span=(start + 1, end + 1),
        snippet=_render(lines, indices),
        truncated=truncated,
    )


def _may_define(lines: list[str], target: list[str]) -> bool:
    """Cheap necessary-condition filter before the per-line regex pass.

    A hit requires ``split_identifier(captured) == target``, so *every* token of
    the target must appear somewhere in the file's text. Checking that with a
    couple of substring scans skips the regex work for the overwhelming majority
    of files in the unscoped case, without changing which symbols are found.
    """

    if not target:
        return False
    lowered = "\n".join(lines).lower()
    return all(token in lowered for token in target)


def _read_lines(root: Path, path: str) -> list[str] | None:
    try:
        return (root / path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


def find_symbol(
    name: str,
    scan: RepoScan,
    root: Path,
    config: Config,
    *,
    in_path: str | None = None,
) -> list[SymbolHit]:
    """Locate every definition of *name* and return a bounded snippet for each.

    With *in_path* the search is restricted to that one file; otherwise it scans
    all candidate files (a name can be defined in several places) and returns
    **all** matches sorted by path, so the caller disambiguates rather than us
    guessing. An unknown name yields an empty list — not an error. Matching is
    style-insensitive (``adminPanel`` matches ``admin_panel``) via identifier
    token equality.

    Cost note: with no *in_path* this reads every candidate file in full. Prefer
    passing the path you already know (e.g. from ``select_context``).
    """

    target = split_identifier(name)
    if not target:
        return []

    if in_path is not None:
        resolved, _candidates = resolve_target(in_path, scan)
        files = [resolved] if resolved else []
    else:
        files = [f.path for f in scan.files]

    max_lines = config.analysis.max_snippet_lines_per_file
    hits: list[SymbolHit] = []
    for path in files:
        lines = _read_lines(root, path)
        if lines is None:
            continue
        if not _may_define(lines, target):
            continue
        for start, captured in _iter_symbol_defs(path, lines):
            if split_identifier(captured) == target:
                hits.append(_make_hit(path, lines, start, captured, max_lines))

    hits.sort(key=lambda h: (h.path, h.span[0]))
    return hits
