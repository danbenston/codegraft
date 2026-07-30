"""Symbol-definition patterns shared by ranking and symbol lookup.

These describe *what a definition looks like* in a given language, which is not a
ranking concept and not a lookup concept — both stages need the same answer, and
they must not drift or `get_symbol` would find definitions the ranker never
scored (and vice versa).

They previously lived in ``repo/rank.py``, which made ``repo/symbols.py`` reach
upward into the ranker for four private names. Nothing here depends on ranking,
so it lives on its own and both callers import it.
"""

from __future__ import annotations

import re

from codegraft.repo import detect

# Languages whose files carry meaningful def/class symbols. Docs and data files
# (Markdown, JSON, TOML, ...) are excluded so code blocks inside a README or the
# project blueprint don't get scored as if they were real source symbols.
NON_CODE_LANGS = {"Markdown", "JSON", "YAML", "TOML", "HTML", "CSS", "Text"}


def is_code(path: str) -> bool:
    """True if *path* is a source file whose def/class symbols are meaningful."""

    lang = detect.language_of(path)
    return lang is not None and lang not in NON_CODE_LANGS


SYMBOL_RE = re.compile(
    r"\b(?:def|class|func|function|type|struct|interface|fn|const|var)\s+([A-Za-z_]\w*)"
)

# Template "symbols": the keyword-bearing names in an HTML template that a request
# can actually name. Django/Jinja block names (`{% block content %}`) and
# component-like custom element tags (`<UserCard>`, `<user-card>`) — but NOT plain
# HTML tags (`<div>`), which would be pure noise. Gives templates a symbol signal
# they were denied by being a "non-code" language.
TEMPLATE_SYMBOL_RE = re.compile(
    r"{%-?\s*block\s+([A-Za-z_][\w-]*)"          # {% block content %}
    r"|<([A-Z][A-Za-z0-9]*|[a-z][\w]*-[\w-]+)\b"  # <Component> or <web-component>
)

# Stylesheet "symbols": the keyword-bearing names in a CSS file a request can
# actually name — class selectors (`.hex-grid`), id selectors (`#board`), and
# custom properties (`--hex-size`). Plain property names/values and numeric
# fractions (`.5em`) are deliberately *not* matched (pure noise), mirroring how
# the template regex skips bare `<div>` tags. Gives stylesheets a symbol signal
# they were denied by being a "non-code" language.
STYLE_SYMBOL_RE = re.compile(
    r"[.#]([A-Za-z_][\w-]*)"      # .class or #id selector
    r"|(--[A-Za-z_][\w-]*)"        # --custom-property
)
