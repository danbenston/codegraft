"""Tests for heuristic symbol-granularity fetch (get_symbol / find_symbol)."""

from __future__ import annotations

from pathlib import Path

from codegraft.config import Config
from codegraft.repo.discover import discover_repo
from codegraft.repo.symbols import find_symbol
from tests.conftest import write


def _config(root: Path) -> Config:
    config = Config.load(root)
    config.repo.prefer_git_ls_files = False
    return config


def _scan(root: Path):
    config = _config(root)
    return discover_repo(root, config), config


def test_python_block_extent_stops_at_dedent(tmp_path: Path) -> None:
    """A Python def's span ends where indentation returns to the def's level —
    not at EOF, and not swallowing the following top-level statement."""

    write(tmp_path, "pyproject.toml", "[project]\n")
    write(
        tmp_path,
        "app/svc.py",
        "import os\n"                    # 1
        "\n"                              # 2
        "def target(x):\n"                # 3  <- def
        "    y = x + 1\n"                 # 4
        "\n"                              # 5  blank inside body
        "    return y\n"                  # 6
        "\n"                              # 7
        "AFTER = 1\n",                    # 8  <- must NOT be in the span
    )
    scan, config = _scan(tmp_path)

    hits = find_symbol("target", scan, tmp_path, config)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.path == "app/svc.py"
    assert hit.span == (3, 6)                      # def line through `return y`
    assert hit.signature == "def target(x):"
    assert "AFTER = 1" not in hit.snippet
    assert "def target" in hit.snippet


def test_python_multiline_signature_keeps_body(tmp_path: Path) -> None:
    """A def whose signature spans several lines (closing `)` at column 0) must
    not be cut off at the signature — the body has to be included."""

    write(tmp_path, "pyproject.toml", "[project]\n")
    write(
        tmp_path,
        "app/svc.py",
        "def target(\n"                   # 1  <- def
        "    a,\n"                        # 2
        "    b,\n"                        # 3
        ") -> int:\n"                     # 4  closing paren at column 0
        "    total = a + b\n"             # 5
        "    return total\n"              # 6
        "\n"                              # 7
        "AFTER = 1\n",                    # 8
    )
    scan, config = _scan(tmp_path)

    hits = find_symbol("target", scan, tmp_path, config)

    assert len(hits) == 1
    assert hits[0].span == (1, 6)                  # through `return total`
    assert "AFTER = 1" not in hits[0].snippet
    assert "return total" in hits[0].snippet


def test_python_class_block_extent(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[project]\n")
    write(
        tmp_path,
        "app/models.py",
        "class Widget:\n"                 # 1
        "    def __init__(self):\n"       # 2
        "        self.n = 0\n"            # 3
        "\n"                              # 4
        "    def bump(self):\n"           # 5
        "        self.n += 1\n"           # 6
        "\n"                              # 7
        "TOP = 1\n",                      # 8
    )
    scan, config = _scan(tmp_path)

    hits = find_symbol("Widget", scan, tmp_path, config)

    assert len(hits) == 1
    assert hits[0].span == (1, 6)
    assert "TOP = 1" not in hits[0].snippet


def test_js_function_block_via_brace_balance(tmp_path: Path) -> None:
    write(tmp_path, "package.json", '{"dependencies": {}}\n')
    write(
        tmp_path,
        "src/app.js",
        "function renderBoard(state) {\n"  # 1
        "  if (state) {\n"                  # 2
        "    return draw(state);\n"         # 3
        "  }\n"                             # 4
        "  return null;\n"                  # 5
        "}\n"                               # 6
        "const after = 1;\n",              # 7
    )
    scan, config = _scan(tmp_path)

    hits = find_symbol("renderBoard", scan, tmp_path, config)

    assert len(hits) == 1
    assert hits[0].span == (1, 6)                  # balances the nested braces
    assert "const after" not in hits[0].snippet


def test_css_selector_block_extent(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[project]\n")
    write(
        tmp_path,
        "static/style.css",
        ".hex-grid {\n"                    # 1
        "  display: grid;\n"               # 2
        "  gap: 4px;\n"                    # 3
        "}\n"                              # 4
        ".other { color: red; }\n",        # 5
    )
    scan, config = _scan(tmp_path)

    hits = find_symbol("hex-grid", scan, tmp_path, config)

    assert len(hits) == 1
    assert hits[0].span == (1, 4)
    assert ".other" not in hits[0].snippet


def test_style_insensitive_match(tmp_path: Path) -> None:
    """camelCase query matches a snake_case definition (identifier-token equality)."""

    write(tmp_path, "pyproject.toml", "[project]\n")
    write(tmp_path, "app/svc.py", "def admin_panel(role):\n    return role\n")
    scan, config = _scan(tmp_path)

    hits = find_symbol("adminPanel", scan, tmp_path, config)

    assert len(hits) == 1
    assert hits[0].name == "admin_panel"


def test_name_defined_in_two_files_returns_both(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[project]\n")
    write(tmp_path, "a/dup.py", "def handler():\n    return 1\n")
    write(tmp_path, "b/dup.py", "def handler():\n    return 2\n")
    scan, config = _scan(tmp_path)

    hits = find_symbol("handler", scan, tmp_path, config)

    assert [h.path for h in hits] == ["a/dup.py", "b/dup.py"]   # sorted by path


def test_in_path_restricts_search(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[project]\n")
    write(tmp_path, "a/dup.py", "def handler():\n    return 1\n")
    write(tmp_path, "b/dup.py", "def handler():\n    return 2\n")
    scan, config = _scan(tmp_path)

    hits = find_symbol("handler", scan, tmp_path, config, in_path="b/dup.py")

    assert len(hits) == 1 and hits[0].path == "b/dup.py"


def test_unknown_name_is_empty_not_error(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[project]\n")
    write(tmp_path, "app/svc.py", "def real():\n    return 1\n")
    scan, config = _scan(tmp_path)

    assert find_symbol("nonexistent", scan, tmp_path, config) == []


def test_overlong_body_sets_truncated(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[project]\n")
    body = "\n".join(f"    x{i} = {i}" for i in range(50))
    write(tmp_path, "app/big.py", f"def huge():\n{body}\n")
    scan, config = _scan(tmp_path)
    config.analysis.max_snippet_lines_per_file = 10

    hits = find_symbol("huge", scan, tmp_path, config)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.truncated
    assert hit.span[1] - hit.span[0] + 1 == 10     # capped at the line limit


def test_multiple_symbols_on_one_line_are_all_findable(tmp_path: Path) -> None:
    # Regression: iteration stopped at the first match per line, so a symbol
    # sharing a line was unreachable — a multi-selector CSS rule exposed only its
    # first selector, which is a documented get_symbol use case.
    write(tmp_path, "pyproject.toml", "[project]\n")
    write(tmp_path, "web/site.css", ".card, .panel {\n  color: red;\n}\n")
    scan, config = discover_repo(tmp_path, _config(tmp_path)), _config(tmp_path)

    card = find_symbol("card", scan, tmp_path, config)
    panel = find_symbol("panel", scan, tmp_path, config)

    assert [h.path for h in card] == ["web/site.css"]
    assert [h.path for h in panel] == ["web/site.css"]
    # Both resolve to the same definition line, and neither is duplicated.
    assert card[0].span[0] == panel[0].span[0] == 1


def test_repeated_identifier_on_a_line_yields_one_hit(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", "[project]\n")
    write(tmp_path, "web/dup.css", ".btn, .btn {\n  color: red;\n}\n")
    config = _config(tmp_path)
    hits = find_symbol("btn", discover_repo(tmp_path, config), tmp_path, config)
    assert len(hits) == 1


def test_prefilter_does_not_change_results(tmp_path: Path) -> None:
    # The cheap "every target token must appear in the text" filter is a
    # necessary condition only — it must never drop a real definition.
    write(tmp_path, "pyproject.toml", "[project]\n")
    write(tmp_path, "app/a.py", "def admin_panel():\n    return 1\n")
    write(tmp_path, "app/b.py", "def unrelated():\n    return 2\n")
    config = _config(tmp_path)
    scan = discover_repo(tmp_path, config)

    # Style-insensitive matching still works through the prefilter.
    hits = find_symbol("adminPanel", scan, tmp_path, config)
    assert [h.path for h in hits] == ["app/a.py"]
    assert find_symbol("nosuchsymbol", scan, tmp_path, config) == []
