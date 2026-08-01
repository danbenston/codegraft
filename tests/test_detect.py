"""Tests for language/manifest/entry-point detection and tree rendering."""

from __future__ import annotations

from pathlib import Path

from codegraft.repo import detect
from codegraft.repo.tree import build_tree


def test_language_mix_counts_code_only() -> None:
    paths = ["app/main.py", "app/util.py", "web/index.ts", "README.md", "data.json"]
    mix = detect.language_mix(paths)
    assert mix["Python"] == 2
    assert mix["TypeScript"] == 1
    assert "Markdown" not in mix  # docs/config excluded from the mix
    assert detect.primary_language(mix) == "Python"


def test_manifests_prefer_shallow() -> None:
    paths = ["sub/pyproject.toml", "pyproject.toml", "app/models.py"]
    manifests = detect.find_manifests(paths)
    assert manifests[0] == "pyproject.toml"  # root before nested


def test_entry_points_and_tests() -> None:
    paths = ["app/main.py", "tests/test_x.py", "pkg/util.py"]
    assert "app/main.py" in detect.find_entry_points(paths)
    assert "tests" in detect.find_test_paths(paths)


def test_find_test_paths_recognizes_js_ts_test_matrix() -> None:
    """Co-located `*.test.*` / `*.spec.*` files across the JS/TS extensions are
    tests — including the `.tsx`/`.jsx` variants that React component tests use
    (the omission that made `affected_tests` drop them)."""
    paths = [
        "src/components/Board.test.tsx",
        "src/components/Board.tsx",
        "src/util/fleet.test.ts",
        "src/util/calc.spec.tsx",
        "src/util/legacy.spec.js",
        "src/util/widget.jsx",  # not a test
        "src/util/data.test.json",  # not a JS/TS file → not a test
    ]
    tests = detect.find_test_paths(paths)
    assert "src/components/Board.test.tsx" in tests
    assert "src/util/fleet.test.ts" in tests
    assert "src/util/calc.spec.tsx" in tests
    assert "src/util/legacy.spec.js" in tests
    assert "src/components/Board.tsx" not in tests
    assert "src/util/widget.jsx" not in tests
    assert "src/util/data.test.json" not in tests


def test_build_tree_depth_and_width() -> None:
    paths = [f"src/pkg/mod{i}.py" for i in range(30)]
    tree = build_tree(paths, max_depth=2, max_entries=20)
    assert "src/" in tree
    assert "pkg/" in tree
    # Depth limit collapses deeper contents into an entry-count hint.
    assert "entries" in tree or "more" in tree


def test_django_style_tests_py_is_a_test_path() -> None:
    # `django-admin startapp` generates `<app>/tests.py`, which matches none of
    # the dir/prefix/suffix rules — so on a stock Django project every test file
    # was invisible and an empty affected_tests selection looked like a real answer.
    tests = detect.find_test_paths(
        ["shop/models.py", "shop/tests.py", "blog/test.py", "core/util.py"]
    )
    assert "shop/tests.py" in tests
    assert "blog/test.py" in tests
    assert "shop/models.py" not in tests
    assert "core/util.py" not in tests


def test_frameworks_ignore_manifest_prose(tmp_path: Path) -> None:
    # Regression: markers were matched as substrings of the whole file, so a
    # description mentioning "next" or "reactive" reported Next.js and React —
    # false frameworks that went into the planning prompt as fact.
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n'
        'description = "Plan the next feature; reactive views"\n'
        'dependencies = ["fastapi[standard]>=0.110", "SQLAlchemy>=2.0"]\n',
        encoding="utf-8",
    )
    found = detect.detect_frameworks(tmp_path, ["pyproject.toml"])
    assert found == ["FastAPI", "SQLAlchemy"]


def test_frameworks_match_hyphenated_and_scoped_packages(tmp_path: Path) -> None:
    # A marker must still match when it is a package *prefix* (actix -> actix-web)
    # or a path segment (gin-gonic inside a go.mod module path).
    (tmp_path / "package.json").write_text(
        '{"description":"a next-generation reactive app",'
        '"dependencies":{"@nestjs/core":"^10","react-dom":"18"}}',
        encoding="utf-8",
    )
    assert detect.detect_frameworks(tmp_path, ["package.json"]) == ["React", "NestJS"]

    (tmp_path / "go.mod").write_text(
        "module x\nrequire github.com/gin-gonic/gin v1.9.1\n", encoding="utf-8"
    )
    assert detect.detect_frameworks(tmp_path, ["go.mod"]) == ["Gin"]


def test_frameworks_survive_a_malformed_manifest(tmp_path: Path) -> None:
    # Unparseable JSON must fall back to the token scan, not crash or go silent.
    (tmp_path / "package.json").write_text(
        '{"dependencies": {"express": "4",,,', encoding="utf-8"
    )
    assert detect.detect_frameworks(tmp_path, ["package.json"]) == ["Express"]
