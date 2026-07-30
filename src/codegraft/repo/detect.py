"""Lightweight codebase characterization.

No language server, no AST — just extension mapping, well-known filenames, and a
shallow read of manifests for framework hints. Enough to tell a planning model
"this is a Python FastAPI service with a tests/ dir", which is all it needs.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

# Extension → language. Lowercased extension without the dot.
_EXT_LANG: dict[str, str] = {
    "py": "Python",
    "pyi": "Python",
    "js": "JavaScript",
    "jsx": "JavaScript",
    "mjs": "JavaScript",
    "cjs": "JavaScript",
    "ts": "TypeScript",
    "tsx": "TypeScript",
    "go": "Go",
    "rs": "Rust",
    "java": "Java",
    "kt": "Kotlin",
    "rb": "Ruby",
    "php": "PHP",
    "cs": "C#",
    "cpp": "C++",
    "cc": "C++",
    "c": "C",
    "h": "C/C++ header",
    "hpp": "C++ header",
    "swift": "Swift",
    "scala": "Scala",
    "sh": "Shell",
    "sql": "SQL",
    "html": "HTML",
    "css": "CSS",
    "scss": "CSS",
    "vue": "Vue",
    "svelte": "Svelte",
    "md": "Markdown",
    "yml": "YAML",
    "yaml": "YAML",
    "toml": "TOML",
    "json": "JSON",
}

# Well-known filenames → language (extensionless or special).
_FILENAME_LANG: dict[str, str] = {
    "Dockerfile": "Dockerfile",
    "Makefile": "Makefile",
    "pyproject.toml": "Python",
    "package.json": "JavaScript",
    "go.mod": "Go",
    "Cargo.toml": "Rust",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
    "pom.xml": "Java",
}

_MANIFEST_FILES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    "composer.json",
    "Dockerfile",
    "Makefile",
}

# Entry-point filenames (basename match).
_ENTRY_BASENAMES = {
    "main.py",
    "app.py",
    "__main__.py",
    "manage.py",
    "wsgi.py",
    "asgi.py",
    "index.js",
    "index.ts",
    "server.js",
    "server.ts",
    "app.js",
    "app.ts",
    "main.go",
    "main.rs",
}

# Substring → framework. Checked against manifest *contents* (dependency names).
_FRAMEWORK_MARKERS: dict[str, str] = {
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "starlette": "Starlette",
    "sqlalchemy": "SQLAlchemy",
    "pydantic": "Pydantic",
    "express": "Express",
    "next": "Next.js",
    "react": "React",
    "vue": "Vue",
    "svelte": "Svelte",
    "nestjs": "NestJS",
    "@nestjs": "NestJS",
    "gin-gonic": "Gin",
    "fiber": "Fiber",
    "rails": "Rails",
    "laravel": "Laravel",
    "spring-boot": "Spring Boot",
    "actix": "Actix",
    "axum": "Axum",
}


def _ext(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if "." in name:
        return name.rsplit(".", 1)[-1].lower()
    return ""


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def language_of(path: str) -> str | None:
    """Best-guess language for a single path, or None if unknown."""

    base = _basename(path)
    if base in _FILENAME_LANG:
        return _FILENAME_LANG[base]
    return _EXT_LANG.get(_ext(path))


def language_mix(paths: list[str]) -> dict[str, int]:
    """Count files per detected language, sorted high-to-low."""

    counts: dict[str, int] = {}
    for path in paths:
        lang = language_of(path)
        if lang and lang not in {"Markdown", "JSON", "YAML", "TOML"}:
            counts[lang] = counts.get(lang, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def primary_language(mix: dict[str, int]) -> str:
    return next(iter(mix), "")


def find_manifests(paths: list[str]) -> list[str]:
    """Manifest files, preferring shallower paths (root manifests matter most)."""

    found = [p for p in paths if _basename(p) in _MANIFEST_FILES]
    return sorted(found, key=lambda p: (p.count("/"), p))


def find_entry_points(paths: list[str]) -> list[str]:
    found = [p for p in paths if _basename(p) in _ENTRY_BASENAMES]
    return sorted(found, key=lambda p: (p.count("/"), p))


# JS/TS source extensions that carry `.test.`/`.spec.` test files. Kept in step
# with imports._JS_EXTS so "what counts as a test" and "what resolves as a module"
# don't drift — the omission of the `x`/`jsx` variants here used to make
# `affected_tests` and the ranking `test` signal silently ignore React component
# tests (`*.test.tsx`), the dominant test form on a TS frontend.
_JS_TS_EXTS = {"ts", "tsx", "js", "jsx", "mjs", "cjs"}


def _is_js_ts_test(base: str) -> bool:
    """True for `*.test.{ts,tsx,js,jsx,mjs,cjs}` and the `*.spec.*` variants."""

    stem, _, ext = base.rpartition(".")
    return ext in _JS_TS_EXTS and (stem.endswith(".test") or stem.endswith(".spec"))


def find_test_paths(paths: list[str]) -> list[str]:
    """Distinct top-level-ish directories/files that look like tests."""

    seen: list[str] = []
    for p in paths:
        base = _basename(p)
        segments = p.split("/")
        is_test = (
            any(seg in {"tests", "test", "__tests__", "spec"} for seg in segments)
            or base.startswith("test_")
            or base.endswith(("_test.go", "_test.py"))
            # `django-admin startapp` generates `<app>/tests.py`, which matches
            # none of the rules above — so on a stock Django project every test
            # file was invisible to `affected_tests` and to the ranking `test`
            # signal, and an empty selection looked like a correct answer.
            or base in {"tests.py", "test.py"}
            or _is_js_ts_test(base)
        )
        if is_test:
            # Record the test root directory if there is one, else the file.
            root = next(
                (
                    "/".join(segments[: i + 1])
                    for i, seg in enumerate(segments)
                    if seg in {"tests", "test", "__tests__", "spec"}
                ),
                p,
            )
            if root not in seen:
                seen.append(root)
    return sorted(seen)


# A dependency name is a run of these characters; anything else (quotes, commas,
# whitespace, version operators, `:`, `=`) terminates it. Used only for manifests
# we cannot parse structurally (requirements.txt, go.mod, Gemfile, pom.xml, ...),
# which are far less prose-heavy than pyproject.toml / package.json.
_DEP_TOKEN_RE = re.compile(r"[A-Za-z0-9._@/-]+")

# Keys whose *contents* are dependencies. Descending only into these is what
# stops a free-text `description` from contributing names.
_DEP_KEYS = frozenset(
    {
        "dependencies", "dev-dependencies", "devdependencies", "peerdependencies",
        "optionaldependencies", "optional-dependencies", "build-dependencies",
        "require", "require-dev",
    }
)

_PEP508_NAME_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _requirement_name(spec: str) -> str:
    """The bare package name from a requirement string.

    ``fastapi[standard]>=0.110`` -> ``fastapi``; ``django~=5.0`` -> ``django``.
    """

    match = _PEP508_NAME_RE.match(spec.lower())
    return match.group(1) if match else ""


def _dependency_tokens(text: str) -> set[str]:
    """Every candidate dependency name in an unparseable manifest, lowercased.

    Slash-separated paths contribute *all* their segments, because the framework
    marker is often a middle one (``github.com/gin-gonic/gin``).
    """

    tokens: set[str] = set()
    for raw in _DEP_TOKEN_RE.findall(text.lower()):
        token = raw.strip(".-/")
        if not token:
            continue
        tokens.add(token)
        tokens.update(seg for seg in token.split("/") if seg)
    return tokens


def _collect_dep_names(node: object, inside: bool, out: set[str]) -> None:
    """Recursively gather dependency names, descending into dependency keys only."""

    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).lower()
            if inside:
                out.add(lowered)  # a mapping under a dep key is name -> spec
            _collect_dep_names(value, inside or lowered in _DEP_KEYS, out)
    elif isinstance(node, list):
        for item in node:
            if inside and isinstance(item, str):
                out.add(_requirement_name(item))
            else:
                _collect_dep_names(item, inside, out)


def _structured_dep_names(manifest: str, text: str) -> set[str] | None:
    """Dependency names parsed out of a manifest, or None if it isn't parseable.

    pyproject.toml and package.json — the two most common manifests — both carry
    free-text ``description``/``keywords`` fields, which is exactly why a raw scan
    misfires on them. For those we read only where dependencies actually live.
    """

    base = _basename(manifest)
    try:
        if base in {"pyproject.toml", "Cargo.toml"}:
            data: object = tomllib.loads(text)
        elif base in {"package.json", "composer.json"}:
            data = json.loads(text)
        else:
            return None
    except (tomllib.TOMLDecodeError, json.JSONDecodeError, ValueError):
        return None  # malformed manifest: fall back to the token scan

    names: set[str] = set()
    _collect_dep_names(data, False, names)
    names.discard("")
    return names


def _marker_matches(marker: str, names: set[str]) -> bool:
    """True if *marker* names one of *names* — exactly, or as its hyphenated or
    scoped package prefix (``actix`` matches ``actix-web``; ``spring-boot``
    matches ``spring-boot-starter-web``) — but never as a bare substring, so
    "reactive" does not count as "react"."""

    if marker in names:
        return True
    return any(n.startswith(marker + "-") or n.startswith(marker + "/") for n in names)


def detect_frameworks(root: Path, manifests: list[str]) -> list[str]:
    """Shallow-read manifests and match *dependency names* to known frameworks.

    Matching is on whole dependency names, not substrings of the raw file. A
    substring scan let a manifest's prose masquerade as a dependency — a
    description containing "next" or "reactive" reported Next.js and React — and
    those false frameworks went into the planning prompt as if they were fact.
    """

    found: list[str] = []
    for manifest in manifests:
        path = root / manifest
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        names = _structured_dep_names(manifest, text)
        if names is None:
            names = _dependency_tokens(text)
        for marker, name in _FRAMEWORK_MARKERS.items():
            if name not in found and _marker_matches(marker, names):
                found.append(name)
    return found
