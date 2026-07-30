"""Snippet extraction and budget-enforcement tests."""

from __future__ import annotations

from pathlib import Path

from codegraft.config import Config
from codegraft.models.repo import RankedFile, RepoScan
from codegraft.repo.snippets import extract_snippets
from tests.conftest import write


def _scan_with(root: Path, rel: str, content: str) -> tuple[RepoScan, list[RankedFile]]:
    write(root, rel, content)
    scan = RepoScan(root=str(root), files=[], skipped=[], used_git=False)
    return scan, [RankedFile(path=rel, score=1.0)]


def test_snippet_respects_line_cap(tmp_path: Path) -> None:
    body = "\n".join(f"line role {i}" for i in range(500)) + "\n"
    scan, ranked = _scan_with(tmp_path, "big.py", body)

    config = Config.load(tmp_path)
    config.analysis.max_snippet_lines_per_file = 20
    snippets = extract_snippets("role", ranked, scan, config)

    assert len(snippets) == 1
    assert snippets[0].line_count <= 20
    assert snippets[0].truncated is True


def test_snippet_includes_header_and_keyword_windows(tmp_path: Path) -> None:
    lines = ["import os", "import sys"] + [f"filler {i}" for i in range(40)]
    lines[30] = "def grant_access(role):  # the relevant line"
    scan, ranked = _scan_with(tmp_path, "svc.py", "\n".join(lines) + "\n")

    config = Config.load(tmp_path)
    snippets = extract_snippets("access role", ranked, scan, config)
    content = snippets[0].content

    assert "import os" in content          # header retained
    assert "grant_access" in content       # keyword window retained
    assert "..." in content                # gap marker between header and window


def test_total_budget_enforced(tmp_path: Path) -> None:
    big = "\n".join(f"role line {i}" for i in range(300)) + "\n"
    write(tmp_path, "a.py", big)
    write(tmp_path, "b.py", big)
    write(tmp_path, "c.py", big)
    scan = RepoScan(root=str(tmp_path), files=[], skipped=[], used_git=False)
    ranked = [RankedFile(path=p, score=1.0) for p in ("a.py", "b.py", "c.py")]

    config = Config.load(tmp_path)
    config.analysis.context_char_budget = 400
    config.analysis.max_snippet_lines_per_file = 100
    snippets = extract_snippets("role", ranked, scan, config)

    total = sum(s.char_count for s in snippets)
    # First snippet may exceed on its own, but we must stop adding past budget.
    assert total <= 400 + snippets[0].char_count
    assert len(snippets) < 3


def test_tight_cap_keeps_the_dense_block_not_just_the_top(tmp_path: Path) -> None:
    """A binding line cap must not spend the whole budget on the file's head.

    Regression: selection trimmed the sorted union of (header + keyword windows),
    which keeps the *earliest* lines — so the 12-line header plus one incidental
    early mention could consume the cap and drop the definition the request is
    actually about. Density now decides which windows survive.
    """

    lines = ["import os"] + ["# filler"] * 20
    lines += ["def helper(): pass  # mentions role in passing"]
    lines += ["# filler"] * 300
    lines += ["class RolePolicy:", "    def check(self): ...", "    # role core logic"]
    scan, ranked = _scan_with(tmp_path, "svc.py", "\n".join(lines) + "\n")

    config = Config.load(tmp_path)
    config.analysis.max_snippet_lines_per_file = 20
    snippets = extract_snippets("role", ranked, scan, config)

    content = snippets[0].content
    assert snippets[0].line_count <= 20
    assert snippets[0].truncated is True
    assert "class RolePolicy:" in content       # the dense cluster survives
    assert "import os" in content               # some header context still kept


def test_generous_cap_keeps_header_and_every_window(tmp_path: Path) -> None:
    """When everything fits, selection is unchanged — header plus all windows."""

    lines = ["import os"] + ["# filler"] * 20
    lines += ["def role_one(): ..."] + ["# filler"] * 20 + ["def role_two(): ..."]
    scan, ranked = _scan_with(tmp_path, "svc.py", "\n".join(lines) + "\n")

    config = Config.load(tmp_path)
    snippets = extract_snippets("role", ranked, scan, config)
    content = snippets[0].content

    assert snippets[0].truncated is False
    assert "import os" in content
    assert "role_one" in content and "role_two" in content


def test_missing_file_is_skipped(tmp_path: Path) -> None:
    scan = RepoScan(root=str(tmp_path), files=[], skipped=[], used_git=False)
    ranked = [RankedFile(path="nope.py", score=1.0)]
    config = Config.load(tmp_path)
    assert extract_snippets("anything", ranked, scan, config) == []
