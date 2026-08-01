"""Tests for the token-savings estimate."""

from __future__ import annotations

from pathlib import Path

from codegraft.config import Config
from codegraft.repo.analyze import analyze_repo
from codegraft.utils.tokens import estimate_savings, estimate_tokens
from tests.conftest import write


def test_estimate_tokens_basics() -> None:
    assert estimate_tokens(0) == 0
    assert estimate_tokens(-5) == 0
    assert estimate_tokens(4) == 1
    assert estimate_tokens(4000) == 1000  # ~4 chars/token


def test_estimate_savings_math() -> None:
    est = estimate_savings(baseline_chars=40_000, bundle_chars=8_000, selected_files=12)
    assert est.baseline_tokens == 10_000
    assert est.bundle_tokens == 2_000
    assert est.saved_tokens == 8_000
    assert est.saved_pct == 80.0
    assert "saved" in est.summary()


def test_savings_never_negative() -> None:
    # A bundle larger than the baseline (degenerate) clamps to zero saved.
    est = estimate_savings(baseline_chars=100, bundle_chars=400, selected_files=1)
    assert est.saved_tokens == 0
    assert est.saved_pct == 0.0


def test_empty_repo_is_safe() -> None:
    est = estimate_savings(baseline_chars=0, bundle_chars=0, selected_files=0)
    assert est.baseline_tokens == 0
    assert est.saved_tokens == 0
    assert est.saved_pct == 0.0


def test_analysis_reports_savings(tmp_path: Path) -> None:
    # A long, relevant file should yield a bundle smaller than reading it whole,
    # so savings are positive. The baseline counts only the *selected* files
    # (snippet sources), not the unrelated bulk files in the repo.
    write(tmp_path, "app/auth.py", "def check_role(role):\n    return role\n" * 80)
    for i in range(6):
        write(tmp_path, f"app/unrelated{i}.py", "x = 1\n" * 200)  # bulk, irrelevant
    config = Config.load(tmp_path)
    config.repo.prefer_git_ls_files = False

    est = analyze_repo("check role", tmp_path, config).token_estimate()
    assert est.selected_files >= 1
    assert est.bundle_tokens <= est.baseline_tokens
    assert est.saved_tokens >= 0


def test_baseline_excludes_unselected_files(tmp_path: Path) -> None:
    # The baseline must not include files codegraft never surfaced. With more
    # candidates than codegraft selects, the un-surfaced files (however large)
    # must not inflate the "saved" figure.
    write(tmp_path, "app/auth.py", "def check_role(role):\n    return role\n" * 80)
    for i in range(8):
        write(tmp_path, f"app/unrelated{i}.py", "x = 1\n" * 5_000)  # huge, irrelevant
    config = Config.load(tmp_path)
    config.repo.prefer_git_ls_files = False
    config.analysis.max_ranked_files = 2  # surface fewer files than the repo holds

    analysis = analyze_repo("check role", tmp_path, config)
    est = analysis.token_estimate()
    selected = {s.path for s in analysis.snippets}
    sizes = {f.path: f.size_bytes for f in analysis.scan.files}
    # Fewer files surfaced than exist: the baseline is a genuine subset.
    assert len(selected) < len(sizes)
    # Baseline equals exactly the selected files, read in full — nothing else.
    # Measured in *characters*, the same unit as the snippet side: `size_bytes`
    # counts a CRLF line ending as two and multibyte UTF-8 as several, either of
    # which would overstate a figure `utils/tokens.py` defends as honest.
    expected = sum(
        len((tmp_path / p).read_text(encoding="utf-8", errors="replace"))
        for p in selected
    )
    assert est.baseline_chars == expected
    # The whole-repo size is far larger; the honest baseline ignores it.
    assert est.baseline_chars < sum(sizes.values())
