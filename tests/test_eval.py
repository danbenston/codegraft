"""Tests for the retrospective ranking evaluation.

The scorer is pure and tested without git; the orchestration is exercised once
against a real throwaway git repo (skipped when git is absent).
"""

from __future__ import annotations

from pathlib import Path

from codegraft.config import Config
from codegraft.evaluation import EvalReport, run_eval, score_case
from codegraft.repo.git_scan import (
    commit_changed_files,
    commit_subject,
    recent_commit_shas,
)
from tests.conftest import git, init_git_repo, requires_git, write

# --- Pure scorer -----------------------------------------------------------

def test_score_case_recall_and_rank() -> None:
    gold = {"a.py", "b.py", "c.py"}
    candidates = {"a.py", "b.py", "c.py", "x.py", "y.py", "z.py"}
    ranked = ["a.py", "x.py", "b.py", "y.py", "z.py"]  # c.py never ranked

    result = score_case("sha", "subject", gold, ranked, candidates, k=3)

    assert result.gold_total == 3            # all gold are candidates
    assert result.hits_at_k == 2             # a.py and b.py in top-3
    assert result.recall_at_k == 2 / 3
    assert result.first_rank == 1            # a.py is first
    assert result.reciprocal_rank == 1.0
    assert result.found == 2                 # c.py is missing entirely
    assert result.evaluable


def test_score_case_excludes_non_candidate_gold() -> None:
    """A gold file the scan never saw (deleted/renamed) can't be ranked, so it
    must not count against recall."""

    gold = {"kept.py", "deleted.py"}
    candidates = {"kept.py", "other.py"}
    ranked = ["kept.py"]

    result = score_case("sha", "s", gold, ranked, candidates, k=10)

    assert result.gold_total == 1            # only kept.py is evaluable
    assert result.recall_at_k == 1.0
    assert result.first_rank == 1


def test_score_case_no_evaluable_gold_is_unscored() -> None:
    result = score_case("sha", "docs only", {"README.md"}, ["a.py"], {"a.py"}, k=10)

    assert result.gold_total == 0
    assert result.recall_at_k == 0.0
    assert result.first_rank is None
    assert not result.evaluable


def test_report_aggregates_only_scored_cases() -> None:
    report = EvalReport(k=10, use_import_edge=True)
    report.cases = [
        score_case("s1", "x", {"a.py"}, ["a.py"], {"a.py"}, k=10),       # recall 1.0
        score_case("s2", "y", {"b.py"}, ["c.py"], {"b.py", "c.py"}, k=10),  # recall 0.0
        score_case("s3", "z", {"README.md"}, ["a.py"], {"a.py"}, k=10),  # unscored
    ]

    assert len(report.scored) == 2           # the docs-only case is excluded
    assert report.mean_recall_at_k == 0.5    # (1.0 + 0.0) / 2
    assert report.mean_reciprocal_rank == 0.5  # (1.0 + 0.0) / 2


# --- Git extraction + orchestration ---------------------------------------

def _two_commit_repo(root: Path) -> None:
    init_git_repo(root)
    write(root, "pyproject.toml", '[project]\nname = "demo"\n')
    write(root, "README.md", "# demo\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "Initial project scaffold")

    write(
        root, "app/routes/admin.py",
        "def admin_panel(role):\n    # restrict admin access by role\n    return role\n",
    )
    write(
        root, "app/services/auth.py",
        "def check_access(user, role):\n    return user.role == role\n",
    )
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "Add role-based access control to admin routes")


@requires_git
def test_commit_helpers_extract_request_and_gold(tmp_path: Path) -> None:
    _two_commit_repo(tmp_path)
    shas = recent_commit_shas(tmp_path, 10)

    assert len(shas) == 2                    # newest first, no merges
    assert commit_subject(tmp_path, shas[0]) == \
        "Add role-based access control to admin routes"
    assert commit_changed_files(tmp_path, shas[0]) == {
        "app/routes/admin.py", "app/services/auth.py"
    }
    # --root makes the initial commit report its files too.
    assert "pyproject.toml" in commit_changed_files(tmp_path, shas[1])


@requires_git
def test_run_eval_scores_history(tmp_path: Path) -> None:
    _two_commit_repo(tmp_path)
    shas = recent_commit_shas(tmp_path, 10)

    report = run_eval(tmp_path, Config.load(tmp_path), shas, k=10)

    # The RBAC commit's files share keywords with its subject, so they rank.
    rbac = next(c for c in report.cases if "role-based" in c.subject)
    assert rbac.gold_total == 2
    assert rbac.found >= 1
    assert report.mean_recall_at_k > 0.0


@requires_git
def test_run_eval_ablation_runs_both_ways(tmp_path: Path) -> None:
    _two_commit_repo(tmp_path)
    shas = recent_commit_shas(tmp_path, 10)
    config = Config.load(tmp_path)

    on = run_eval(tmp_path, config, shas, k=10, use_import_edge=True)
    off = run_eval(tmp_path, config, shas, k=10, use_import_edge=False)

    assert on.use_import_edge and not off.use_import_edge
    # The caller's config is copied, never mutated, by either run.
    assert config.analysis.use_import_edge is True


def test_prebuilt_scan_matches_a_fresh_analysis(tmp_path: Path) -> None:
    """`analyze_repo` must rank identically whether it scans or is handed a scan.

    run_eval reuses one scan across every commit (discovery is
    request-independent); that optimisation is only safe if it changes nothing.
    """

    from codegraft.repo.analyze import analyze_repo
    from codegraft.repo.discover import discover_repo
    from codegraft.repo.summarize import summarize

    write(tmp_path, "pyproject.toml", "[project]\n")
    write(tmp_path, "app/routes/admin.py", "def admin(role):\n    return role\n")
    write(tmp_path, "app/models/user.py", "class User:\n    role = 'x'\n")
    config = Config.load(tmp_path)
    config.repo.prefer_git_ls_files = False

    scan = discover_repo(tmp_path, config)
    summary = summarize(scan)
    for request in ("add role based access", "rename the user model"):
        fresh = analyze_repo(request, tmp_path, config)
        reused = analyze_repo(request, tmp_path, config, scan=scan, summary=summary)
        assert [(r.path, r.score) for r in fresh.ranked] == [
            (r.path, r.score) for r in reused.ranked
        ]
        assert [s.path for s in fresh.snippets] == [s.path for s in reused.snippets]
