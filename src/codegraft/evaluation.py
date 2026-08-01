"""Retrospective ranking evaluation — "how accurate was inspect?"

Treats real git history as a labelled benchmark: a past commit's *subject line*
is the feature request, and the *source files it changed* are the gold set of
files a good ranking should surface. For each commit we run the deterministic
analysis and score the ranked files against that gold set with standard
information-retrieval metrics (recall@k, MRR).

This is deliberately model-free and reproducible: no API key, no LLM, just the
ranker versus history. Two honest caveats, surfaced rather than hidden:

- We rank the *current* working tree, not a checkout of the commit's parent. So
  the implemented change is already present — a mild flattery of the ranker, and
  the reason renamed/deleted gold files simply drop out (they aren't candidates).
  A strict counterfactual would check out the parent; that is out of scope here.
- "Files changed in a commit" is noisy ground truth (incidental edits, multiple
  valid implementations). We therefore optimise for **recall** — did the central
  files surface — and let cost, not precision, penalise over-broad bundles.

The scorer (:func:`score_case`) is a pure function and unit-tested without git;
:func:`run_eval` wires git extraction and analysis around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from codegraft.config import Config
from codegraft.repo.analyze import analyze_repo
from codegraft.repo.discover import discover_repo
from codegraft.repo.git_scan import (
    commit_changed_files,
    commit_subject,
    recent_commit_shas,
)
from codegraft.repo.summarize import summarize


@dataclass
class CaseResult:
    """The scored outcome for a single commit."""

    sha: str
    subject: str
    gold_total: int          # changed files that are analysis candidates (evaluable)
    found: int               # evaluable gold files appearing anywhere in the ranking
    hits_at_k: int           # evaluable gold files within the top-k
    recall_at_k: float       # hits_at_k / gold_total  (0.0 when gold_total == 0)
    first_rank: int | None   # 1-based rank of the first gold hit, if any
    reciprocal_rank: float   # 1 / first_rank, else 0.0

    @property
    def evaluable(self) -> bool:
        """Whether this case had any gold files codegraft could have ranked."""

        return self.gold_total > 0


@dataclass
class EvalReport:
    """Per-commit results plus aggregates for one ranking configuration."""

    k: int
    use_import_edge: bool
    use_intent_roles: bool = True
    use_named_file_boost: bool = True
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def scored(self) -> list[CaseResult]:
        """Cases with at least one evaluable gold file (the rest are excluded
        from aggregates — a commit that only touched non-source files tells us
        nothing about ranking quality)."""

        return [c for c in self.cases if c.evaluable]

    @property
    def mean_recall_at_k(self) -> float:
        scored = self.scored
        return sum(c.recall_at_k for c in scored) / len(scored) if scored else 0.0

    @property
    def mean_reciprocal_rank(self) -> float:
        scored = self.scored
        return sum(c.reciprocal_rank for c in scored) / len(scored) if scored else 0.0


def score_case(
    sha: str,
    subject: str,
    gold: set[str],
    ranked_paths: list[str],
    candidate_paths: set[str],
    k: int,
) -> CaseResult:
    """Score one commit's ranking against its gold set. Pure — no git, no I/O.

    Only gold files that are actually analysis *candidates* are evaluable; a gold
    file the scan never saw (deleted/renamed/ignored since) can't be ranked, so
    counting it would unfairly tax recall.
    """

    evaluable = gold & candidate_paths
    if not evaluable:
        return CaseResult(sha, subject, 0, 0, 0, 0.0, None, 0.0)

    top_k = set(ranked_paths[:k])
    hits_at_k = sum(1 for g in evaluable if g in top_k)

    first_rank: int | None = None
    for index, path in enumerate(ranked_paths, 1):
        if path in evaluable:
            first_rank = index
            break

    found = sum(1 for g in evaluable if g in set(ranked_paths))
    return CaseResult(
        sha=sha,
        subject=subject,
        gold_total=len(evaluable),
        found=found,
        hits_at_k=hits_at_k,
        recall_at_k=hits_at_k / len(evaluable),
        first_rank=first_rank,
        reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
    )


def resolve_commits(root: Path, commits: list[str] | None, last: int) -> list[str]:
    """Decide which commits to evaluate: explicit SHAs win, else the most recent
    *last* (defaulting to 10 when neither is given)."""

    if commits:
        return commits
    return recent_commit_shas(root, last if last > 0 else 10)


def run_eval(
    root: Path,
    config: Config,
    shas: list[str],
    *,
    k: int = 10,
    use_import_edge: bool = True,
    use_intent_roles: bool = True,
    use_named_file_boost: bool = True,
) -> EvalReport:
    """Build a gold set from each commit and score the ranker against it.

    *config* is copied before the ranking toggles are applied, so the caller's
    object is never mutated (an ablation runs both ways off one config).
    """

    cfg = config.model_copy(deep=True)
    cfg.analysis.use_import_edge = use_import_edge
    cfg.analysis.use_intent_roles = use_intent_roles
    cfg.analysis.use_named_file_boost = use_named_file_boost

    # Discovery and summarization are request-independent, so build them once and
    # reuse across commits — matching what the graph validators already do. Each
    # commit used to trigger a full rescan (git subprocess + per-file stat +
    # binary sniff), making an --ablation run pay for 2N scans it did not need.
    scan = discover_repo(root, cfg)
    summary = summarize(scan)
    candidate_paths = {f.path for f in scan.files}

    report = EvalReport(
        k=k, use_import_edge=use_import_edge, use_intent_roles=use_intent_roles,
        use_named_file_boost=use_named_file_boost,
    )
    for sha in shas:
        subject = commit_subject(root, sha)
        gold = commit_changed_files(root, sha)
        analysis = analyze_repo(subject, root, cfg, scan=scan, summary=summary)
        ranked_paths = [r.path for r in analysis.ranked]
        report.cases.append(
            score_case(sha, subject, gold, ranked_paths, candidate_paths, k)
        )
    return report
