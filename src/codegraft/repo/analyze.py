"""Single deterministic-analysis entry point.

Runs the whole model-free pipeline — discover → summarize → rank → snippets —
and returns a bundle. Both `inspect` (Phase 3) and the planning service
(Phase 4) build on this; keeping it in one place means they agree on exactly
what evidence a request produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codegraft.config import Config
from codegraft.models.repo import RankedFile, RepoScan, RepoSummary, Snippet
from codegraft.repo.discover import discover_repo
from codegraft.repo.rank import rank_files
from codegraft.repo.snippets import extract_snippets
from codegraft.repo.summarize import summarize
from codegraft.utils.text import extract_named_files, ranking_signal
from codegraft.utils.tokens import TokenEstimate, estimate_savings


@dataclass
class RepoAnalysis:
    """Everything the deterministic layer produces for one request."""

    request: str          # full request text — goes to the model
    ranking_signal: str   # focused signal actually used to rank/snippet
    scan: RepoScan
    summary: RepoSummary
    ranked: list[RankedFile]
    snippets: list[Snippet]

    @property
    def context_chars(self) -> int:
        return sum(s.char_count for s in self.snippets)

    def token_estimate(self) -> TokenEstimate:
        """Estimate tokens sent vs. reading the *selected* files in full.

        The baseline is only the files codegraft actually surfaced as
        evidence (the snippet sources), read whole — not the entire repo. An
        agent implementing one feature would open those files, not unrelated
        ones, so this is the realistic alternative the bundle is measured
        against. The figure thus reflects the snippet-extraction win, not an
        inflated whole-repo comparison.

        Both sides are measured in **characters**. The baseline used to come from
        ``size_bytes``, which overstates any file with non-ASCII content (UTF-8
        multibyte) and so quietly inflated a figure the module docstring defends
        as honest; re-reading the selected files keeps the units comparable.
        """

        selected_paths = {s.path for s in self.snippets}
        sizes = {f.path: f.size_bytes for f in self.scan.files}
        root = Path(self.scan.root)
        baseline_chars = 0
        for path in selected_paths:
            try:
                baseline_chars += len(
                    (root / path).read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                # Unreadable now (deleted/renamed mid-run): fall back to the
                # byte size, which is an upper bound rather than a silent zero.
                baseline_chars += sizes.get(path, 0)
        return estimate_savings(
            baseline_chars=baseline_chars,
            bundle_chars=self.context_chars,
            selected_files=len(selected_paths),
        )


def analyze_repo(
    request: str,
    root: Path,
    config: Config,
    subdir: str | None = None,
    *,
    scan: RepoScan | None = None,
    summary: RepoSummary | None = None,
) -> RepoAnalysis:
    """Discover, summarize, rank, and extract snippets for *request*.

    Ranking and snippet selection use a focused *signal* derived from the
    request (so a long, constraint-heavy request doesn't dilute file ranking),
    while the full request is preserved for the planning prompt.

    Discovery and summarization do not depend on *request*, so a caller running
    many requests against one tree (the eval harness) can pass a prebuilt *scan*
    and *summary* to skip the rescan. Both are treated as read-only. When *scan*
    is supplied it must already be scoped to *subdir*, which is then only
    recorded, not re-applied.
    """

    signal = ranking_signal(request)
    # Files the request names explicitly are pulled from the *full* request, not
    # the focused signal — a long request's file mention often lives in a clause
    # `ranking_signal` truncates away.
    named_files = extract_named_files(request)
    if scan is None:
        scan = discover_repo(root, config, subdir=subdir)
    if summary is None:
        summary = summarize(scan)
    ranked = rank_files(signal, scan, summary, config, named_files=named_files)
    snippets = extract_snippets(signal, ranked, scan, config)
    return RepoAnalysis(
        request=request,
        ranking_signal=signal,
        scan=scan,
        summary=summary,
        ranked=ranked,
        snippets=snippets,
    )
