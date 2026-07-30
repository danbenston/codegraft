"""Value baseline for ``get_symbol`` — the read-side token win, measured.

``get_symbol`` exists to replace a whole-file ``Read`` with one definition, so its
value is **how many lines you avoid reading** when you fetch a symbol instead of
its file. This validator quantifies that against the repo itself — no git, no
model: it locates every definition the same way ``get_symbol`` does, then reports

- **coverage** — the share of files where ``get_symbol`` is even applicable (have
  ≥1 locatable symbol), and
- **savings** — for each located symbol, ``1 - span_lines / file_lines``: the
  fraction of the file you *don't* read. The headline number.

Block-extent correctness (Python dedent, brace balance, CSS block) is pinned by
``tests/test_symbols.py``; this measures the payoff, not the mechanics.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from codegraft.config import Config
from codegraft.repo import detect
from codegraft.repo.discover import discover_repo
from codegraft.repo.symbols import _iter_symbol_defs, _read_lines, symbol_span


@dataclass
class SymbolEvalReport:
    """Aggregate value baseline for ``get_symbol`` over a repository."""

    files_total: int
    files_with_symbols: int
    symbols: int
    truncated: int               # symbols whose body hit the per-file line cap
    mean_savings_pct: float      # mean (1 - span/file) over located symbols, %
    median_span_lines: float
    median_file_lines: float

    @property
    def coverage_pct(self) -> float:
        return 100.0 * self.files_with_symbols / self.files_total if self.files_total else 0.0


def run_symbol_eval(root: Path, config: Config) -> SymbolEvalReport:
    """Locate every definition in the repo and measure the per-symbol read savings."""

    scan = discover_repo(root, config)
    max_lines = config.analysis.max_snippet_lines_per_file

    files_with = 0
    savings: list[float] = []
    spans: list[int] = []
    file_line_counts: list[int] = []
    symbols = 0
    truncated = 0

    for f in scan.files:
        lines = _read_lines(root, f.path)
        if not lines:
            continue
        lang = detect.language_of(f.path)
        defs = list(_iter_symbol_defs(f.path, lines))
        if not defs:
            continue
        files_with += 1
        file_line_counts.append(len(lines))
        for start, _name in defs:
            end, was_truncated = symbol_span(lines, start, lang, max_lines)
            span_len = end - start + 1
            symbols += 1
            spans.append(span_len)
            if was_truncated:
                truncated += 1
            savings.append(1.0 - span_len / len(lines))  # lines is non-empty here

    return SymbolEvalReport(
        files_total=scan.file_count,
        files_with_symbols=files_with,
        symbols=symbols,
        truncated=truncated,
        mean_savings_pct=100.0 * statistics.fmean(savings) if savings else 0.0,
        median_span_lines=statistics.median(spans) if spans else 0.0,
        median_file_lines=statistics.median(file_line_counts) if file_line_counts else 0.0,
    )
