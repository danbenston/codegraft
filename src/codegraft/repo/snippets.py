"""Bounded snippet extraction — the evidence actually sent to the model.

For each ranked file we select a small, relevant slice rather than the whole
file: the header/imports (cheap structural context) plus short windows around
keyword hits. Hard caps on lines-per-file and a total character budget keep the
prompt small and the cost predictable — token discipline is part of the product.
"""

from __future__ import annotations

from pathlib import Path

from codegraft.config import Config
from codegraft.models.repo import RankedFile, RepoScan, Snippet
from codegraft.utils.text import extract_keywords, tokenize

HEADER_LINES = 12   # top-of-file imports / module docstring
WINDOW = 3          # lines of context on each side of a keyword hit


def _line_has_keyword(line: str, keywords: set[str]) -> bool:
    return any(tok in keywords for tok in tokenize(line))


def _hit_blocks(lines: list[str], keywords: set[str]) -> list[list[int]]:
    """Merged keyword windows as ``[start, end, hits, first_hit]``, in file order.

    Each keyword-bearing line contributes a ±``WINDOW`` window; touching or
    overlapping windows merge into one block. ``hits`` counts the keyword lines
    inside the merged block — the density signal that decides which blocks survive
    a binding line cap — and ``first_hit`` is the block's earliest keyword line.
    """

    blocks: list[list[int]] = []
    for i, line in enumerate(lines):
        if not _line_has_keyword(line, keywords):
            continue
        lo, hi = max(0, i - WINDOW), min(len(lines) - 1, i + WINDOW)
        if blocks and lo <= blocks[-1][1] + 1:
            blocks[-1][1] = max(blocks[-1][1], hi)
            blocks[-1][2] += 1
        else:
            blocks.append([lo, hi, 1, i])
    return blocks


def _select_line_indices(
    lines: list[str], keywords: set[str], max_lines: int
) -> tuple[list[int], bool]:
    """Choose which line indices to keep. Returns (sorted_indices, truncated)."""

    header = range(min(HEADER_LINES, len(lines)))
    blocks = _hit_blocks(lines, keywords)

    keep: set[int] = set(header)
    for start, end, _hits, _first in blocks:
        keep.update(range(start, end + 1))
    if len(keep) <= max_lines:
        # Everything fits: identical selection to a plain header + windows union.
        return sorted(keep), False

    # Over the cap. Trimming the sorted union (which is what we used to do) keeps
    # the *earliest* lines, so the header plus an incidental early mention could
    # consume the whole budget and drop the definition the request is actually
    # about. Instead cap the header's share of a tight budget, then spend the rest
    # on the densest keyword blocks first — an aside loses to the real cluster.
    header_budget = min(HEADER_LINES, max(1, max_lines // 4))
    keep = set(range(min(header_budget, len(lines))))
    for start, end, _hits, first_hit in sorted(blocks, key=lambda b: (-b[2], b[0])):
        room = max_lines - len(keep)
        if room <= 0:
            break
        if end + 1 - start <= room:
            keep.update(range(start, end + 1))
        else:
            # Too tight for the whole window: lead with the keyword line itself
            # rather than the leading context that would push it out of frame.
            keep.update(range(first_hit, min(end + 1, first_hit + room)))
    return sorted(keep)[:max_lines], True


def _render(lines: list[str], indices: list[int]) -> str:
    """Render selected lines with 1-based numbers and gap markers."""

    out: list[str] = []
    prev = -1
    for idx in indices:
        if prev != -1 and idx != prev + 1:
            out.append("        ...")
        out.append(f"{idx + 1:>6}  {lines[idx].rstrip()}")
        prev = idx
    return "\n".join(out)


def extract_snippets(
    request: str, ranked: list[RankedFile], scan: RepoScan, config: Config
) -> list[Snippet]:
    """Extract bounded snippets for the ranked files, under the global budget."""

    keywords = extract_keywords(request)
    root = Path(scan.root)
    budget = config.analysis.context_char_budget
    max_lines = config.analysis.max_snippet_lines_per_file

    snippets: list[Snippet] = []
    used_chars = 0

    for r in ranked:
        if used_chars >= budget:
            break
        try:
            text = (root / r.path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()

        indices, truncated = _select_line_indices(lines, keywords, max_lines)
        if not indices:
            continue
        body = _render(lines, indices)

        # Respect the remaining budget; skip a snippet that would blow it.
        if used_chars + len(body) > budget and snippets:
            break

        snippets.append(
            Snippet(
                path=r.path,
                content=body,
                line_count=len(indices),
                char_count=len(body),
                truncated=truncated,
            )
        )
        used_chars += len(body)

    return snippets
