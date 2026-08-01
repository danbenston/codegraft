"""Deterministic relevant-file ranking.

This is the heart of codegraft: given a feature request and a scanned repo,
decide which files are worth showing the planning model. It is intentionally an
explainable heuristic, not an LLM call or an embedding lookup — every file
carries a `signals` breakdown so weak ranking is debuggable via `inspect`.

Two stages keep file I/O bounded on large repos:
  1. Cheap pass over *all* candidates using path/role/structure signals only.
  2. Read content for the top `stage2_limit` files and add content-keyword and
     symbol-overlap signals, then re-rank.

A final diversity penalty stops one directory from dominating the results.
"""

from __future__ import annotations

from pathlib import Path

from codegraft.config import Config
from codegraft.models.repo import RankedFile, RepoScan, RepoSummary
from codegraft.repo import detect
from codegraft.repo.imports import (
    _build_path_index,
    _read_head,
    _referenced_paths,
)
from codegraft.repo.patterns import (
    STYLE_SYMBOL_RE,
    SYMBOL_RE,
    TEMPLATE_SYMBOL_RE,
    is_code,
)
from codegraft.utils.text import (
    extract_keywords,
    request_intent,
    split_identifier,
    tokenize,
)

# --- Scoring weights (tunable). Kept as named constants so `inspect` output can
# be read against them while tuning. ---
W_FILENAME_HIT = 6.0     # a keyword appears in the file's own name
W_NAMED_FILE = 8.0       # request named this exact file (e.g. "port style.css")
W_PATH_HIT = 3.0         # a keyword appears elsewhere in the path
W_ROLE = 1.0             # multiplier applied to per-role weights below
W_ENTRY = 2.0            # file is an application entry point
W_MANIFEST = 1.5         # root/build manifest (always mildly relevant)
W_TEST = 1.5             # test file, when the request looks test-related
W_CONTENT_HIT = 0.4      # per keyword occurrence in content (capped)
CONTENT_HIT_CAP = 12     # don't let one huge file dominate on raw counts
W_SYMBOL = 4.0           # per keyword matching a defined symbol name
ROOT_PROXIMITY = 1.0     # shallow files are slightly more likely to matter
DIVERSITY_PENALTY = 1.5  # subtracted per already-selected file in same dir...
DIVERSITY_PENALTY_CAP = 3  # ...but capped, so a single-folder scope isn't crushed
# Import-edge signal. A file imported by a few *highly relevant* files is likely
# a feature-specific shared module worth surfacing (e.g. a shared screen behind
# thin keyword-named wrappers). A file imported by *many* files is a ubiquitous
# utility (config, theme, base classes) — and commits rarely change those, so
# boosting them just evicts the actually-relevant files. The eval harness showed
# the naive version hurt recall; these bounds make the signal discriminative.
W_IMPORTED_BY = 1.5          # per relevant importer (after the ubiquity gate)
IMPORTED_BY_UBIQUITY_MAX = 3  # more relevant importers than this ⇒ hub, not boosted
_EDGE_IMPORTER_LIMIT = 15    # only edges from the N most-relevant files count

# Architectural role hints keyed by path segment, split by which *side* of the
# stack the segment signals. Splitting matters because role weight is the largest
# path-only signal: in a full-stack repo a backend request and a frontend request
# used to share one backend-only vocabulary, so backend files earned a structural
# bump frontend files structurally could not — even when the request named the
# frontend. `request_intent` (see below) lets us modulate these by the request's
# lean instead of competing flat.
_BACKEND_ROLES: dict[str, float] = {
    "routes": 3.0, "route": 3.0, "router": 3.0, "controllers": 3.0,
    "controller": 3.0, "handlers": 3.0, "handler": 3.0, "endpoints": 3.0,
    "endpoint": 3.0, "api": 2.5,
    "services": 2.5, "service": 2.5, "usecases": 2.5, "use_cases": 2.5,
    "domain": 2.0, "models": 2.5, "model": 2.5, "schemas": 2.5, "schema": 2.5,
    "migrations": 2.5, "migration": 2.5, "entities": 2.0, "middleware": 2.0,
    "auth": 2.5,
}
_FRONTEND_ROLES: dict[str, float] = {
    "components": 2.5, "component": 2.5, "pages": 2.5, "page": 2.5,
    "screens": 2.5, "screen": 2.5, "templates": 2.5, "template": 2.5,
    "hooks": 2.0, "composables": 2.0, "features": 2.0, "static": 2.0,
    "ui": 2.0, "widgets": 2.0, "widget": 2.0, "store": 2.0, "stores": 2.0,
    "assets": 1.5, "styles": 1.5, "style": 1.5,
}
# Roles that are genuinely ambiguous across the stack apply at full weight
# regardless of intent (never modulated). `views`/`view` is Django backend
# (`views.py`) but Vue/Nuxt frontend (`views/`); `core` is side-agnostic.
_NEUTRAL_ROLES: dict[str, float] = {
    "views": 2.5, "view": 2.5, "core": 1.5,
}
# When a request's intent opposes a (sided) role, scale that role down rather than
# zeroing it — a frontend task that genuinely touches an API file can still
# surface it via keyword/symbol signals. Frontend roles are gated *on* only by a
# frontend-leaning request, which keeps the no-intent case byte-for-byte today's.
ROLE_OPPOSING_INTENT_FACTOR = 0.4

_TEST_REQUEST_TOKENS = {"test", "tests", "testing", "coverage", "regression"}

# How many files to read content for, per scale mode.
_STAGE2_LIMIT = {"normal": 50, "medium": 35, "large": 20}


def _mode_cap(config: Config, mode: str) -> int:
    base = config.analysis.max_ranked_files
    if mode == "large":
        return min(base, 6)
    if mode == "medium":
        return min(base, 10)
    return base


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _named_file_hit(path: str, named_files: set[str]) -> bool:
    """True if the request explicitly named this file — by bare basename or by a
    path suffix, so ``static/css/style.css`` in the request matches the candidate
    even when the repo is rooted elsewhere. ``named_files`` is already lowercased."""

    lower = path.lower()
    base = _basename(lower)
    for mention in named_files:
        if "/" in mention:
            if lower == mention or lower.endswith("/" + mention):
                return True
        elif base == mention:
            return True
    return False


def _role_score(segments: list[str], intent: str, use_intent_roles: bool) -> float:
    """Sum architectural role weights for a path's segments, modulated by intent.

    With ``use_intent_roles`` off (or intent ``"none"``), this reduces exactly to
    the legacy backend-only behaviour: backend + neutral roles at full weight,
    frontend roles silent. A frontend-leaning request activates frontend roles and
    scales backend roles down; a backend-leaning request scales frontend roles
    down. Neutral roles are never modulated.
    """

    total = 0.0
    for seg in segments:
        seg = seg.lower()
        if seg in _NEUTRAL_ROLES:
            total += _NEUTRAL_ROLES[seg]
        elif seg in _BACKEND_ROLES:
            weight = _BACKEND_ROLES[seg]
            if use_intent_roles and intent == "frontend":
                weight *= ROLE_OPPOSING_INTENT_FACTOR
            total += weight
        elif seg in _FRONTEND_ROLES:
            if not use_intent_roles:
                continue  # legacy parity: frontend vocabulary did not exist
            if intent == "frontend":
                total += _FRONTEND_ROLES[seg]
            elif intent == "backend":
                total += _FRONTEND_ROLES[seg] * ROLE_OPPOSING_INTENT_FACTOR
            # intent "none": stay silent, preserving today's ranking exactly.
    return total


def _stage1_signals(
    path: str,
    keywords: set[str],
    named_files: set[str],
    summary: RepoSummary,
    request_is_testy: bool,
    intent: str,
    use_intent_roles: bool,
) -> dict[str, float]:
    """Path/structure-only signals — no file read."""

    signals: dict[str, float] = {}
    segments = path.split("/")
    name = segments[-1]
    name_tokens = set(tokenize(name))
    path_tokens = set(tokenize("/".join(segments[:-1])))

    filename_hits = len(keywords & name_tokens)
    if filename_hits:
        signals["filename"] = filename_hits * W_FILENAME_HIT
    path_hits = len(keywords & path_tokens)
    if path_hits:
        signals["path"] = path_hits * W_PATH_HIT

    # A file the request named outright is a near-direct hit: surface it
    # regardless of role/symbol competition, and (because this lands in stage 1)
    # pull it into the stage-2 read set so it can earn content/symbol signals too.
    if named_files and _named_file_hit(path, named_files):
        signals["named_file"] = W_NAMED_FILE

    role = _role_score(segments, intent, use_intent_roles)
    if role:
        signals["role"] = role * W_ROLE

    if path in summary.entry_points:
        signals["entry_point"] = W_ENTRY
    if _basename(path) in {_basename(m) for m in summary.manifests}:
        signals["manifest"] = W_MANIFEST

    is_test = path in summary.test_paths or any(
        path.startswith(t + "/") or path == t for t in summary.test_paths
    )
    if is_test and request_is_testy:
        signals["test"] = W_TEST

    # Shallow files get a small nudge (entry points and core config live near root).
    depth = path.count("/")
    if depth <= 1:
        signals["proximity"] = ROOT_PROXIMITY

    return signals


def _stage2_signals(rel_path: str, text: str, keywords: set[str]) -> dict[str, float]:
    """Content-keyword and symbol-overlap signals from already-read *text*."""

    signals: dict[str, float] = {}

    content_tokens = tokenize(text)
    hits = sum(1 for tok in content_tokens if tok in keywords)
    if hits:
        signals["content"] = min(hits, CONTENT_HIT_CAP) * W_CONTENT_HIT

    # Only score symbols in actual source files — not code blocks inside docs.
    # HTML templates and CSS aren't "code" by the symbol regex, but their block/
    # component names and class/id/custom-property names are real, request-
    # nameable symbols, so each "non-code" frontend language gets its own pattern.
    symbol_tokens: set[str] = set()
    if is_code(rel_path):
        for match in SYMBOL_RE.finditer(text):
            symbol_tokens.update(split_identifier(match.group(1)))
    elif detect.language_of(rel_path) == "HTML":
        for block, tag in TEMPLATE_SYMBOL_RE.findall(text):
            symbol_tokens.update(split_identifier(block or tag))
    elif detect.language_of(rel_path) == "CSS":
        for selector, prop in STYLE_SYMBOL_RE.findall(text):
            symbol_tokens.update(split_identifier(selector or prop))

    symbol_overlap = len(keywords & symbol_tokens)
    if symbol_overlap:
        signals["symbol"] = symbol_overlap * W_SYMBOL

    return signals


def rank_files(
    request: str,
    scan: RepoScan,
    summary: RepoSummary,
    config: Config,
    *,
    named_files: set[str] | None = None,
) -> list[RankedFile]:
    """Return the top relevant files for *request*, scored and explained.

    *named_files* are files the request named explicitly (see
    ``extract_named_files``); they are extracted from the *full* request by the
    caller and passed in, because ``request`` here is the focused ranking signal
    that may have truncated the file mention away.
    """

    keywords = extract_keywords(request)
    if not keywords:
        return []

    request_is_testy = bool(keywords & _TEST_REQUEST_TOKENS)
    use_intent_roles = config.analysis.use_intent_roles
    # A clear frontend/backend lean steers role weights toward that side; an
    # ambiguous request stays "none" and ranks exactly as it does today.
    intent = request_intent(keywords) if use_intent_roles else "none"
    # The named-file boost is toggleable so `eval --ablation` can isolate it.
    named = named_files or set()
    if not config.analysis.use_named_file_boost:
        named = set()
    root = Path(scan.root)

    # Stage 1: cheap path-based scoring over *every* candidate. We keep files
    # with a zero path score too — many relevant files (e.g. a config module
    # that only mentions the keyword in its body) match on content alone, so
    # dropping them here would prevent them from ever being read in stage 2.
    stage1: list[RankedFile] = []
    for f in scan.files:
        signals = _stage1_signals(
            f.path, keywords, named, summary, request_is_testy, intent, use_intent_roles
        )
        stage1.append(RankedFile(path=f.path, score=sum(signals.values()), signals=signals))

    stage1.sort(key=lambda r: (-r.score, r.path))

    # Stage 2: read content for the strongest path-level candidates. On large
    # repos this bounds I/O; content-only files past the cutoff keep their
    # (path-only) score, which is the documented large-repo tradeoff. We read
    # each file once and reuse the text for both content signals and the
    # import-edge tally below.
    use_import_edge = config.analysis.use_import_edge
    all_paths, by_basename = _build_path_index(scan.files)
    ranked_by_path = {r.path: r for r in stage1}
    import_counts: dict[str, int] = {}

    stage2_limit = _STAGE2_LIMIT.get(summary.mode, 50)
    for importer_rank, ranked in enumerate(stage1[:stage2_limit]):
        text = _read_head(root / ranked.path, config.repo.max_file_lines)
        if text is None:
            continue
        extra = _stage2_signals(ranked.path, text, keywords)
        if extra:
            ranked.signals.update(extra)
            ranked.score += sum(extra.values())
        # Only edges from the most-relevant files inform centrality; a far-down
        # candidate importing a file says little about that file's relevance.
        if use_import_edge and importer_rank < _EDGE_IMPORTER_LIMIT:
            for ref in _referenced_paths(text, ranked.path, all_paths, by_basename):
                import_counts[ref] = import_counts.get(ref, 0) + 1

    # Apply the import-edge boost, but only to files imported by a *handful* of
    # relevant files. A file imported by more than IMPORTED_BY_UBIQUITY_MAX of
    # them is a ubiquitous utility, and boosting it evicts the files a feature
    # change actually touches (the eval harness measured this regression). This
    # is a lightweight reference scan, not an AST. Toggleable for `eval --ablation`.
    for path, count in import_counts.items():
        if count > IMPORTED_BY_UBIQUITY_MAX:
            continue
        target = ranked_by_path.get(path)
        if target is None:
            continue
        boost = count * W_IMPORTED_BY
        target.signals["imported_by"] = boost
        target.score += boost

    # Now drop files with no signal at all and re-rank.
    stage1 = [r for r in stage1 if r.score > 0]
    stage1.sort(key=lambda r: (-r.score, r.path))

    # Diversity penalty + final cut: discourage one *folder* from dominating.
    # Key on the immediate parent directory, not the top-level segment — in a
    # src/-layout repo the whole source tree shares one top segment, so keying
    # on it would wrongly penalize all real code relative to scattered root docs.
    cap = _mode_cap(config, summary.mode)
    selected: list[RankedFile] = []
    dir_counts: dict[str, int] = {}
    for ranked in stage1:
        parent = ranked.path.rsplit("/", 1)[0] if "/" in ranked.path else ""
        penalty = min(dir_counts.get(parent, 0), DIVERSITY_PENALTY_CAP) * DIVERSITY_PENALTY
        if penalty:
            ranked.signals["diversity_penalty"] = -penalty
            ranked.score -= penalty
        selected.append(ranked)
        dir_counts[parent] = dir_counts.get(parent, 0) + 1

    selected.sort(key=lambda r: (-r.score, r.path))
    return selected[:cap]
