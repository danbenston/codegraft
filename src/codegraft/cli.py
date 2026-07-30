"""codegraft command-line interface.

Phase 1 surface:
- ``init``    write a default codegraft.toml and report env-key requirements
- ``inspect`` placeholder; the real deterministic-analysis preview lands in Phase 3
- ``plan``    run the (stub) end-to-end path and write a Markdown plan to /plans
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

from codegraft import __version__

# Rich + the legacy Windows console default to the active code page (often
# cp1252), which crashes on non-ASCII output — including arbitrary model-
# generated plan text printed via --stdout. Force UTF-8 with a replacement
# fallback so output never raises a UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
from codegraft.branding import BANNER, TAGLINE
from codegraft.config import CONFIG_FILENAME, DEFAULT_CONFIG_TOML, Config
from codegraft.errors import CodegraftError
from codegraft.planning.filenames import plan_stem
from codegraft.planning.renderer import render_markdown
from codegraft.planning.service import build_stub_plan

app = typer.Typer(
    name="codegraft",
    help=f"{TAGLINE}",
    no_args_is_help=True,
    add_completion=True,
)
# legacy_windows=False keeps Rich on the standard write path (honouring the
# UTF-8 reconfigure above) instead of the win32 console API that encodes with
# the system code page.
console = Console(legacy_windows=False)
err_console = Console(stderr=True, legacy_windows=False)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"codegraft {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """codegraft — implementation plans for coding agents."""


@app.command()
def init(
    repo: Path = typer.Option(
        Path("."), "--repo", help="Repository root to initialize.",
        file_okay=False, dir_okay=True, resolve_path=True,
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
) -> None:
    """Create a ``codegraft.toml`` and print required environment keys."""

    console.print(BANNER)
    config_path = repo / CONFIG_FILENAME
    if config_path.exists() and not force:
        err_console.print(
            f"[yellow]{CONFIG_FILENAME} already exists[/yellow] at {config_path}. "
            "Use --force to overwrite."
        )
        raise typer.Exit(code=1)

    config_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    console.print(f"[green]Wrote[/green] {config_path}")
    console.print(
        "\nSet provider keys in your environment or a [bold].env[/bold] file:\n"
        "  [dim]ANTHROPIC_API_KEY=...[/dim]\n"
        "  [dim]OPENAI_API_KEY=...[/dim]\n"
    )


@app.command()
def inspect(
    request: str = typer.Argument(
        None, help="Feature request to rank files for. Omit to show summary only."
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Repository root to analyze.",
        file_okay=False, dir_okay=True, resolve_path=True,
    ),
    request_file: Path | None = typer.Option(
        None, "--request-file", help="Read the feature request from a file.",
        exists=True, dir_okay=False, resolve_path=True,
    ),
    subdir: str | None = typer.Option(
        None, "--subdir", help="Scope analysis to a repo-relative subdirectory."
    ),
    snippets: bool = typer.Option(
        False, "--snippets", help="Also print the extracted context snippets."
    ),
    max_files: int | None = typer.Option(
        None, "--max-files", help="Cap ranked files (and snippet candidates) to top N."
    ),
    max_snippet_lines: int | None = typer.Option(
        None, "--max-snippet-lines", help="Cap lines per extracted snippet."
    ),
) -> None:
    """Preview what codegraft sees and ranks for a request (no model call).

    With a request, this is the ranking/tuning surface: repo summary, the
    deterministically-ranked files with their score breakdown, and the bounded
    context snippets that would be sent to a provider. Pair with --request-file
    to preview ranking for a long-form request before spending an API call.
    The --max-files / --max-snippet-lines caps mirror the select_context MCP
    knobs, so you can preview a slimmer bundle.
    """

    from rich.table import Table

    from codegraft.repo.analyze import analyze_repo

    try:
        request_text = _read_request(request, request_file, required=False)
        config = Config.load(repo)
        if max_files is not None:
            config.analysis.max_ranked_files = max(1, max_files)
        if max_snippet_lines is not None:
            config.analysis.max_snippet_lines_per_file = max(1, max_snippet_lines)
        analysis = analyze_repo(request_text or "", repo, config, subdir=subdir)
    except CodegraftError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)
    request = request_text  # may be None for summary-only mode

    scan, summary = analysis.scan, analysis.summary
    console.print(BANNER)

    source = "git ls-files" if scan.used_git else "filesystem walk (no git)"
    scope = f"  Subdir: [bold]{subdir}[/bold]" if subdir else ""
    console.print(
        f"\nRepo: {scan.root}{scope}\n"
        f"Discovery: [bold]{source}[/bold]  Kept: [green]{scan.file_count}[/green]  "
        f"Skipped: [yellow]{len(scan.skipped)}[/yellow]  "
        f"Mode: [bold]{summary.mode}[/bold]"
        + ("  [red](truncated)[/red]" if scan.truncated else "")
    )
    if summary.mode == "large":
        console.print(
            "[yellow]Large repo[/yellow] — ranking is conservative; "
            "consider scoping with [bold]--subdir[/bold]."
        )

    lang_mix = "  ".join(f"{lang}={n}" for lang, n in list(summary.languages.items())[:6])
    console.print(
        f"[dim]Primary:[/dim] {summary.primary_language or '—'}   "
        f"[dim]Frameworks:[/dim] {', '.join(summary.frameworks) or '—'}\n"
        f"[dim]Languages:[/dim] {lang_mix or '—'}\n"
        f"[dim]Manifests:[/dim] {', '.join(summary.manifests[:6]) or '—'}\n"
        f"[dim]Entry points:[/dim] {', '.join(summary.entry_points[:6]) or '—'}"
    )

    if not request:
        console.print(
            "\n[dim]Pass a feature request to see ranked files, e.g.:[/dim]\n"
            '  codegraft inspect "add role-based access control"'
        )
        return

    # A long request is ranked by a focused signal — show it so the user knows
    # what actually drove file selection.
    if analysis.ranking_signal.strip() != (request or "").strip():
        console.print(
            f"[dim]Ranking signal:[/dim] [italic]{analysis.ranking_signal}[/italic] "
            "[dim](full request still goes to the model)[/dim]"
        )

    if not analysis.ranked:
        console.print(
            f"\n[yellow]No files ranked[/yellow] for: [italic]{analysis.ranking_signal}[/italic]\n"
            "[dim]Try more specific keywords, or widen --subdir.[/dim]"
        )
        return

    flat_req = " ".join(request.split())
    title_req = flat_req if len(flat_req) <= 60 else flat_req[:57] + "..."
    table = Table(title=f"Ranked files for: {title_req}")
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Path", overflow="fold")
    table.add_column("Signals", overflow="fold")
    for i, r in enumerate(analysis.ranked, 1):
        signals = "  ".join(
            f"{k}={v:+.1f}" for k, v in sorted(r.signals.items(), key=lambda kv: -abs(kv[1]))
        )
        table.add_row(str(i), f"{r.score:.1f}", r.path, signals)
    console.print(table)
    console.print(
        f"[dim]Context bundle:[/dim] {len(analysis.snippets)} snippets, "
        f"{analysis.context_chars} chars "
        f"(budget {config.analysis.context_char_budget})"
    )
    est = analysis.token_estimate()
    console.print(f"[dim]Est. tokens (rough):[/dim] {est.summary()}")

    if snippets:
        for s in analysis.snippets:
            console.print(f"\n[bold cyan]{s.path}[/bold cyan] "
                          f"[dim]({s.line_count} lines"
                          + (", truncated" if s.truncated else "") + ")[/dim]")
            console.print(s.content)


@app.command()
def impact(
    target: str = typer.Argument(
        ..., help="File to find dependents of (repo-relative path or bare filename)."
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Repository root to analyze.",
        file_okay=False, dir_okay=True, resolve_path=True,
    ),
    transitive: bool = typer.Option(
        False, "--transitive", help="Also show the full transitive reverse closure."
    ),
    subdir: str | None = typer.Option(
        None, "--subdir", help="Scope analysis to a repo-relative subdirectory."
    ),
) -> None:
    """Show what depends on a file — blast-radius triage before you edit it.

    Inverts codegraft's heuristic import graph: lists the files that import
    `target`. Resolution is heuristic (misses dynamic imports, re-exports, and
    ambiguous basenames), so treat it as triage, not a correctness oracle.
    """

    from codegraft.repo.discover import discover_repo
    from codegraft.repo.graph import impact_of

    try:
        config = Config.load(repo)
        scan = discover_repo(repo, config, subdir=subdir)
        result = impact_of(target, scan, repo, config, transitive=transitive)
    except CodegraftError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(BANNER)
    if not result.found:
        if result.ambiguous:
            console.print(
                f"[yellow]Ambiguous target[/yellow] [bold]{target}[/bold] — "
                "matches several files:"
            )
            for cand in result.ambiguous:
                console.print(f"  {cand}")
            console.print("[dim]Re-run with a path-qualified target.[/dim]")
            raise typer.Exit(code=1)
        console.print(
            f"[yellow]No candidate file[/yellow] matched [bold]{target}[/bold].\n"
            "[dim]Pass a repo-relative path or an exact filename.[/dim]"
        )
        raise typer.Exit(code=1)

    console.print(
        f"\nImpact of [bold cyan]{result.resolved}[/bold cyan]  "
        f"[dim](resolution: {result.resolution})[/dim]"
    )
    if not result.imported_by:
        console.print(
            "[dim]No files import this (heuristically) — looks like a leaf.[/dim]"
        )
    else:
        console.print(
            f"[dim]Directly imported by ({len(result.imported_by)}):[/dim]"
        )
        for path in result.imported_by:
            console.print(f"  {path}")
    if result.via_reexport:
        console.print(
            f"[dim]Reaches it through a barrel re-export ({len(result.via_reexport)}):[/dim]"
        )
        for path in result.via_reexport:
            console.print(f"  {path}")
    if transitive and result.transitive:
        console.print(
            f"[dim]Transitively reaches ({len(result.transitive)} more):[/dim]"
        )
        for path in result.transitive:
            console.print(f"  [dim]{path}[/dim]")
    console.print(
        "\n[dim]Heuristic reference scan — misses dynamic imports. "
        "Blast-radius triage, not a guarantee.[/dim]"
    )


@app.command()
def symbol(
    name: str = typer.Argument(..., help="Symbol name to locate (function/class/selector)."),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Repository root to analyze.",
        file_okay=False, dir_okay=True, resolve_path=True,
    ),
    in_path: str | None = typer.Option(
        None, "--in", help="Restrict the search to one repo-relative file."
    ),
    subdir: str | None = typer.Option(
        None, "--subdir", help="Scope analysis to a repo-relative subdirectory."
    ),
) -> None:
    """Fetch one symbol definition instead of reading the whole file.

    Locates where `name` is defined and prints its signature, body, and span.
    Location and extent are heuristic (regex + block balancing, not an LSP), so
    this is "good enough to avoid a full-file read," not go-to-definition.
    """

    from codegraft.repo.discover import discover_repo
    from codegraft.repo.symbols import find_symbol

    try:
        config = Config.load(repo)
        scan = discover_repo(repo, config, subdir=subdir)
        hits = find_symbol(name, scan, repo, config, in_path=in_path)
    except CodegraftError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(BANNER)
    if not hits:
        scope = f" in [bold]{in_path}[/bold]" if in_path else ""
        console.print(
            f"\n[yellow]No definition of[/yellow] [bold]{name}[/bold]{scope} "
            "(heuristically).\n"
            "[dim]Try the exact identifier, or drop --in to search all files.[/dim]"
        )
        raise typer.Exit(code=1)

    if len(hits) > 1:
        console.print(
            f"[dim]{len(hits)} definitions of[/dim] [bold]{name}[/bold] "
            "[dim](returning all — disambiguate with --in):[/dim]"
        )
    for h in hits:
        trunc = " [yellow](truncated)[/yellow]" if h.truncated else ""
        console.print(
            f"\n[bold cyan]{h.path}[/bold cyan]:{h.span[0]}-{h.span[1]}  "
            f"[dim](resolution: {h.resolution})[/dim]{trunc}"
        )
        console.print(h.snippet)
    console.print(
        "\n[dim]Heuristic location/extent — not go-to-definition. "
        "Good enough to skip a full-file read.[/dim]"
    )


@app.command("affected-tests")
def affected_tests_cmd(
    files: list[str] = typer.Argument(
        None, help="Changed files to find dependent tests for."
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Repository root to analyze.",
        file_okay=False, dir_okay=True, resolve_path=True,
    ),
    since: str | None = typer.Option(
        None, "--since", help="Derive the changed set from `git diff --name-only <ref>`."
    ),
    subdir: str | None = typer.Option(
        None, "--subdir", help="Scope analysis to a repo-relative subdirectory."
    ),
) -> None:
    """Select the tests that depend on changed files (selection only — never runs them).

    Reverse-reachable closure of the changed files over the import graph,
    intersected with the repo's test files. Heuristic: false negatives are
    possible, so this speeds the inner loop but never replaces a full-suite run
    before merge. Running the selected tests is a separate tool's job.
    """

    from codegraft.repo.discover import discover_repo
    from codegraft.repo.graph import affected_tests
    from codegraft.repo.summarize import summarize

    try:
        config = Config.load(repo)
        changed = list(files or [])
        if since:
            changed.extend(_changed_since(repo, since))
        if not changed:
            raise CodegraftError(
                "no changed files given — pass files or --since <ref>"
            )
        scan = discover_repo(repo, config, subdir=subdir)
        summary = summarize(scan)
        result = affected_tests(changed, scan, summary, repo, config)
    except CodegraftError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(BANNER)
    console.print(
        f"\nAffected tests for {len(result.changed)} changed file(s)  "
        f"[dim](completeness: {result.completeness})[/dim]"
    )
    if result.unresolved:
        console.print(
            f"[yellow]Not found in scan[/yellow] ({len(result.unresolved)}): "
            + ", ".join(result.unresolved)
        )
    if not result.tests:
        console.print("[dim]No dependent tests found (heuristically).[/dim]")
    else:
        for path in result.tests:
            console.print(f"  {path}")
    console.print(
        "\n[dim]Heuristic — can miss tests. A fast-feedback hint, not a "
        "replacement for the full suite before merge. codegraft does not run them.[/dim]"
    )


@app.command()
def plan(
    request: str = typer.Argument(
        None, help="Feature request. Omit when using --request-file."
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Repository root to plan against.",
        file_okay=False, dir_okay=True, resolve_path=True,
    ),
    request_file: Path | None = typer.Option(
        None, "--request-file", help="Read the feature request from a file.",
        exists=True, dir_okay=False, resolve_path=True,
    ),
    subdir: str | None = typer.Option(
        None, "--subdir", help="Scope analysis to a repo-relative subdirectory."
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="Override provider (anthropic|openai)."
    ),
    model: str | None = typer.Option(None, "--model", help="Override model id."),
    stub: bool = typer.Option(
        False, "--stub", help="Offline placeholder plan; no repo analysis, no API call."
    ),
    stdout: bool = typer.Option(
        False, "--stdout", help="Print the plan to stdout instead of a file."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the plan as JSON to stdout (for scripting)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Build the plan but do not write any file."
    ),
) -> None:
    """Generate an implementation plan and write it to the output directory.

    Calls the configured provider (default: anthropic) to produce a real,
    evidence-backed plan. Use --stub to produce an offline placeholder without an
    API key.
    """

    from codegraft.planning.service import generate_plan

    try:
        request_text = _read_request(request, request_file, required=True)
        config = Config.load(repo)
        if provider:
            config.provider.name = provider
        if model:
            config.provider.model = model

        if stub:
            plan_obj = build_stub_plan(request_text, repo, config)
        else:
            with console.status(
                f"Analyzing repo and planning with "
                f"{config.provider.name}/{config.provider.model}..."
            ):
                plan_obj = generate_plan(request_text, repo, config, subdir=subdir)

        if as_json:
            # Plain echo so Rich markup/reflow can't corrupt the JSON.
            typer.echo(plan_obj.model_dump_json(indent=2))
            return

        markdown = render_markdown(plan_obj)
        if stdout:
            console.print(markdown)
            return

        stem = plan_stem(request_text)
        out_path = _write_plan(repo, config.output.output_dir, stem, markdown, dry_run)
        if dry_run:
            console.print(
                f"[cyan]--dry-run[/cyan]: plan built ({len(markdown)} chars), not written."
            )
            return

        console.print(f"[green]Wrote plan[/green] -> {out_path}")
        if config.output.write_debug_json and not stub:
            debug_path = _write_debug_json(repo, config.output.output_dir, stem, plan_obj)
            console.print(f"[dim]Debug plan JSON -> {debug_path}[/dim]")
    except CodegraftError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("eval")
def evaluate(
    commits: list[str] | None = typer.Argument(
        None, help="Commit SHAs to evaluate. Omit to use recent history."
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Repository root to evaluate.",
        file_okay=False, dir_okay=True, resolve_path=True,
    ),
    last: int = typer.Option(
        0, "--last", help="Evaluate the most recent N commits (default 10 if no SHAs)."
    ),
    k: int = typer.Option(10, "--k", help="Rank cutoff for recall@k."),
    no_import_edge: bool = typer.Option(
        False, "--no-import-edge", help="Disable the import-edge ranking signal."
    ),
    no_intent_roles: bool = typer.Option(
        False, "--no-intent-roles",
        help="Disable intent-modulated role weights (frontend/backend lean).",
    ),
    no_named_file: bool = typer.Option(
        False, "--no-named-file",
        help="Disable the explicit named-file boost (e.g. 'port style.css').",
    ),
    ablation: bool = typer.Option(
        False, "--ablation",
        help="Score with and without the import-edge signal and show the delta.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Score ranking accuracy against real git history (recall@k, MRR). No model call.

    Each commit's subject becomes a request and its changed source files become
    the gold set; this measures how well `inspect` would have surfaced them.
    """

    import subprocess

    from codegraft.evaluation import resolve_commits, run_eval

    try:
        config = Config.load(repo)
        shas = resolve_commits(repo, commits, last)
        if not shas:
            raise CodegraftError("no commits to evaluate")
    except CodegraftError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)
    except (subprocess.SubprocessError, OSError) as exc:
        err_console.print(
            f"[red]error:[/red] could not read git history "
            f"(is {repo} a git repository?): {exc}"
        )
        raise typer.Exit(code=1)

    try:
        if ablation:
            on = run_eval(
                repo, config, shas, k=k,
                use_import_edge=True, use_intent_roles=not no_intent_roles,
                use_named_file_boost=not no_named_file,
            )
            off = run_eval(
                repo, config, shas, k=k,
                use_import_edge=False, use_intent_roles=not no_intent_roles,
                use_named_file_boost=not no_named_file,
            )
        else:
            report = run_eval(
                repo, config, shas, k=k,
                use_import_edge=not no_import_edge,
                use_intent_roles=not no_intent_roles,
                use_named_file_boost=not no_named_file,
            )
    except (subprocess.SubprocessError, OSError) as exc:
        err_console.print(f"[red]error:[/red] git failed while evaluating: {exc}")
        raise typer.Exit(code=1)

    if ablation:
        if as_json:
            typer.echo(_eval_json(on, off))
            return
        _print_eval_report(on)
        _print_ablation_delta(on, off)
    else:
        if as_json:
            typer.echo(_eval_json(report))
            return
        _print_eval_report(report)


@app.command("eval-impact")
def evaluate_impact(
    commits: list[str] | None = typer.Argument(
        None, help="Commit SHAs to evaluate. Omit to use recent history."
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Repository root to evaluate.",
        file_okay=False, dir_okay=True, resolve_path=True,
    ),
    last: int = typer.Option(
        0, "--last", help="Evaluate the most recent N commits (default 20 if no SHAs)."
    ),
    direct: bool = typer.Option(
        False, "--direct",
        help="Use only direct importers instead of the transitive reverse closure.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Measure impact_of as a co-change predictor against git history. No model call.

    For each commit, scores the fraction of its co-changed file *pairs* that the
    import graph links — i.e. how often `impact_of` on one changed file would have
    surfaced another. This is the baseline value of the blast-radius query.
    """

    import subprocess

    from codegraft.evaluation import resolve_commits
    from codegraft.evaluation_graph import run_impact_eval

    try:
        config = Config.load(repo)
        shas = resolve_commits(repo, commits, last if last > 0 else 20)
        if not shas:
            raise CodegraftError("no commits to evaluate")
        report = run_impact_eval(repo, config, shas, transitive=not direct)
    except CodegraftError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)
    except (subprocess.SubprocessError, OSError) as exc:
        err_console.print(
            f"[red]error:[/red] could not read git history "
            f"(is {repo} a git repository?): {exc}"
        )
        raise typer.Exit(code=1)

    if as_json:
        typer.echo(_impact_eval_json(report))
        return
    _print_impact_report(report)


@app.command("eval-symbol")
def evaluate_symbol(
    repo: Path = typer.Option(
        Path("."), "--repo", help="Repository root to evaluate.",
        file_okay=False, dir_okay=True, resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Measure the read-side token savings get_symbol delivers. No git, no model.

    Locates every definition in the repo and reports the mean fraction of a file
    you avoid reading by fetching one symbol — the baseline value of get_symbol —
    plus how widely it's applicable (share of files with locatable symbols).
    """

    from codegraft.evaluation_symbol import run_symbol_eval

    try:
        config = Config.load(repo)
        report = run_symbol_eval(repo, config)
    except CodegraftError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)

    if as_json:
        import json
        from dataclasses import asdict

        payload = asdict(report)
        payload["coverage_pct"] = report.coverage_pct
        typer.echo(json.dumps(payload, indent=2))
        return

    console.print(BANNER)
    console.print(
        f"\n[bold]get_symbol value baseline[/bold]  [dim](repo: {repo})[/dim]\n"
        f"  Files with locatable symbols: "
        f"[green]{report.files_with_symbols}[/green]/{report.files_total} "
        f"([cyan]{report.coverage_pct:.0f}%[/cyan] coverage)\n"
        f"  Symbols located: [green]{report.symbols}[/green]  "
        f"([yellow]{report.truncated}[/yellow] hit the line cap)\n"
        f"  Median symbol span: [bold]{report.median_span_lines:.0f}[/bold] lines  "
        f"vs median file: [bold]{report.median_file_lines:.0f}[/bold] lines\n"
        f"  [bold]Mean read savings: [green]{report.mean_savings_pct:.1f}%[/green][/bold] "
        "[dim](fraction of the file you skip by fetching one symbol)[/dim]"
    )


@app.command("eval-tests")
def evaluate_tests(
    commits: list[str] | None = typer.Argument(
        None, help="Commit SHAs to evaluate. Omit to use recent history."
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Repository root to evaluate.",
        file_okay=False, dir_okay=True, resolve_path=True,
    ),
    last: int = typer.Option(
        0, "--last", help="Evaluate the most recent N commits (default 30 if no SHAs)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Measure affected_tests selection recall against git history. No model call.

    For commits that changed source *and* tests, scores the fraction of changed
    tests that `affected_tests` on the changed source would have selected — the
    false-negative risk made measurable.
    """

    import subprocess

    from codegraft.evaluation import resolve_commits
    from codegraft.evaluation_graph import run_tests_eval

    try:
        config = Config.load(repo)
        shas = resolve_commits(repo, commits, last if last > 0 else 30)
        if not shas:
            raise CodegraftError("no commits to evaluate")
        report = run_tests_eval(repo, config, shas)
    except CodegraftError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)
    except (subprocess.SubprocessError, OSError) as exc:
        err_console.print(
            f"[red]error:[/red] could not read git history "
            f"(is {repo} a git repository?): {exc}"
        )
        raise typer.Exit(code=1)

    if as_json:
        import json
        from dataclasses import asdict

        typer.echo(json.dumps(
            {
                "mean_recall": report.mean_recall,
                "micro_recall": report.micro_recall,
                "scored": len(report.scored),
                "commits": len(report.cases),
                "cases": [asdict(c) for c in report.cases],
            },
            indent=2,
        ))
        return
    _print_tests_report(report)


def _print_tests_report(report) -> None:
    """Render a TestSelectReport: per-commit test-selection recall plus aggregates."""

    from rich.table import Table

    console.print(BANNER)
    table = Table(title="affected_tests selection recall")
    table.add_column("#", justify="right")
    table.add_column("Commit")
    table.add_column("Selected/Gold", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Subject", overflow="fold")
    for i, c in enumerate(report.cases, 1):
        if not c.evaluable:
            why = "no test change" if c.gold_tests == 0 else "no source change"
            table.add_row(str(i), c.sha[:7], "[dim]—[/dim]", "[dim]—[/dim]",
                          f"[dim]{c.subject}  ({why})[/dim]")
            continue
        table.add_row(str(i), c.sha[:7], f"{c.selected}/{c.gold_tests}",
                      f"{c.recall:.2f}", c.subject)
    console.print(table)

    scored = report.scored
    console.print(
        f"[bold]{len(scored)}[/bold] scored / {len(report.cases)} commits   "
        f"[dim]macro[/dim] recall=[green]{report.mean_recall:.3f}[/green]   "
        f"[dim]micro[/dim] recall=[green]{report.micro_recall:.3f}[/green]"
    )
    console.print(
        "[dim]Share of co-changed tests affected_tests would have selected. "
        "Below 1.0 is the false-negative risk — never skip the full suite at merge.[/dim]"
    )


def _print_impact_report(report) -> None:
    """Render an ImpactReport: per-commit pair-linkage plus aggregates."""

    from rich.table import Table

    console.print(BANNER)
    mode = "transitive" if report.transitive else "direct"
    table = Table(title=f"impact_of co-change linkage  ({mode})")
    table.add_column("#", justify="right")
    table.add_column("Commit")
    table.add_column("Linked/Pairs", justify="right")
    table.add_column("Pair-recall", justify="right")
    table.add_column("Subject", overflow="fold")
    for i, c in enumerate(report.cases, 1):
        if not c.evaluable:
            table.add_row(str(i), c.sha[:7], "0/0", "[dim]—[/dim]",
                          f"[dim]{c.subject}  (<2 candidate files)[/dim]")
            continue
        table.add_row(str(i), c.sha[:7], f"{c.linked_pairs}/{c.pairs}",
                      f"{c.pair_recall:.2f}", c.subject)
    console.print(table)

    scored = report.scored
    console.print(
        f"[bold]{len(scored)}[/bold] scored / {len(report.cases)} commits   "
        f"[dim]macro[/dim] pair-recall=[green]{report.mean_pair_recall:.3f}[/green]   "
        f"[dim]micro[/dim] pair-recall=[green]{report.micro_pair_recall:.3f}[/green]"
    )
    console.print(
        "[dim]Fraction of co-changed file pairs the import graph links — "
        "the share of blast radius impact_of would have surfaced.[/dim]"
    )


def _impact_eval_json(report) -> str:
    import json
    from dataclasses import asdict

    return json.dumps(
        {
            "transitive": report.transitive,
            "mean_pair_recall": report.mean_pair_recall,
            "micro_pair_recall": report.micro_pair_recall,
            "scored": len(report.scored),
            "commits": len(report.cases),
            "cases": [asdict(c) for c in report.cases],
        },
        indent=2,
    )


def _print_eval_report(report) -> None:
    """Render an EvalReport as a per-commit table plus aggregates."""

    from rich.table import Table

    console.print(BANNER)
    edge = "on" if report.use_import_edge else "off"
    intent = "on" if report.use_intent_roles else "off"
    named = "on" if report.use_named_file_boost else "off"
    table = Table(
        title=f"Ranking eval  (k={report.k}, import-edge {edge}, "
        f"intent-roles {intent}, named-file {named})"
    )
    table.add_column("#", justify="right")
    table.add_column("Commit")
    table.add_column(f"Recall@{report.k}", justify="right")
    table.add_column("1st", justify="right")
    table.add_column("Found/Gold", justify="right")
    table.add_column("Subject", overflow="fold")
    for i, c in enumerate(report.cases, 1):
        if not c.evaluable:
            table.add_row(str(i), c.sha[:7], "[dim]—[/dim]", "—", "0/0",
                          f"[dim]{c.subject}  (no source files)[/dim]")
            continue
        rank = str(c.first_rank) if c.first_rank else "—"
        recall = f"{c.recall_at_k:.2f}"
        table.add_row(str(i), c.sha[:7], recall, rank,
                      f"{c.found}/{c.gold_total}", c.subject)
    console.print(table)

    scored = report.scored
    console.print(
        f"[bold]{len(scored)}[/bold] scored / {len(report.cases)} commits   "
        f"[dim]mean[/dim] Recall@{report.k}=[green]{report.mean_recall_at_k:.3f}[/green]   "
        f"MRR=[green]{report.mean_reciprocal_rank:.3f}[/green]"
    )


def _print_ablation_delta(on, off) -> None:
    """Print the with/without-import-edge comparison — the headline ablation."""

    d_recall = on.mean_recall_at_k - off.mean_recall_at_k
    d_mrr = on.mean_reciprocal_rank - off.mean_reciprocal_rank
    console.print(
        f"\n[bold]Import-edge ablation[/bold] (k={on.k}, {len(on.scored)} scored):\n"
        f"  with    edge:  Recall@{on.k}=[green]{on.mean_recall_at_k:.3f}[/green]  "
        f"MRR=[green]{on.mean_reciprocal_rank:.3f}[/green]\n"
        f"  without edge:  Recall@{off.k}={off.mean_recall_at_k:.3f}  "
        f"MRR={off.mean_reciprocal_rank:.3f}\n"
        f"  [bold]delta[/bold]:        Recall@{on.k}=[cyan]{d_recall:+.3f}[/cyan]  "
        f"MRR=[cyan]{d_mrr:+.3f}[/cyan]"
    )


def _eval_json(report, ablation_off=None) -> str:
    """Serialize one or two EvalReports (the second is the ablation-off run)."""

    import json
    from dataclasses import asdict

    def as_dict(rep) -> dict:
        return {
            "k": rep.k,
            "use_import_edge": rep.use_import_edge,
            "use_intent_roles": rep.use_intent_roles,
            "use_named_file_boost": rep.use_named_file_boost,
            "mean_recall_at_k": rep.mean_recall_at_k,
            "mean_reciprocal_rank": rep.mean_reciprocal_rank,
            "scored": len(rep.scored),
            "commits": len(rep.cases),
            "cases": [asdict(c) for c in rep.cases],
        }

    payload: dict = {"report": as_dict(report)}
    if ablation_off is not None:
        payload["ablation_off"] = as_dict(ablation_off)
        payload["delta"] = {
            "mean_recall_at_k": report.mean_recall_at_k - ablation_off.mean_recall_at_k,
            "mean_reciprocal_rank": report.mean_reciprocal_rank
            - ablation_off.mean_reciprocal_rank,
        }
    return json.dumps(payload, indent=2)


def _changed_since(repo: Path, ref: str) -> list[str]:
    """Resolve ``--since <ref>`` to a changed-file list via a read-only git diff."""

    import subprocess

    from codegraft.repo.git_scan import diff_changed_files

    try:
        return sorted(diff_changed_files(repo, ref))
    except (subprocess.SubprocessError, OSError) as exc:
        raise CodegraftError(
            f"could not read changes since {ref} (is {repo} a git repo?): {exc}"
        ) from exc


def _read_request(
    request: str | None, request_file: Path | None, *, required: bool
) -> str | None:
    """Resolve the feature request from an argument or a file.

    A file (when given) wins over the positional argument. Returns None when
    nothing is supplied and *required* is False (summary-only inspect).
    """

    if request_file is not None:
        text = request_file.read_text(encoding="utf-8").strip()
        if not text:
            raise CodegraftError(f"request file {request_file} is empty")
        return text
    if request and request.strip():
        return request.strip()
    if required:
        raise CodegraftError("provide a feature request argument or --request-file")
    return None


def _write_plan(
    repo: Path, output_dir: str, stem: str, markdown: str, dry_run: bool
) -> Path:
    plans_dir = repo / output_dir
    out_path = plans_dir / f"{stem}.md"
    if not dry_run:
        plans_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
    return out_path


def _write_debug_json(repo: Path, output_dir: str, stem: str, plan_obj) -> Path:
    """Write the raw plan object to plans/_debug/<stem>.plan.json for transparency."""

    debug_dir = repo / output_dir / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{stem}.plan.json"
    debug_path.write_text(plan_obj.model_dump_json(indent=2), encoding="utf-8")
    return debug_path


if __name__ == "__main__":
    app()
