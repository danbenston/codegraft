"""Discovery tests for the git-aware path."""

from __future__ import annotations

from pathlib import Path

from codegraft.config import Config
from codegraft.repo.discover import discover_repo
from tests.conftest import git, init_git_repo, requires_git, write


def _scan(root: Path):
    return discover_repo(root, Config.load(root))


@requires_git
def test_git_scan_returns_tracked_and_untracked(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    write(tmp_path, "app/main.py", "print('hi')\n")
    git(tmp_path, "add", "app/main.py")
    git(tmp_path, "commit", "-q", "-m", "init")
    write(tmp_path, "app/new_file.py")  # untracked

    scan = _scan(tmp_path)
    tracked = {f.path: f.tracked for f in scan.files}

    assert scan.used_git is True
    assert tracked["app/main.py"] is True
    assert tracked["app/new_file.py"] is False


@requires_git
def test_untracked_ignored_file_excluded(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    write(tmp_path, "keep.py")
    write(tmp_path, "debug.log")
    write(tmp_path, ".gitignore", "*.log\n")
    git(tmp_path, "add", "keep.py", ".gitignore")
    git(tmp_path, "commit", "-q", "-m", "init")

    paths = {f.path for f in _scan(tmp_path).files}
    assert "keep.py" in paths
    assert "debug.log" not in paths  # untracked + gitignored


@requires_git
def test_tracked_file_survives_later_gitignore(tmp_path: Path) -> None:
    """A tracked file stays listed even after a matching ignore rule is added."""

    init_git_repo(tmp_path)
    write(tmp_path, "config.yaml")
    git(tmp_path, "add", "config.yaml")
    git(tmp_path, "commit", "-q", "-m", "init")
    # Now ignore it — but it is already tracked.
    write(tmp_path, ".gitignore", "config.yaml\n")

    paths = {f.path for f in _scan(tmp_path).files}
    assert "config.yaml" in paths


@requires_git
def test_secrets_and_binaries_excluded(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    write(tmp_path, "app.py")
    write(tmp_path, ".env", "SECRET=abc\n")
    write(tmp_path, "server.pem", "-----BEGIN KEY-----\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x00binary")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")

    scan = _scan(tmp_path)
    paths = {f.path for f in scan.files}
    reasons = {s.path: s.reason for s in scan.skipped}

    assert "app.py" in paths
    assert ".env" not in paths and reasons[".env"] == "secret"
    assert "server.pem" not in paths and reasons["server.pem"] == "secret"
    assert "logo.png" not in paths and reasons["logo.png"] == "binary"


@requires_git
def test_oversized_file_excluded(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    write(tmp_path, "small.py", "ok\n")
    write(tmp_path, "huge.py", "x" * 5000)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")

    config = Config.load(tmp_path)
    config.repo.max_file_bytes = 1000
    scan = discover_repo(tmp_path, config)

    paths = {f.path for f in scan.files}
    reasons = {s.path: s.reason for s in scan.skipped}
    assert "small.py" in paths
    assert "huge.py" not in paths and reasons["huge.py"] == "oversized"


@requires_git
def test_truncates_at_max_candidate_files(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    for i in range(10):
        write(tmp_path, f"f{i}.py")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")

    config = Config.load(tmp_path)
    config.repo.max_candidate_files = 3
    scan = discover_repo(tmp_path, config)
    assert scan.file_count == 3
    assert scan.truncated is True


def test_additional_credential_shapes_are_excluded(tmp_path: Path) -> None:
    """Credential files that slipped past the original denylist.

    Notably the *suffix* env form: `.env` and `.env.*` were covered but
    "production.env" matched neither, and this denylist is the one place whose
    posture is "when unsure, exclude".
    """

    from codegraft.repo import safety

    for name in (
        "production.env", "local.env", "credentials.json", ".netrc",
        "_netrc", ".git-credentials", "terraform.tfvars", "auth.p8", "key.asc",
    ):
        assert safety.classify(name, tmp_path / name, 10, 1_000) == "secret", name

    # Ordinary source files are untouched by the widened patterns.
    for name in ("environment.py", "credentials_test_helper.py", "main.py"):
        write(tmp_path, name, "x = 1\n")
        assert safety.classify(name, tmp_path / name, 10, 1_000) is None, name
