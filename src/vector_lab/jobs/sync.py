"""Source sync to unique remote run directories."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from vector_lab.config.models import ClusterProfile
from vector_lab.exec import CommandRunner
from vector_lab.jobs.generate import ENV_NAME, RUNNER_NAME, SUBMIT_NAME
from vector_lab.ssh.session import SshSession

RSYNC_EXCLUDES = (
    ".git/",
    ".github/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.pyc",
    "*.egg-info/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "docker/cluster/exports/",
    ".vector-lab/artifacts/",
    ".vector-lab/work/",
    ".isaac-cluster/artifacts/",
    ".isaac-cluster/work/",
    "logs/",
    "**/logs/",
    "**/runs/",
    "**/output/",
    "**/outputs/",
    "**/videos/",
    "**/wandb/",
    "_isaac_sim",
    "env_isaaclab/",
    "*.sif",
    "*.tar",
)

GENERATED_REMOTE_SUBDIR = ".vector-lab/generated"


def run_dir_name(*, when: datetime | None = None) -> str:
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    return f"isaaclab_{stamp}"


def remote_run_path(profile: ClusterProfile, *, when: datetime | None = None) -> str:
    scratch = (profile.scratch_dir or "").rstrip("/")
    if not scratch:
        raise ValueError("profile.scratch_dir is required for source sync")
    return f"{scratch}/{run_dir_name(when=when)}"


def rsync_source_args(
    local_repo: Path,
    ssh_alias: str,
    remote_dir: str,
    *,
    excludes: tuple[str, ...] = RSYNC_EXCLUDES,
) -> list[str]:
    args = [
        "rsync",
        "-a",
        "--info=progress2",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=15",
    ]
    for pattern in excludes:
        args.extend(["--exclude", pattern])
    args.append(str(local_repo).rstrip("/") + "/")
    args.append(f"{ssh_alias}:{remote_dir.rstrip('/')}/")
    return args


def stage_generated_into_repo(repo: Path, generated_files: dict[str, str]) -> Path:
    """Write generated wrappers under repo/.vector-lab/generated for rsync."""
    dest = Path(repo) / ".vector-lab" / "generated"
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in generated_files.items():
        path = dest / name
        path.write_text(content)
        if name.endswith(".sh"):
            path.chmod(path.stat().st_mode | 0o111)
    return dest


def sync_run_directory(
    runner: CommandRunner,
    session: SshSession,
    *,
    local_repo: Path,
    remote_dir: str,
    generated: dict[str, str],
) -> list[str]:
    """Create remote dir, stage generated wrappers, rsync source. Returns planned cmds."""
    if not runner.dry_run:
        stage_generated_into_repo(local_repo, generated)
        session.exec(
            f"mkdir -p {remote_dir}",
            category="remote-fs",
            check=True,
        )
    else:
        session.exec(
            f"mkdir -p {remote_dir}",
            category="remote-fs",
            check=False,
        )
    rsync_args = rsync_source_args(local_repo, session.alias, remote_dir)
    runner.run(
        rsync_args,
        category="rsync-source",
        suggestion="Check SSH alias and remote scratch permissions.",
    )
    return [
        f"mkdir -p {remote_dir}",
        f"stage generated wrappers -> {local_repo}/.vector-lab/generated/",
        runner.format_args(rsync_args),
    ]


def remote_submit_path(remote_run_dir: str) -> str:
    return f"{remote_run_dir.rstrip('/')}/{GENERATED_REMOTE_SUBDIR}/{SUBMIT_NAME}"


def remote_wrapper_paths(remote_run_dir: str) -> dict[str, str]:
    base = f"{remote_run_dir.rstrip('/')}/{GENERATED_REMOTE_SUBDIR}"
    return {
        ENV_NAME: f"{base}/{ENV_NAME}",
        SUBMIT_NAME: f"{base}/{SUBMIT_NAME}",
        RUNNER_NAME: f"{base}/{RUNNER_NAME}",
    }
