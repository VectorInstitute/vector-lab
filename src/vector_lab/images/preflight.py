"""Local preflights for Docker build, Apptainer conversion, and push."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from vector_lab.exec import CommandRunner
from vector_lab.local.docker_util import (
    DOCKER_ACCESS_HINT,
    DOCKER_MISSING_HINT,
    docker_socket_accessible,
    fakeroot_primary_group_problem,
    permission_denied,
    require_docker,
    which_docker,
)
from vector_lab.ssh.session import SshSession


@dataclass
class PreflightIssue:
    severity: str  # ERROR or WARNING
    message: str
    suggestion: str | None = None


def docker_preflight(runner: CommandRunner) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    binary = which_docker()
    if not binary:
        issues.append(
            PreflightIssue(
                "ERROR",
                "Docker is not on PATH",
                DOCKER_MISSING_HINT,
            )
        )
        return issues
    if not docker_socket_accessible():
        issues.append(PreflightIssue("ERROR", "Cannot access the Docker socket", DOCKER_ACCESS_HINT))
        return issues
    info = runner.run(
        [binary, "info"],
        category="docker-preflight",
        check=False,
        dry_run_skip=True,
        suggestion=DOCKER_ACCESS_HINT,
    )
    if info.skipped:
        return issues
    if info.returncode != 0:
        hint = DOCKER_ACCESS_HINT if permission_denied(info.stderr) else info.stderr.strip()
        issues.append(PreflightIssue("ERROR", "Docker daemon is not usable", hint))
    return issues


def apptainer_preflight(runner: CommandRunner, *, group_name: str | None = None) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    binary = shutil.which("apptainer")
    if not binary:
        issues.append(
            PreflightIssue(
                "ERROR",
                "apptainer is not on PATH",
                "Install Apptainer locally for conversion: https://apptainer.org/docs/admin/main/installation.html",
            )
        )
        return issues
    version = runner.run(
        [binary, "--version"],
        category="apptainer-preflight",
        check=False,
        dry_run_skip=True,
    )
    if not version.skipped and version.returncode != 0:
        issues.append(PreflightIssue("ERROR", "apptainer --version failed", version.stderr.strip()))
    problem = fakeroot_primary_group_problem(group_name)
    if problem:
        issues.append(PreflightIssue("ERROR", problem, "Restore your normal primary group before conversion."))
    return issues


def disk_preflight(path: Path, *, min_bytes: int = 40 * 1024**3, free_bytes: int | None = None) -> list[PreflightIssue]:
    try:
        free = shutil.disk_usage(path).free if free_bytes is None else free_bytes
    except OSError as exc:
        return [PreflightIssue("WARNING", f"could not measure disk space at {path}: {exc}")]
    if free < min_bytes:
        gb = free / 1024**3
        need = min_bytes / 1024**3
        return [
            PreflightIssue(
                "WARNING",
                f"only {gb:.1f} GiB free at {path}; conversion/upload may need ~{need:.0f} GiB",
            )
        ]
    return []


def remote_push_preflight(
    session: SshSession,
    *,
    containers_dir: str | None,
) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    if not containers_dir:
        issues.append(
            PreflightIssue(
                "ERROR",
                "profile has no containers path",
                "Run: vector-lab onboard bonecho so scratch container paths are discovered.",
            )
        )
        return issues
    probe = session.exec(
        f'test -d {containers_dir!s} && echo DIR_OK; command -v sha256sum; df -k {containers_dir} | tail -n 1 || true',
        category="ssh-preflight",
        check=False,
    )
    if probe.skipped:
        return issues
    if probe.returncode != 0:
        issues.append(
            PreflightIssue(
                "ERROR",
                "SSH preflight failed",
                probe.stderr.strip() or "Run: vector-lab auth bonecho",
            )
        )
        return issues
    out = probe.stdout
    if "DIR_OK" not in out:
        issues.append(
            PreflightIssue(
                "ERROR",
                f"remote container directory does not exist: {containers_dir}",
                "Run: vector-lab onboard bonecho to create remote scratch directories.",
            )
        )
    if "sha256sum" not in out:
        issues.append(
            PreflightIssue(
                "ERROR",
                "sha256sum is not available on the cluster",
                "Ask the site admins to provide coreutils, or configure an alternate checksum tool.",
            )
        )
    return issues


def raise_if_errors(issues: list[PreflightIssue]) -> None:
    from vector_lab.errors import VectorLabError

    errors = [i for i in issues if i.severity == "ERROR"]
    if not errors:
        return
    first = errors[0]
    raise VectorLabError(
        first.message,
        category="preflight",
        suggestion=first.suggestion,
    )


def require_docker_bin() -> str:
    return require_docker()
