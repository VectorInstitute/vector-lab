"""Read-only environment checks with INFO / WARNING / ERROR classification."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from vector_lab.cluster.adapters import adapter_for
from vector_lab.config.models import ClusterProfile
from vector_lab.config.store import ConfigStore
from vector_lab.errors import ConfigError, DiscoveryError
from vector_lab.exec import CommandRunner
from vector_lab.compat import WRAPPER_SCHEMA_VERSION, classify_isaaclab_commit
from vector_lab.images.state import ImageStateStore
from vector_lab.local.repo import discover_isaac_lab
from vector_lab.ssh.multiplex import RECOMMENDED_SSH_SNIPPET
from vector_lab.ssh.session import SshSession, resolved_ssh_from_g


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def classify_result(*, passed: bool, required: bool) -> Severity:
    """Map a boolean check onto doctor severity."""
    if passed:
        return Severity.INFO
    if required:
        return Severity.ERROR
    return Severity.WARNING


@dataclass
class CheckResult:
    section: str
    name: str
    severity: Severity
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.severity != Severity.ERROR


def tool_on_path(name: str) -> str | None:
    return shutil.which(name)


def docker_socket_accessible(sock: Path | None = None) -> bool:
    path = sock or Path("/var/run/docker.sock")
    return path.exists() and os.access(path, os.R_OK | os.W_OK)


def docker_access_hint() -> str:
    return (
        "Docker is on PATH but this user cannot access the daemon. "
        "Recommended: sudo usermod -aG docker $USER  then log out and back in. "
        "Keep docker as a supplementary group; do not use newgrp docker, chmod 666 on "
        "docker.sock, or sudo for the whole workflow."
    )


class Doctor:
    def __init__(
        self,
        runner: CommandRunner,
        store: ConfigStore,
        *,
        cluster: str | None = None,
        start_dir: Path | None = None,
    ) -> None:
        self.runner = runner
        self.store = store
        self.cluster = cluster or store.get_active()
        self.start_dir = Path(start_dir or Path.cwd())

    def run(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        results.extend(self._local_checks())
        profile = None
        if self.cluster:
            try:
                profile = self.store.load_profile(self.cluster)
                results.append(
                    CheckResult("SSH", "alias", Severity.INFO, "OK", profile.ssh_alias)
                )
            except ConfigError as exc:
                results.append(
                    CheckResult("SSH", "profile", Severity.ERROR, "missing", str(exc))
                )
        else:
            results.append(
                CheckResult(
                    "SSH",
                    "profile",
                    Severity.ERROR,
                    "missing",
                    "no cluster profile; run vector-lab onboard bonecho",
                )
            )

        results.extend(self._ssh_checks(profile))
        results.extend(self._cluster_checks(profile))
        results.extend(self._container_checks(profile))
        results.extend(self._deployment_checks(profile))
        return results

    def _local_checks(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for tool, required in (("git", True), ("ssh", True), ("rsync", True), ("docker", True)):
            path = tool_on_path(tool)
            passed = path is not None
            results.append(
                CheckResult(
                    "Local",
                    tool.capitalize() if tool != "ssh" else "SSH client",
                    classify_result(passed=passed, required=required),
                    "OK" if passed else "missing",
                    path or f"{tool} not found on PATH",
                )
            )

        docker_path = tool_on_path("docker")
        if docker_path:
            accessible = docker_socket_accessible()
            results.append(
                CheckResult(
                    "Local",
                    "Docker access",
                    classify_result(passed=accessible, required=True),
                    "OK" if accessible else "denied",
                    docker_path if accessible else docker_access_hint(),
                )
            )

        apptainer = tool_on_path("apptainer") or tool_on_path("singularity")
        results.append(
            CheckResult(
                "Local",
                "Apptainer",
                classify_result(passed=apptainer is not None, required=False),
                "OK" if apptainer else "missing",
                apptainer or "needed later for local image conversion",
            )
        )

        try:
            configured = None
            if self.cluster:
                try:
                    configured = self.store.load_profile(self.cluster).isaaclab_path
                except ConfigError:
                    configured = None
            repo = discover_isaac_lab(configured=configured, start=self.start_dir)
            results.append(CheckResult("Local", "Isaac Lab", Severity.INFO, "OK", str(repo.root)))
            head = None
            git = self.runner.run(
                ["git", "-C", str(repo.root), "rev-parse", "HEAD"],
                category="git",
                check=False,
                dry_run_skip=False,
            )
            if not git.skipped and git.returncode == 0:
                head = git.stdout.strip()
            match = classify_isaaclab_commit(head)
            sev = Severity.INFO if match.classification == "tested" else Severity.WARNING
            results.append(
                CheckResult(
                    "Local",
                    "Isaac Lab version",
                    sev,
                    match.classification,
                    match.detail,
                )
            )
        except DiscoveryError as exc:
            results.append(CheckResult("Local", "Isaac Lab", Severity.ERROR, "missing", f"{exc}  next: {exc.suggestion or 'vector-lab onboard bonecho'}"))
        return results

    def _ssh_checks(self, profile: ClusterProfile | None) -> list[CheckResult]:
        if profile is None:
            return []
        session = SshSession(
            profile.ssh_alias,
            self.runner,
            login_shell=profile.scheduler.remote_shell == "login",
        )
        results: list[CheckResult] = []
        g = session.runner.run(
            ["ssh", "-G", profile.ssh_alias],
            category="ssh-config",
            check=False,
            dry_run_skip=False,
            suggestion=f"Add Host {profile.ssh_alias} to ~/.ssh/config",
        )
        if g.returncode != 0:
            snippet = RECOMMENDED_SSH_SNIPPET.replace("bonecho", profile.ssh_alias)
            results.append(
                CheckResult(
                    "SSH",
                    "config",
                    Severity.ERROR,
                    "failed",
                    (g.stderr.strip() or "ssh alias not resolved")
                    + f". Add Host {profile.ssh_alias} to ~/.ssh/config yourself (vector-lab will not write it):\n{snippet}",
                )
            )
            return results
        if g.skipped:
            results.append(CheckResult("SSH", "config", Severity.INFO, "skipped", "dry-run"))
        else:
            resolved = resolved_ssh_from_g(profile.ssh_alias, g.stdout)
            host = resolved.hostname or profile.resolved_host or "unknown"
            results.append(CheckResult("SSH", "resolved host", Severity.INFO, "OK", host))

        who = session.exec(
            "whoami",
            category="ssh",
            check=False,
            suggestion="SSH session missing or MFA expired. Run: vector-lab auth " + profile.ssh_alias,
        )
        if who.skipped:
            results.append(
                CheckResult("SSH", "connection", Severity.INFO, "skipped", "dry-run; not connecting")
            )
            if profile.remote_user:
                results.append(CheckResult("SSH", "username", Severity.INFO, "OK", profile.remote_user))
            if profile.scratch_dir:
                results.append(CheckResult("SSH", "scratch", Severity.INFO, "OK", profile.scratch_dir))
            if profile.home_dir:
                results.append(CheckResult("SSH", "home", Severity.INFO, "OK", profile.home_dir))
            return results
        if who.returncode != 0:
            results.append(
                CheckResult(
                    "SSH",
                    "connection",
                    Severity.ERROR,
                    "failed",
                    (who.stderr.strip() or "non-interactive SSH failed")
                    + f". Run: vector-lab auth {profile.ssh_alias}",
                )
            )
            return results
        user = who.stdout.strip()
        results.append(CheckResult("SSH", "connection", Severity.INFO, "OK", profile.ssh_alias))
        results.append(CheckResult("SSH", "authentication", Severity.INFO, "OK", "BatchMode"))
        results.append(CheckResult("SSH", "username", Severity.INFO, "OK", user))
        results.append(CheckResult("SSH", "remote user", Severity.INFO, "OK", user))
        if profile.scratch_dir:
            results.append(CheckResult("SSH", "scratch", Severity.INFO, "OK", profile.scratch_dir))
        else:
            results.append(
                CheckResult(
                    "SSH",
                    "scratch",
                    Severity.WARNING,
                    "unresolved",
                    "no scratch_dir in profile; re-run init after connectivity works",
                )
            )
        if profile.home_dir:
            results.append(CheckResult("SSH", "home", Severity.INFO, "OK", profile.home_dir))
        return results

    def _cluster_checks(self, profile: ClusterProfile | None) -> list[CheckResult]:
        if profile is None:
            return []
        results = [
            CheckResult("Cluster", "scheduler", Severity.INFO, "OK", profile.scheduler.type),
            CheckResult("Cluster", "SLURM", Severity.INFO, "OK", profile.scheduler.type),
            CheckResult(
                "Cluster",
                "login shell",
                Severity.INFO,
                "required" if profile.scheduler.remote_shell == "login" else "not required",
                profile.scheduler.remote_shell,
            ),
        ]
        apptainer = profile.apptainer
        if apptainer.module or apptainer.writable_mode:
            detail = f"module={apptainer.module or 'none'} writable_mode={apptainer.writable_mode or 'unset'}"
            results.append(CheckResult("Cluster", "Apptainer", Severity.INFO, "OK", detail))
        else:
            results.append(
                CheckResult(
                    "Cluster",
                    "Apptainer",
                    Severity.WARNING,
                    "unconfigured",
                    "no module/writable_mode in profile",
                )
            )
        gpu = profile.default_job.gpu_type
        partition = profile.default_job.partition
        if partition:
            results.append(
                CheckResult(
                    "Cluster",
                    partition,
                    Severity.INFO,
                    "OK",
                    partition,
                )
            )
            results.append(
                CheckResult(
                    "Cluster",
                    f"{(gpu or 'GPU').upper()} partition",
                    Severity.INFO,
                    "OK",
                    partition,
                )
            )
        else:
            results.append(
                CheckResult("Cluster", "GPU partition", Severity.WARNING, "unset", "no default partition")
            )
        adapter = adapter_for(profile.cluster_type)
        results.append(
            CheckResult(
                "Cluster",
                "cluster_type",
                Severity.INFO,
                "OK",
                profile.cluster_type or "slurm",
            )
        )
        results.append(CheckResult("Cluster", "adapter", Severity.INFO, "OK", adapter.name))
        if profile.scratch_dir:
            results.append(CheckResult("Cluster", "scratch", Severity.INFO, "OK", profile.scratch_dir))
        else:
            results.append(CheckResult("Cluster", "scratch", Severity.WARNING, "unresolved", "run vector-lab setup"))
        results.append(
            CheckResult("Cluster", "wrapper schema", Severity.INFO, "OK", WRAPPER_SCHEMA_VERSION)
        )
        return results

    def _deployment_checks(self, profile: ClusterProfile | None) -> list[CheckResult]:
        if profile is None:
            return []
        state = ImageStateStore(self.store).get(profile.image.name or "base")
        docker_ready = bool(state and state.built_image_digest)
        conv_ready = bool(state and state.conversion_fingerprint and state.local_artifact)
        remote_ready = bool(state and state.remote_sha256)
        return [
            CheckResult(
                "Deployment",
                "Docker image",
                Severity.INFO,
                "READY" if docker_ready else "NEEDS BUILD",
                (state.built_image_digest if state else "") or profile.image.docker_image,
            ),
            CheckResult(
                "Deployment",
                "Converted artifact",
                Severity.INFO,
                "READY" if conv_ready else "NEEDS CONVERSION",
                (state.local_artifact if state else "") or "",
            ),
            CheckResult(
                "Deployment",
                "Remote artifact",
                Severity.INFO,
                "READY" if remote_ready else "NEEDS PUSH",
                (state.remote_path if state else "") or "",
            ),
        ]

    def _container_checks(self, profile: ClusterProfile | None) -> list[CheckResult]:
        if profile is None:
            return []
        fp = profile.fingerprints
        local = "present" if fp.built_image_digest else "unknown"
        conv = "present" if fp.conversion_fingerprint else "unknown"
        match = (
            "matching"
            if fp.built_image_digest and fp.conversion_fingerprint
            else "not checked"
        )
        return [
            CheckResult("Container", "local image", Severity.INFO, local, profile.image.docker_image),
            CheckResult("Container", "conversion", Severity.INFO, conv, fp.conversion_fingerprint or "no fingerprint"),
            CheckResult("Container", "checksum", Severity.INFO, match, fp.build_input_fingerprint or "no build input fingerprint"),
        ]


def format_doctor_report(results: list[CheckResult]) -> str:
    lines: list[str] = []
    current = None
    for item in results:
        if item.section != current:
            current = item.section
            lines.append(f"{current}:")
        detail = f"  {item.detail}" if item.detail else ""
        lines.append(f"  {item.name:<18} {item.status:<12} [{item.severity.value}]{detail}")
    errors = sum(1 for r in results if r.severity == Severity.ERROR)
    warnings = sum(1 for r in results if r.severity == Severity.WARNING)
    lines.append("")
    lines.append(f"summary: {errors} error(s), {warnings} warning(s)")
    lines.append("Overall:")
    lines.append("  READY" if errors == 0 else "  NOT READY")
    return "\n".join(lines)
