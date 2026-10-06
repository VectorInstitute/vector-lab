"""Read-only cluster capability probes (SLURM, GPU GRES, Apptainer)."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from vector_sim.cluster.detect import (
    SCRATCH_ENV_KEYS,
    candidate_scratch_dirs,
    infer_scratch,
    parse_env_assignments,
)
from vector_sim.cluster.slurm import (
    merge_partitions,
    parse_scontrol_partitions,
    parse_sinfo_pipe_table,
    select_default_partition,
)
from vector_sim.config.models import ClusterProfile, GpuPartition
from vector_sim.exec import CommandResult, CommandRunner
from vector_sim.ssh.session import SshSession

SINFO_FORMAT = r"sinfo -h -o '%P|%G|%a'"

# Commands whose presence distinguishes a usable remote shell from a bare one.
CAPABILITY_COMMANDS = ("sinfo", "sbatch", "apptainer", "singularity", "module")
CAPABILITY_PROBE = (
    "for c in " + " ".join(CAPABILITY_COMMANDS) + "; do "
    'command -v "$c" >/dev/null 2>&1 && echo "$c"; done; '
    '[ -n "$SCRATCH" ] && echo SCRATCH; true'
)
SCRATCH_ENV_PROBE = (
    "env | grep -E '^(" + "|".join(SCRATCH_ENV_KEYS) + ")=' || true"
)
APPTAINER_MODULE_KEYWORDS = ("apptainer", "singularity")


@dataclass
class RemoteShellProbe:
    remote_shell: str = "none"
    capabilities: frozenset[str] = frozenset()
    detail: str = ""


@dataclass
class ApptainerProbe:
    available: bool
    command: str | None = None
    module: str | None = None
    module_ok: bool = False
    detail: str = ""


def parse_capabilities(text: str) -> frozenset[str]:
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def discover_remote_shell(session: SshSession) -> RemoteShellProbe:
    """Decide whether remote commands need ``bash -l``.

    Sites that export SLURM, Lmod, and scratch only from ``/etc/profile.d`` look
    completely empty to a plain ``ssh host cmd``, so compare both shells and keep
    whichever exposes more. Callers only ever upgrade to ``login``.
    """
    plain = session.exec(
        CAPABILITY_PROBE, category="ssh-capabilities", check=False, login_shell=False
    )
    if plain.skipped:
        return RemoteShellProbe(detail="dry-run")
    plain_caps = parse_capabilities(plain.stdout) if plain.returncode == 0 else frozenset()

    login = session.exec(
        CAPABILITY_PROBE, category="ssh-capabilities", check=False, login_shell=True
    )
    login_caps = (
        parse_capabilities(login.stdout)
        if not login.skipped and login.returncode == 0
        else frozenset()
    )

    if len(login_caps) > len(plain_caps):
        gained = ", ".join(sorted(login_caps - plain_caps))
        return RemoteShellProbe("login", login_caps, f"login shell adds {gained}")
    detail = ", ".join(sorted(plain_caps)) or "no scheduler tools found in either shell"
    return RemoteShellProbe("none", plain_caps, detail)


def resolve_remote_shell(
    profile: ClusterProfile,
    session: SshSession,
    runner: CommandRunner,
) -> tuple[SshSession, RemoteShellProbe]:
    """Upgrade the profile and session to a login shell when the plain shell is bare."""
    probe = discover_remote_shell(session)
    if probe.remote_shell == "login":
        profile.scheduler.remote_shell = "login"
    if profile.scheduler.remote_shell == "login" and not session.login_shell:
        session = SshSession(profile.ssh_alias, runner, login_shell=True)
    return session, probe


def discover_scratch(
    session: SshSession,
    *,
    remote_user: str | None,
    home_dir: str | None,
) -> tuple[str | None, str]:
    """Resolve scratch from exported variables, then from conventional directories."""
    env_probe = session.exec(SCRATCH_ENV_PROBE, category="ssh", check=False)
    env = parse_env_assignments(env_probe.stdout) if result_ok(env_probe) else {}

    existing: list[str] = []
    candidates = candidate_scratch_dirs(remote_user=remote_user, home_dir=home_dir)
    if candidates:
        listing = " ".join(shlex.quote(c) for c in candidates)
        probe = f'for d in {listing}; do [ -d "$d" ] && [ -w "$d" ] && echo "$d"; done; true'
        result = session.exec(probe, category="ssh", check=False)
        if result_ok(result):
            existing = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    return infer_scratch(env=env, existing_directories=existing)


def parse_module_names(text: str, *, keywords: tuple[str, ...]) -> list[str]:
    """Extract base module names from ``module -t avail`` output."""
    names: list[str] = []
    wanted = {k.lower() for k in keywords}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.endswith(":"):
            continue
        base = line.split("(", 1)[0].split("/", 1)[0].strip()
        if base.lower() in wanted and base not in names:
            names.append(base)
    return names


def discover_partitions(session: SshSession) -> list[GpuPartition]:
    sinfo = session.exec(
        SINFO_FORMAT,
        category="slurm",
        check=False,
        suggestion="Ensure SLURM is available in a login shell (ssh alias && bash -l -c sinfo).",
    )
    scontrol = session.exec(
        "scontrol show partition",
        category="slurm",
        check=False,
    )
    groups = []
    if not sinfo.skipped and sinfo.returncode == 0:
        groups.append(parse_sinfo_pipe_table(sinfo.stdout))
    if not scontrol.skipped and scontrol.returncode == 0:
        groups.append(parse_scontrol_partitions(scontrol.stdout))
    return merge_partitions(*groups)


def probe_apptainer_command(session: SshSession, module: str | None) -> ApptainerProbe:
    """Check whether apptainer/singularity is callable, optionally after loading a module."""
    load = f"module load {module} >/dev/null 2>&1; " if module else ""
    result = session.exec(
        load + "command -v singularity || command -v apptainer || true",
        category="apptainer",
        check=False,
    )
    if result.skipped:
        return ApptainerProbe(
            available=False, module=module, module_ok=bool(module), detail="dry-run"
        )
    path = result.stdout.strip().splitlines()
    resolved = path[-1].strip() if path else ""
    if not resolved:
        return ApptainerProbe(
            available=False,
            module=module,
            module_ok=False,
            detail=result.stderr.strip() or "apptainer/singularity not found",
        )
    command = "singularity" if resolved.endswith("singularity") else "apptainer"
    return ApptainerProbe(
        available=True,
        command=command,
        module=module,
        module_ok=bool(module),
        detail=resolved,
    )


def discover_apptainer_modules(session: SshSession) -> list[str]:
    result = session.exec(
        "module -t avail " + " ".join(APPTAINER_MODULE_KEYWORDS) + " 2>&1 || true",
        category="apptainer",
        check=False,
    )
    if not result_ok(result):
        return []
    return parse_module_names(result.stdout, keywords=APPTAINER_MODULE_KEYWORDS)


def discover_apptainer(session: SshSession, *, module: str | None) -> ApptainerProbe:
    """Find a working container runtime, searching the module system when needed."""
    probe = probe_apptainer_command(session, module)
    if probe.available or probe.detail == "dry-run":
        return probe
    for candidate in discover_apptainer_modules(session):
        if candidate == module:
            continue
        found = probe_apptainer_command(session, candidate)
        if found.available:
            return found
    return probe


def apply_discovery(profile: ClusterProfile, partitions: list[GpuPartition], apptainer: ApptainerProbe) -> ClusterProfile:
    from vector_sim.cluster.adapters import ISAAC_LAB_EXEC_ARGS, WRITABLE_TMPFS

    if partitions:
        profile.discovered_partitions = partitions
        if not profile.default_job.partition:
            name, gpu_type = select_default_partition(partitions)
            if name:
                profile.default_job.partition = name
            if gpu_type and not profile.default_job.gpu_type:
                profile.default_job.gpu_type = gpu_type
    if apptainer.command and not profile.apptainer.command:
        profile.apptainer.command = apptainer.command
    if apptainer.module and not profile.apptainer.module:
        profile.apptainer.module = apptainer.module
    if apptainer.available and not profile.apptainer.extra_exec_args:
        # Isaac Sim needs GPU passthrough and a writable overlay wherever it runs.
        profile.apptainer.extra_exec_args = list(ISAAC_LAB_EXEC_ARGS)
        profile.apptainer.writable_mode = profile.apptainer.writable_mode or WRITABLE_TMPFS
    return profile


def result_ok(result: CommandResult) -> bool:
    return (not result.skipped) and result.returncode == 0
