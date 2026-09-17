"""Read-only cluster capability probes (SLURM, GPU GRES, Apptainer)."""

from __future__ import annotations

from dataclasses import dataclass

from vector_lab.cluster.slurm import merge_partitions, parse_scontrol_partitions, parse_sinfo_pipe_table
from vector_lab.config.models import ClusterProfile, GpuPartition
from vector_lab.exec import CommandResult
from vector_lab.ssh.session import SshSession

SINFO_FORMAT = r"sinfo -h -o '%P|%G|%a'"


@dataclass
class ApptainerProbe:
    available: bool
    command: str | None = None
    module_ok: bool = False
    detail: str = ""


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


def discover_apptainer(session: SshSession, *, module: str | None) -> ApptainerProbe:
    load = f"module load {module} >/dev/null 2>&1; " if module else ""
    result = session.exec(
        load + "command -v singularity || command -v apptainer || true",
        category="apptainer",
        check=False,
    )
    if result.skipped:
        return ApptainerProbe(available=False, module_ok=bool(module), detail="dry-run")
    path = result.stdout.strip().splitlines()
    resolved = path[-1].strip() if path else ""
    if not resolved:
        return ApptainerProbe(
            available=False,
            module_ok=False,
            detail=result.stderr.strip() or "apptainer/singularity not found",
        )
    command = "singularity" if resolved.endswith("singularity") else "apptainer"
    return ApptainerProbe(available=True, command=command, module_ok=bool(module), detail=resolved)


def apply_discovery(profile: ClusterProfile, partitions: list[GpuPartition], apptainer: ApptainerProbe) -> ClusterProfile:
    if partitions:
        profile.discovered_partitions = partitions
    if apptainer.command and not profile.apptainer.command:
        profile.apptainer.command = apptainer.command
    return profile


def result_ok(result: CommandResult) -> bool:
    return (not result.skipped) and result.returncode == 0
