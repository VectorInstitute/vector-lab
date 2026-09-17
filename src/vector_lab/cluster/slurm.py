"""SLURM output parsing and resource formatting. No Bonecho-specific defaults here."""

from __future__ import annotations

import re

from vector_lab.config.models import DefaultJob, GpuPartition

_GPU_IN_GRES = re.compile(r"gpu(?::([^:,]+))?(?::(\d+))?", re.IGNORECASE)


def slurm_time(value: str | None) -> str:
    """Normalize walltime to SLURM's ``HH:MM:SS`` (or ``D-HH:MM:SS``)."""
    if not value:
        return "01:00:00"
    text = value.strip().lower()
    if re.fullmatch(r"\d+-\d{1,2}:\d{2}:\d{2}", text):
        return text
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", text):
        hours, minutes, seconds = text.split(":")
        return f"{int(hours):02d}:{minutes}:{seconds}"
    if re.fullmatch(r"\d{1,2}:\d{2}", text):
        hours, minutes = text.split(":")
        return f"{int(hours):02d}:{minutes}:00"
    match = re.fullmatch(r"(\d+)\s*h(ours?)?", text)
    if match:
        return f"{int(match.group(1)):02d}:00:00"
    match = re.fullmatch(r"(\d+)\s*m(in(utes?)?)?", text)
    if match:
        minutes = int(match.group(1))
        return f"{minutes // 60:02d}:{minutes % 60:02d}:00"
    return value


def gres_request(job: DefaultJob) -> str | None:
    if not job.gpu_count:
        return None
    if job.gpu_type:
        return f"gpu:{job.gpu_type}:{job.gpu_count}"
    return f"gpu:{job.gpu_count}"


def gpu_types_from_gres(gres: str | None) -> list[str]:
    if not gres or gres in {"(null)", "N/A"}:
        return []
    types: list[str] = []
    for match in _GPU_IN_GRES.finditer(gres):
        kind = match.group(1)
        if kind and kind.isdigit():
            continue
        if kind and kind.lower() != "gpu" and kind not in types:
            types.append(kind)
    return types


def parse_sinfo_pipe_table(text: str) -> list[GpuPartition]:
    """Parse ``sinfo -h -o '%P|%G|%a'`` output."""
    partitions: list[GpuPartition] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("PARTITION"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if not parts or not parts[0]:
            continue
        name = parts[0].rstrip("*")
        if name in seen:
            continue
        seen.add(name)
        gres = parts[1] if len(parts) > 1 else None
        state = parts[2] if len(parts) > 2 else None
        partitions.append(
            GpuPartition(
                name=name,
                gres=gres or None,
                state=state or None,
                gpu_types=gpu_types_from_gres(gres),
            )
        )
    return partitions


def parse_scontrol_partitions(text: str) -> list[GpuPartition]:
    """Parse ``scontrol show partition`` blocks."""
    partitions: list[GpuPartition] = []
    current: dict[str, str] = {}

    def flush() -> None:
        if not current.get("PartitionName"):
            return
        name = current["PartitionName"].rstrip("*")
        gres = current.get("Gres") or current.get("TRES")
        partitions.append(
            GpuPartition(
                name=name,
                gres=current.get("Gres"),
                gpu_types=gpu_types_from_gres(gres),
            )
        )

    for raw in text.splitlines() + [""]:
        line = raw.strip()
        if not line:
            flush()
            current = {}
            continue
        tokens = line.split()
        if tokens and tokens[0].startswith("PartitionName=") and current.get("PartitionName"):
            flush()
            current = {}
        for token in tokens:
            if "=" in token:
                key, value = token.split("=", 1)
                current[key] = value
    return partitions


def merge_partitions(*groups: list[GpuPartition]) -> list[GpuPartition]:
    by_name: dict[str, GpuPartition] = {}
    for group in groups:
        for part in group:
            existing = by_name.get(part.name)
            if existing is None:
                by_name[part.name] = part
                continue
            gres = existing.gres or part.gres
            types = list(dict.fromkeys([*existing.gpu_types, *part.gpu_types]))
            by_name[part.name] = GpuPartition(
                name=part.name,
                gres=gres,
                state=existing.state or part.state,
                gpu_types=types,
            )
    return list(by_name.values())
