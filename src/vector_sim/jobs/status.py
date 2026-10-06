"""SLURM status parsing via squeue and sacct."""

from __future__ import annotations

from dataclasses import dataclass

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


@dataclass
class JobStatus:
    job_id: str
    name: str | None = None
    state: str | None = None
    elapsed: str | None = None
    exit_code: str | None = None
    partition: str | None = None
    node: str | None = None
    source: str = "unknown"


def parse_squeue_line(text: str, job_id: str) -> JobStatus | None:
    """Parse ``squeue -h -j ID -o '%i|%j|%T|%M|%P|%N'``."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if not parts or parts[0] != job_id:
            continue
        return JobStatus(
            job_id=parts[0],
            name=parts[1] if len(parts) > 1 else None,
            state=parts[2] if len(parts) > 2 else None,
            elapsed=parts[3] if len(parts) > 3 else None,
            partition=parts[4] if len(parts) > 4 else None,
            node=parts[5] if len(parts) > 5 else None,
            source="squeue",
        )
    return None


def parse_sacct_line(text: str, job_id: str) -> JobStatus | None:
    """Parse ``sacct -n -P -j ID --format=JobID,JobName,State,Elapsed,ExitCode,Partition,NodeList``."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if not parts:
            continue
        jid = parts[0].split(".")[0]
        if jid != job_id:
            continue
        # Prefer the batch step if present; otherwise first matching row
        if "." in parts[0] and not parts[0].endswith(".batch") and parts[0] != job_id:
            continue
        return JobStatus(
            job_id=job_id,
            name=parts[1] if len(parts) > 1 else None,
            state=parts[2] if len(parts) > 2 else None,
            elapsed=parts[3] if len(parts) > 3 else None,
            exit_code=parts[4] if len(parts) > 4 else None,
            partition=parts[5] if len(parts) > 5 else None,
            node=parts[6] if len(parts) > 6 else None,
            source="sacct",
        )
    return None


def probe_job(session, job_id: str) -> JobStatus | None:
    """Query squeue then sacct. session is vector_sim.ssh.session.SshSession."""
    import shlex

    sq = session.exec(
        f"squeue -h -j {shlex.quote(job_id)} -o '%i|%j|%T|%M|%P|%N' 2>/dev/null || true",
        category="slurm",
        check=False,
    )
    status = parse_squeue_line(sq.stdout, job_id)
    if status is None:
        sa = session.exec(
            "sacct -n -P -j "
            f"{shlex.quote(job_id)} "
            "--format=JobID,JobName,State,Elapsed,ExitCode,Partition,NodeList 2>/dev/null || true",
            category="slurm",
            check=False,
        )
        status = parse_sacct_line(sa.stdout, job_id)
    if status is not None:
        status.state = normalize_state(status.state)
    return status


def normalize_state(state: str | None) -> str | None:
    if not state:
        return None
    return state.strip().upper().split()[0]


def format_status(status: JobStatus) -> str:
    state = normalize_state(status.state) or "UNKNOWN"
    bits = [
        f"JobID:     {status.job_id}",
        f"State:     {state}",
    ]
    if status.name:
        bits.append(f"Name:      {status.name}")
    if status.elapsed:
        bits.append(f"Elapsed:   {status.elapsed}")
    if status.exit_code:
        bits.append(f"ExitCode:  {status.exit_code}")
    if status.partition:
        bits.append(f"Partition: {status.partition}")
    if status.node:
        bits.append(f"Node:      {status.node}")
    bits.append(f"Source:    {status.source}")
    return "\n".join(bits)
