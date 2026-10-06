"""SLURM job ID parsing and submission helpers."""

from __future__ import annotations

import re
import shlex

from vector_sim.exec import CommandResult
from vector_sim.ssh.session import SshSession

_JOB_ID = re.compile(r"Submitted batch job\s+(\d+)", re.IGNORECASE)


def parse_sbatch_job_id(text: str) -> str | None:
    match = _JOB_ID.search(text or "")
    return match.group(1) if match else None


def submit_remote_job(
    session: SshSession,
    *,
    remote_run_dir: str,
    submit_script: str,
    container_name: str,
    train_args: list[str],
) -> CommandResult:
    """Invoke generated submit script on the login node under the run directory."""
    quoted_args = " ".join(shlex.quote(a) for a in train_args)
    # submit_job_slurm.sh "$RUN_DIR" "isaac-lab-base" <train args...>
    remote_cmd = (
        f"cd {shlex.quote(remote_run_dir)} && "
        f"bash {shlex.quote(submit_script)} "
        f"{shlex.quote(remote_run_dir)} {shlex.quote(container_name)} {quoted_args}"
    )
    return session.exec(
        remote_cmd,
        category="sbatch",
        suggestion="Ensure SLURM is available via login shell and the container was pushed.",
    )
