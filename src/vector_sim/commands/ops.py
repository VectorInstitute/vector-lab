"""vector-sim status / logs / cancel."""

from __future__ import annotations

import shlex
from pathlib import Path

from vector_sim.config.store import ConfigStore
from vector_sim.errors import ConfigError, VectorSimError
from vector_sim.exec import CommandRunner
from vector_sim.images.build import require_cluster_profile
from vector_sim.jobs.gui import format_saved_gui_connection
from vector_sim.jobs.status import format_status, probe_job
from vector_sim.ssh.session import SshSession


def _session(runner: CommandRunner, store: ConfigStore, cluster: str | None) -> tuple:
    profile = require_cluster_profile(store, cluster)
    session = SshSession(
        profile.ssh_alias,
        runner,
        login_shell=profile.scheduler.remote_shell == "login",
    )
    return profile, session


class StatusCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore) -> None:
        self.runner = runner
        self.store = store

    def run(self, *, cluster: str | None, job_id: str | None) -> int:
        profile, session = _session(self.runner, self.store, cluster)
        if job_id:
            return self._one(session, job_id, profile.ssh_alias)
        jobs = sorted(self.store.load_jobs(), key=lambda j: j.submitted_at, reverse=True)
        if not jobs:
            self.runner.emit(f"No local job metadata for cluster {profile.name}.")
            return 0
        for job in jobs[:10]:
            self.runner.emit(f"— {job.job_id}  {job.task}  {job.submitted_at}")
            self._one(session, job.job_id, profile.ssh_alias)
            self.runner.emit("")
        return 0

    def _one(self, session: SshSession, job_id: str, ssh_alias: str) -> int:
        if self.runner.dry_run:
            self.runner.emit(f"[dry-run] would query squeue/sacct for {job_id}")
            return 0
        status = probe_job(session, job_id)
        if status is None:
            self.runner.emit(f"Job {job_id}: not found in squeue or sacct")
            return 1
        self.runner.emit(format_status(status))
        extra = format_saved_gui_connection(self.store.find_job(job_id), ssh_alias=ssh_alias)
        if extra:
            self.runner.emit("")
            self.runner.emit(extra)
        return 0


class LogsCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore) -> None:
        self.runner = runner
        self.store = store

    def run(self, *, cluster: str | None, job_id: str, follow: bool = False) -> int:
        profile, session = _session(self.runner, self.store, cluster)
        record = self.store.find_job(job_id)
        if record is None:
            raise ConfigError(
                f"no local metadata for job {job_id}",
                suggestion="Submit with vector-sim run, or pass a known SLURM log path later.",
            )
        log_path = record.slurm_log
        if self.runner.dry_run:
            cmd = f"tail -n 100 {log_path}" if not follow else f"tail -n +1 -f {log_path}"
            self.runner.emit(f"[dry-run] would: ssh {profile.ssh_alias} '{cmd}'")
            return 0
        exists = session.exec(
            f"test -f {shlex.quote(log_path)} && echo LOG_OK",
            category="ssh",
            check=False,
        )
        if "LOG_OK" not in exists.stdout:
            self.runner.emit(
                f"SLURM log not available yet (job may be PENDING):\n  {log_path}"
            )
            return 0
        if follow:
            # Interactive follow: use non-capturing subprocess via runner.execute path.
            # For BatchMode SSH, stream with tail -f until local Ctrl+C.
            remote = f"tail -n +1 -f {shlex.quote(log_path)}"
            args = session.ssh_prefix() + [
                f"bash -l -c {shlex.quote(remote)}"
                if profile.scheduler.remote_shell == "login"
                else remote
            ]
            self.runner.emit(f"Following {log_path} (Ctrl+C stops follow, job keeps running)")
            try:
                import subprocess

                proc = subprocess.Popen(args)
                proc.wait()
            except KeyboardInterrupt:
                self.runner.emit("\nStopped following logs (job was not cancelled).")
            return 0
        result = session.exec(
            f"tail -n 200 {shlex.quote(log_path)}",
            category="ssh",
        )
        self.runner.emit(result.stdout.rstrip())
        return 0


class CancelCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore) -> None:
        self.runner = runner
        self.store = store

    def run(self, *, cluster: str | None, job_id: str) -> int:
        if not job_id.isdigit():
            raise VectorSimError("job id must be numeric", category="cancel")
        profile, session = _session(self.runner, self.store, cluster)
        if self.runner.dry_run:
            self.runner.emit(f"[dry-run] would scancel {job_id} on {profile.ssh_alias}")
            return 0
        session.exec(
            f"scancel {shlex.quote(job_id)}",
            category="scancel",
            suggestion="Verify the job id with vector-sim status",
        )
        self.runner.emit(f"Cancelled job {job_id} on {profile.name}")
        self.runner.emit("Remote run directories and artifacts were not deleted.")
        return 0
