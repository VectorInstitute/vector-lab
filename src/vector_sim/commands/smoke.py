"""vector-sim smoke-test — product-level Cartpole acceptance workload."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vector_sim.commands.run import RunCommand
from vector_sim.config.store import ConfigStore
from vector_sim.config.store import ConfigStore
from vector_sim.exec import CommandRunner
from vector_sim.images.build import require_cluster_profile
from vector_sim.jobs.status import TERMINAL_STATES, JobStatus, normalize_state, probe_job
from vector_sim.jobs.videos import find_videos_command, parse_find_videos
from vector_sim.ssh.session import SshSession

SMOKE_TASK = "Isaac-Cartpole-v0"


@dataclass
class SmokeResult:
    passed: bool
    phase: str
    job_id: str | None = None
    state: str | None = None
    exit_code: str | None = None
    video_ok: bool | None = None
    detail: str = ""


def smoke_passed(result: SmokeResult, *, require_video: bool = False) -> bool:
    if not result.passed:
        return False
    if require_video and not result.video_ok:
        return False
    return True


def evaluate_job(status: JobStatus, *, require_zero_exit: bool = True) -> SmokeResult:
    state = normalize_state(status.state) or "UNKNOWN"
    exit_code = status.exit_code
    if state not in TERMINAL_STATES:
        return SmokeResult(False, "RUNNING", status.job_id, state, exit_code, detail="job still running")
    if state != "COMPLETED":
        return SmokeResult(False, "FAILED", status.job_id, state, exit_code, detail=f"terminal state {state}")
    if require_zero_exit and exit_code not in {None, "0:0", "0:0:0"}:
        # sacct uses 0:0 for success
        if not str(exit_code).startswith("0:0"):
            return SmokeResult(False, "FAILED", status.job_id, state, exit_code, detail=f"ExitCode {exit_code}")
    return SmokeResult(True, "COMPLETED", status.job_id, state, exit_code or "0:0", detail="COMPLETED / 0:0")


class SmokeTestCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore, *, start_dir: Path | None = None) -> None:
        self.runner = runner
        self.store = store
        self.start_dir = Path(start_dir or Path.cwd())

    def run(
        self,
        *,
        cluster: str | None,
        video: bool = False,
        wait: bool = True,
        timeout: float = 3600,
        poll_interval: float = 15,
        max_iterations: int = 50,
        sleep: Callable[[float], None] | None = None,
    ) -> int:
        profile = require_cluster_profile(self.store, cluster)
        extra = ["--max_iterations", str(max_iterations)]
        before = {j.job_id for j in self.store.load_jobs()}
        code = RunCommand(self.runner, self.store, start_dir=self.start_dir).run(
            cluster=cluster,
            task=SMOKE_TASK,
            image_profile=None,
            video=video,
            video_length=200,
            video_interval=1000,
            headless=True,
            extra_args=extra,
        )
        if code != 0:
            self._fail(SmokeResult(False, "SUBMIT", detail="run command failed"))
            return 1
        if self.runner.dry_run:
            self.runner.emit("[dry-run] smoke-test would submit Cartpole and wait for COMPLETED / 0:0")
            return 0
        jobs = [j for j in self.store.load_jobs() if j.job_id not in before]
        if not jobs:
            jobs = self.store.load_jobs()[-1:]
        if not jobs:
            self._fail(SmokeResult(False, "SUBMIT", detail="no job id recorded"))
            return 1
        record = jobs[-1]
        session = SshSession(
            profile.ssh_alias,
            self.runner,
            login_shell=profile.scheduler.remote_shell == "login",
        )
        if not wait:
            self.runner.emit(f"Submitted {record.job_id}; not waiting (--no-wait)")
            return 0
        waiter = sleep or time.sleep
        deadline = time.monotonic() + timeout
        status = None
        while time.monotonic() < deadline:
            status = probe_job(session, record.job_id)
            if status and (normalize_state(status.state) in TERMINAL_STATES):
                break
            waiter(poll_interval)
        if status is None:
            self._fail(SmokeResult(False, "TIMEOUT", job_id=record.job_id, detail="job not found in squeue/sacct"))
            return 1
        result = evaluate_job(status)
        result.job_id = record.job_id
        if video and result.passed and profile.paths.logs:
            listing = session.exec(find_videos_command(profile.paths.logs), category="ssh", check=False)
            videos = parse_find_videos(listing.stdout)
            result.video_ok = bool(videos)
            if not videos:
                result.passed = False
                result.phase = "VIDEO"
                result.detail = "COMPLETED but no MP4 under persistent logs"
        self._emit(result)
        return 0 if result.passed else 1

    def _fail(self, result: SmokeResult) -> None:
        self._emit(result)

    def _emit(self, result: SmokeResult) -> None:
        verdict = "PASS" if result.passed else "FAIL"
        self.runner.emit(f"smoke-test           {verdict}")
        self.runner.emit(f"phase                {result.phase}")
        if result.job_id:
            self.runner.emit(f"job                  {result.job_id}")
        if result.state:
            self.runner.emit(f"state                {result.state}")
        if result.exit_code:
            self.runner.emit(f"ExitCode             {result.exit_code}")
        if result.video_ok is not None:
            self.runner.emit(f"video                {'OK' if result.video_ok else 'MISSING'}")
        if result.detail:
            self.runner.emit(result.detail)
