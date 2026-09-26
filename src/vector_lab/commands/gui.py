"""vector-lab gui — Isaac Sim GUI through Xvfb, localhost x11vnc, and noVNC."""

from __future__ import annotations

import shlex
import time
from collections.abc import Callable
from pathlib import Path

from vector_lab.commands.run import RunCommand
from vector_lab.config.models import JobRecord
from vector_lab.config.store import ConfigStore
from vector_lab.errors import ConfigError, VectorLabError
from vector_lab.exec import CommandRunner
from vector_lab.images.build import require_cluster_profile, resolve_isaaclab
from vector_lab.jobs.args import resolve_job_spec
from vector_lab.jobs.generate import GUI_BOOTSTRAP_NAME, generate_runtime, load_docker_env_base
from vector_lab.jobs.gui import (
    GUI_PYTHON,
    classification_from_log,
    format_gui_failure,
    format_gui_ready,
    gui_result_path,
    parse_gui_result,
    parse_resolution,
)
from vector_lab.jobs.status import TERMINAL_STATES, normalize_state, probe_job
from vector_lab.jobs.submit import parse_sbatch_job_id, submit_remote_job
from vector_lab.jobs.sync import remote_run_path, remote_submit_path, sync_run_directory
from vector_lab.ssh.session import SshSession

GUI_TASK = "Isaac-Sim-GUI"


class GuiCommand:
    def __init__(
        self,
        runner: CommandRunner,
        store: ConfigStore,
        *,
        start_dir: Path | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.runner = runner
        self.store = store
        self.start_dir = Path(start_dir or Path.cwd())
        self.sleep = sleep or time.sleep
        self.clock = clock or time.monotonic

    def run(self, **cli) -> int:
        profile = require_cluster_profile(self.store, cli.get("cluster"))
        if not profile.paths.cache:
            raise ConfigError(
                "cluster profile has no cache directory",
                suggestion="Run: vector-lab setup",
            )
        resolution = parse_resolution(str(cli.get("resolution") or "1920x1080x24"))
        ready_timeout = float(cli.get("ready_timeout") if cli.get("ready_timeout") is not None else 2400)
        poll_interval = float(cli.get("poll_interval") if cli.get("poll_interval") is not None else 15)
        script = str(cli.get("python_executable") or GUI_PYTHON)
        spec = resolve_job_spec(
            profile,
            task=GUI_TASK,
            image_profile=cli.get("image_profile"),
            partition=cli.get("partition"),
            gpu=cli.get("gpu"),
            gpus=cli.get("gpus"),
            gres=cli.get("gres"),
            cpus=cli.get("cpus"),
            memory=cli.get("memory"),
            time=cli.get("time"),
            python_executable=script,
            video=False,
            headless=False,
        )
        repo = resolve_isaaclab(profile.isaaclab_path, self.start_dir)
        session = SshSession(
            profile.ssh_alias,
            self.runner,
            login_shell=profile.scheduler.remote_shell == "login",
        )
        RunCommand(self.runner, self.store, start_dir=self.start_dir)._preflight(profile, session, spec)

        remote_dir = remote_run_path(profile)
        generated = generate_runtime(
            profile,
            docker_env=load_docker_env_base(repo),
            partition=spec.partition,
            gres=spec.resolved_gres,
            cpus=spec.cpus,
            memory=spec.memory,
            time=spec.time,
            output=f"{remote_dir}/slurm-%j.out",
            gui=True,
            python_executable=script,
            resolution=resolution,
        )
        if self.runner.dry_run:
            self.runner.emit("[dry-run] would sync, bootstrap the GUI runtime if needed, and submit")
            self.runner.emit(f"Remote run dir:    {remote_dir}")
            self.runner.emit(f"GUI cache:         {profile.paths.cache.rstrip('/')}/gui")
            self.runner.emit(f"Resolution:        {resolution}")
            self.runner.emit(f"Python executable: {script}")
            self.runner.emit("GUI startup:       Xvfb, Isaac GUI, localhost x11vnc, noVNC/websockify")
            self.runner.emit("Headless run wrappers are not modified.")
            return 0

        sync_run_directory(
            self.runner,
            session,
            local_repo=repo,
            remote_dir=remote_dir,
            generated=generated.as_dict(),
        )
        bootstrap = f"{remote_dir.rstrip('/')}/.vector-lab/generated/{GUI_BOOTSTRAP_NAME}"
        self.runner.emit(
            "Checking the GUI runtime cache (x11vnc and noVNC). "
            "The first run builds them once under the cluster cache."
        )
        session.exec(
            f"bash {shlex.quote(bootstrap)}",
            category="gui-bootstrap",
            suggestion="The login node needs curl, cmake, a C compiler, and X11 headers to build x11vnc once.",
        )

        result = submit_remote_job(
            session,
            remote_run_dir=remote_dir,
            submit_script=remote_submit_path(remote_dir),
            container_name=spec.container_name,
            train_args=[],
        )
        job_id = parse_sbatch_job_id(result.stdout + "\n" + result.stderr)
        if not job_id:
            raise VectorLabError(
                "could not parse SLURM job id from sbatch output",
                category="sbatch",
                stderr=result.stderr,
                suggestion=f"Inspect remote output:\n{result.stdout}",
            )
        slurm_log = f"{remote_dir.rstrip('/')}/slurm-{job_id}.out"
        record = JobRecord.create(
            job_id=job_id,
            cluster=profile.name,
            remote_run_dir=remote_dir,
            slurm_log=slurm_log,
            task=GUI_TASK,
            image_profile=spec.image_profile,
            partition=spec.partition,
            gres=spec.resolved_gres,
            cpus=spec.cpus,
            memory=spec.memory,
            time=spec.resolved_time,
            python_executable=script,
            train_args=[],
            gui=True,
            gui_status="submitted",
        )
        self.store.upsert_job(record)
        self.runner.emit(f"Submitted GUI job {job_id}. Waiting until noVNC is actually ready.")
        ready = self._wait_until_ready(
            session,
            record,
            ready_timeout=ready_timeout,
            poll_interval=poll_interval,
        )
        self._apply_result(record, ready)
        if ready.classification != "GUI_READY":
            raise VectorLabError(
                f"{ready.classification}: {ready.detail or ready.classification}",
                category=ready.classification,
                suggestion=format_gui_failure(ready),
            )
        if ready.novnc_port is None or ready.vnc_port is None or not ready.private_ip:
            ready.classification = "NOVNC_FAILED"
            ready.status = "failed"
            ready.detail = "GUI_READY result did not include the noVNC address"
            self._apply_result(record, ready)
            raise VectorLabError(
                f"NOVNC_FAILED: {ready.detail}",
                category="NOVNC_FAILED",
                suggestion=format_gui_failure(ready),
            )
        if not self._login_node_http_ok(session, ready.private_ip, ready.novnc_port):
            ready.classification = "NOVNC_FAILED"
            ready.status = "failed"
            ready.detail = "login node could not fetch vnc.html from the compute node"
            self._apply_result(record, ready)
            raise VectorLabError(
                format_gui_failure(ready),
                category="NOVNC_FAILED",
                suggestion=format_gui_failure(ready),
            )
        self.runner.emit(format_gui_ready(ready, ssh_alias=profile.ssh_alias).rstrip())
        self.runner.emit("")
        self.runner.emit(f"The job keeps running. Stop it with: vector-lab cancel {job_id}")
        return 0

    def _apply_result(self, record: JobRecord, result) -> None:
        record.gui_status = "ready" if result.classification == "GUI_READY" else "failed"
        record.hostname = result.hostname or record.hostname
        record.private_ip = result.private_ip or record.private_ip
        record.display = result.display or record.display
        record.vnc_port = result.vnc_port
        record.novnc_port = result.novnc_port
        self.store.upsert_job(record)

    def _wait_until_ready(
        self,
        session: SshSession,
        record: JobRecord,
        *,
        ready_timeout: float,
        poll_interval: float,
    ):
        result_path = gui_result_path(record.remote_run_dir)
        deadline = self.clock() + ready_timeout
        while True:
            parsed = parse_gui_result(self._read_remote(session, result_path))
            if parsed is not None:
                if not parsed.job_id:
                    parsed.job_id = record.job_id
                return parsed
            status = probe_job(session, record.job_id)
            state = normalize_state(status.state) if status else None
            if state in TERMINAL_STATES:
                log_text = self._read_remote(session, record.slurm_log)
                classification = classification_from_log(log_text) or "ISAAC_GUI_FAILED"
                from vector_lab.jobs.gui import GuiResult

                return GuiResult(
                    classification=classification,
                    status="failed",
                    job_id=record.job_id,
                    detail=f"SLURM job ended in {state} before GUI_READY",
                    logs={"isaac": record.slurm_log, "cuda": record.slurm_log, "xvfb": record.slurm_log},
                )
            if self.clock() >= deadline:
                raise VectorLabError(
                    f"GUI readiness timed out after {ready_timeout:.0f}s for job {record.job_id}",
                    category="GUI_TIMEOUT",
                    suggestion=(
                        f"SLURM log: {record.slurm_log}\n"
                        f"Result: {result_path}\n"
                        f"Status: vector-lab status {record.job_id}"
                    ),
                )
            self.sleep(poll_interval)

    def _read_remote(self, session: SshSession, path: str) -> str:
        quoted = shlex.quote(path)
        result = session.exec(
            f"if [ -f {quoted} ]; then cat {quoted}; fi",
            category="gui-status",
            check=False,
        )
        return result.stdout or ""

    def _login_node_http_ok(self, session: SshSession, ip: str, port: int) -> bool:
        url = f"http://{ip}:{port}/vnc.html"
        remote = (
            "curl -s -o /dev/null -w '%{http_code}' --noproxy '*' --max-time 5 "
            + shlex.quote(url)
            + " || true"
        )
        for _ in range(10):
            result = session.exec(remote, category="gui-http", check=False)
            lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
            if lines and lines[-1] == "200":
                return True
            self.sleep(3)
        return False
