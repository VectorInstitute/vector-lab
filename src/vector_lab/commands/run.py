"""vector-lab run — sync source and submit a Vector SLURM job."""

from __future__ import annotations

import shlex
from pathlib import Path

from vector_lab.config.models import JobRecord
from vector_lab.config.store import ConfigStore
from vector_lab.errors import ConfigError, VectorLabError
from vector_lab.exec import CommandRunner
from vector_lab.images.build import require_cluster_profile, resolve_isaaclab
from vector_lab.images.naming import remote_tar_name
from vector_lab.jobs.args import JobSpec, expand_train_args, resolve_job_spec
from vector_lab.jobs.generate import (
    ENV_NAME,
    RUNNER_NAME,
    SUBMIT_NAME,
    generate_runtime,
    load_docker_env_base,
)
from vector_lab.jobs.submit import parse_sbatch_job_id, submit_remote_job
from vector_lab.jobs.sync import remote_run_path, remote_submit_path, sync_run_directory
from vector_lab.ssh.session import SshSession


class RunCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore, *, start_dir: Path | None = None) -> None:
        self.runner = runner
        self.store = store
        self.start_dir = Path(start_dir or Path.cwd())

    def run(self, **cli) -> int:
        profile = require_cluster_profile(self.store, cli.get("cluster"))
        spec = resolve_job_spec(profile, **{k: v for k, v in cli.items() if k != "cluster"})
        repo = resolve_isaaclab(profile.isaaclab_path, self.start_dir)
        session = SshSession(
            profile.ssh_alias,
            self.runner,
            login_shell=profile.scheduler.remote_shell == "login",
        )
        self._preflight(profile, session, spec)

        remote_dir = remote_run_path(profile)
        train_args = expand_train_args(spec)
        docker_env = load_docker_env_base(repo)
        # Per-run submit with resource overrides; logs land in the run dir by cwd.
        generated = generate_runtime(
            profile,
            docker_env=docker_env,
            partition=spec.partition,
            gres=spec.resolved_gres,
            cpus=spec.cpus,
            memory=spec.memory,
            time=spec.time,
        ).as_dict()

        planned = sync_run_directory(
            self.runner,
            session,
            local_repo=repo,
            remote_dir=remote_dir,
            generated=generated,
        )

        submit_script = remote_submit_path(remote_dir)
        if self.runner.dry_run:
            self._print_dry_run(spec, remote_dir, train_args, planned, submit_script)
            return 0

        result = submit_remote_job(
            session,
            remote_run_dir=remote_dir,
            submit_script=submit_script,
            container_name=spec.container_name,
            train_args=train_args,
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
            task=spec.task,
            image_profile=spec.image_profile,
            partition=spec.partition,
            gres=spec.resolved_gres,
            cpus=spec.cpus,
            memory=spec.memory,
            time=spec.resolved_time,
            video=spec.video,
            python_executable=spec.python_executable,
            train_args=train_args,
        )
        self.store.upsert_job(record)
        self._print_submitted(record)
        return 0

    def _preflight(self, profile, session: SshSession, spec: JobSpec) -> None:
        if not profile.scratch_dir or not profile.paths.containers:
            raise ConfigError(
                "cluster profile is incomplete",
                suggestion="Run: vector-lab onboard bonecho",
            )
        # Generated wrappers should exist locally from setup; run regenerates per-job.
        if not self.store.generated_dir.is_dir() and not self.runner.dry_run:
            self.runner.emit("[run] WARNING: .vector-lab/generated missing; regenerating for this job")

        remote_tar = f"{profile.paths.containers.rstrip('/')}/{remote_tar_name(spec.image_profile)}"
        probe = session.exec(
            f"test -f {shlex.quote(remote_tar)} && echo CONTAINER_OK",
            category="ssh-preflight",
            check=False,
        )
        if not probe.skipped and ("CONTAINER_OK" not in probe.stdout):
            combined = f"{probe.stdout}\n{probe.stderr}"
            if "Permission denied" in combined or "keyboard-interactive" in combined:
                raise VectorLabError(
                    "SSH to the cluster was denied before the container could be checked",
                    category="ssh",
                    stderr=probe.stderr,
                    suggestion=f"The login session expired. Run: vector-lab auth {profile.ssh_alias}",
                )
            raise VectorLabError(
                f"container not deployed at {remote_tar}",
                category="preflight",
                suggestion="Container is not on the cluster yet. Run: vector-lab deploy",
            )
        if not probe.skipped:
            slurm = session.exec("command -v sbatch", category="slurm", check=False)
            if slurm.returncode != 0 or not slurm.stdout.strip():
                raise VectorLabError(
                    "sbatch not available in the remote login shell",
                    category="preflight",
                    suggestion=(
                        "SLURM is not available on the cluster login. "
                        "Run: vector-lab doctor. If SSH expired: vector-lab auth bonecho"
                    ),
                )

    def _print_dry_run(
        self,
        spec: JobSpec,
        remote_dir: str,
        train_args: list[str],
        planned: list[str],
        submit_script: str,
    ) -> None:
        self.runner.emit("[dry-run] would sync / submit (no remote mutations)")
        self.runner.emit(f"Task:              {spec.task}")
        self.runner.emit(f"Image profile:     {spec.image_profile}")
        self.runner.emit(f"Partition:         {spec.partition}")
        self.runner.emit(f"GRES:              {spec.resolved_gres}")
        self.runner.emit(f"CPUs:             {spec.cpus}")
        self.runner.emit(f"Memory:            {spec.memory}")
        self.runner.emit(f"Time:              {spec.resolved_time}")
        self.runner.emit(f"Python executable: {spec.python_executable}")
        self.runner.emit(f"Train args:        {shlex.join(train_args)}")
        self.runner.emit(f"Remote run dir:    {remote_dir}")
        self.runner.emit(f"Generated submit:  {submit_script}")
        self.runner.emit(f"Generated files:   {ENV_NAME}, {SUBMIT_NAME}, {RUNNER_NAME}")
        for item in planned:
            self.runner.emit(f"plan: {item}")
        self.runner.emit(
            "submit: "
            f"bash {submit_script} {remote_dir} {spec.container_name} {shlex.join(train_args)}"
        )

    def _print_submitted(self, record: JobRecord) -> None:
        self.runner.emit(f"Job ID:       {record.job_id}")
        self.runner.emit(f"Cluster:      {record.cluster}")
        self.runner.emit(f"Task:         {record.task}")
        self.runner.emit(f"Remote run:   {record.remote_run_dir}")
        self.runner.emit(f"SLURM log:    {record.slurm_log}")
        self.runner.emit("")
        self.runner.emit("View status:")
        self.runner.emit(f"  vector-lab status {record.job_id}")
        self.runner.emit("")
        self.runner.emit("Follow log:")
        self.runner.emit(f"  vector-lab logs {record.job_id} --follow")
