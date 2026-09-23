"""Materialize a reproducible Isaac Lab cluster setup (no image build/upload)."""

from __future__ import annotations

import shlex
from pathlib import Path

from vector_lab.cluster.adapters import adapter_for
from vector_lab.cluster.discover import (
    apply_discovery,
    discover_apptainer,
    discover_partitions,
    discover_scratch,
    resolve_remote_shell,
)
from vector_lab.commands.init import InitCommand
from vector_lab.config.models import ClusterProfile
from vector_lab.config.store import ConfigStore
from vector_lab.errors import ConfigError
from vector_lab.exec import CommandRunner
from vector_lab.jobs.generate import (
    generate_runtime,
    load_docker_env_base,
    persistent_remote_dirs,
    write_runtime,
)
from vector_lab.local.repo import discover_isaac_lab
from vector_lab.ssh.session import SshSession


class SetupCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore, *, start_dir: Path | None = None) -> None:
        self.runner = runner
        self.store = store
        self.start_dir = Path(start_dir or Path.cwd())

    def run(
        self,
        name: str | None = None,
        *,
        isaaclab: str | Path | None = None,
        ssh_alias: str | None = None,
        create_remote_dirs: bool = True,
    ) -> ClusterProfile:
        profile_name = name or self.store.get_active()
        if not profile_name:
            raise ConfigError(
                "no cluster profile specified",
                suggestion="Run: vector-lab setup bonecho --isaaclab /path/to/IsaacLab",
            )
        if not self.store.profile_path(profile_name).is_file():
            self.runner.emit(f"[setup] profile '{profile_name}' missing; running init")
            profile = InitCommand(self.runner, self.store, start_dir=self.start_dir).run(
                profile_name,
                ssh_alias=ssh_alias,
                isaaclab_path=isaaclab,
            )
        else:
            profile = self.store.load_profile(profile_name)
            if ssh_alias:
                profile.ssh_alias = ssh_alias
            if isaaclab:
                profile.isaaclab_path = str(Path(isaaclab).expanduser().resolve())
        if isaaclab:
            profile.isaaclab_path = str(Path(isaaclab).expanduser().resolve())
        elif not profile.isaaclab_path:
            try:
                profile.isaaclab_path = str(discover_isaac_lab(start=self.start_dir).root)
            except Exception:
                pass

        session = SshSession(
            profile.ssh_alias,
            self.runner,
            login_shell=profile.scheduler.remote_shell == "login",
        )
        session, shell_probe = resolve_remote_shell(profile, session, self.runner)
        if shell_probe.detail:
            self.runner.emit(
                f"[setup] remote shell: {profile.scheduler.remote_shell} ({shell_probe.detail})"
            )

        if not profile.scratch_dir:
            found, source = discover_scratch(
                session,
                remote_user=profile.remote_user,
                home_dir=profile.home_dir,
            )
            if found:
                profile.scratch_dir = found
                self.runner.emit(f"[setup] scratch discovered via {source}: {found}")

        adapter = adapter_for(profile.cluster_type)
        adapter.apply_unresolved_paths(profile)
        adapter.derive_scratch_paths(profile)

        partitions = discover_partitions(session)
        apptainer = discover_apptainer(session, module=profile.apptainer.module)
        apply_discovery(profile, partitions, apptainer)

        docker_env = load_docker_env_base(profile.isaaclab_path)
        generated = generate_runtime(profile, docker_env=docker_env)
        written = write_runtime(generated, self.store.generated_dir)
        extra_root = None
        if profile.isaaclab_path:
            extra_root = Path(profile.isaaclab_path) / ".vector-lab" / "generated"
            write_runtime(generated, extra_root)

        if create_remote_dirs:
            self._create_remote_dirs(session, profile)

        if not self.runner.dry_run:
            self.store.save_profile_preserving_overrides(profile)
            self.store.set_active(profile.name)

        self.runner.emit(f"[setup] profile: {profile.name}")
        self.runner.emit(f"[setup] ssh alias: {profile.ssh_alias}")
        self.runner.emit(f"[setup] remote user: {profile.remote_user or '(unknown)'}")
        self.runner.emit(f"[setup] scratch: {profile.scratch_dir or '(unresolved)'}")
        self.runner.emit(f"[setup] isaac lab: {profile.isaaclab_path or '(not found)'}")
        for label, path in written.items():
            self.runner.emit(f"[setup] generated {label}: {path}")
        if extra_root:
            self.runner.emit(f"[setup] also copied generated files to {extra_root}")
        if partitions:
            summary = ", ".join(
                f"{p.name}({','.join(p.gpu_types) or p.gres or '?'})" for p in partitions[:12]
            )
            self.runner.emit(f"[setup] GPU partitions: {summary}")
        self.runner.emit("[setup] next: vector-lab doctor && vector-lab deploy")
        return profile

    def _create_remote_dirs(self, session: SshSession, profile: ClusterProfile) -> None:
        dirs = persistent_remote_dirs(profile)
        if not dirs:
            self.runner.emit("[setup] no remote paths to create yet (user/scratch unresolved)")
            return
        quoted = " ".join(shlex.quote(d) for d in dirs)
        result = session.exec(
            f"mkdir -p {quoted}",
            category="remote-fs",
            check=False,
            suggestion="Verify SSH access and scratch permissions.",
        )
        if result.skipped:
            self.runner.emit(f"[dry-run] would mkdir -p {quoted}")
            return
        if result.returncode != 0:
            self.runner.emit(f"[setup] WARNING: remote mkdir failed: {result.stderr.strip()}")
        else:
            self.runner.emit(f"[setup] remote directories ready: {', '.join(dirs)}")
