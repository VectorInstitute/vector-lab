"""Create a cluster profile from local + read-only SSH probes."""

from __future__ import annotations

from pathlib import Path

from vector_sim.cluster.adapters import adapter_for, apply_generic_path_detection
from vector_sim.cluster.detect import infer_home
from vector_sim.cluster.discover import discover_scratch, resolve_remote_shell
from vector_sim.config.models import BUILTIN_PROFILES, ClusterProfile
from vector_sim.config.store import ConfigStore
from vector_sim.exec import CommandRunner
from vector_sim.local.repo import discover_isaac_lab
from vector_sim.ssh.session import SshSession


def build_initial_profile(name: str, ssh_alias: str | None = None) -> ClusterProfile:
    """Load a built-in profile factory by name, or a blank generic SLURM profile.

    Adapter selection uses ``profile.cluster_type``, never ``profile.name``.
    """
    alias = ssh_alias or name
    factory = BUILTIN_PROFILES.get(name.lower())
    if factory is not None:
        profile = factory(ssh_alias=alias)
        profile.name = name
        return profile
    return ClusterProfile(name=name, ssh_alias=alias, cluster_type="slurm")


class InitCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore, *, start_dir: Path | None = None) -> None:
        self.runner = runner
        self.store = store
        self.start_dir = Path(start_dir or Path.cwd())

    def run(self, name: str, *, ssh_alias: str | None = None, isaaclab_path: str | Path | None = None) -> ClusterProfile:
        profile = build_initial_profile(name, ssh_alias=ssh_alias)
        if isaaclab_path:
            profile.isaaclab_path = str(Path(isaaclab_path).expanduser().resolve())
        else:
            try:
                repo = discover_isaac_lab(start=self.start_dir)
                profile.isaaclab_path = str(repo.root)
            except Exception:
                profile.isaaclab_path = None

        session = SshSession(
            profile.ssh_alias,
            self.runner,
            login_shell=profile.scheduler.remote_shell == "login",
        )
        resolved = session.resolve_config()
        if resolved.hostname:
            profile.resolved_host = resolved.hostname

        # Decide the shell before anything else: on sites that set up SLURM, Lmod,
        # and scratch from /etc/profile.d, every later probe is blind without it.
        session, shell_probe = resolve_remote_shell(profile, session, self.runner)
        if shell_probe.detail:
            self.runner.emit(f"[init] remote shell: {profile.scheduler.remote_shell} ({shell_probe.detail})")

        who = session.exec("whoami", category="ssh", check=False)
        if not who.skipped and who.returncode == 0:
            profile.remote_user = who.stdout.strip() or profile.remote_user

        home_probe = session.exec("printf %s \"$HOME\"", category="ssh", check=False)
        home = None
        if not home_probe.skipped and home_probe.returncode == 0:
            home = infer_home(printenv_home=home_probe.stdout)

        scratch, _source = discover_scratch(
            session,
            remote_user=profile.remote_user,
            home_dir=home or profile.home_dir,
        )

        apply_generic_path_detection(profile, home=home, scratch=scratch)
        adapter = adapter_for(profile.cluster_type)
        adapter.apply_unresolved_paths(profile)
        adapter.derive_scratch_paths(profile)

        if self.runner.dry_run:
            self.runner.emit(f"[dry-run] would write profile {self.store.profile_path(profile.name)}")
            return profile

        self.store.save_profile_preserving_overrides(profile)
        self.store.set_active(profile.name)
        return profile
