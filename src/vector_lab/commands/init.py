"""Create a cluster profile from local + read-only SSH probes."""

from __future__ import annotations

from pathlib import Path

from vector_lab.cluster.adapters import adapter_for, apply_generic_path_detection
from vector_lab.cluster.detect import infer_home, infer_scratch, parse_env_assignments
from vector_lab.config.models import ClusterProfile, bonecho_defaults
from vector_lab.config.store import ConfigStore
from vector_lab.exec import CommandRunner
from vector_lab.local.repo import discover_isaac_lab
from vector_lab.ssh.session import SshSession


def build_initial_profile(name: str, ssh_alias: str | None = None) -> ClusterProfile:
    alias = ssh_alias or name
    if name.lower() == "bonecho":
        profile = bonecho_defaults(ssh_alias=alias)
        profile.name = name
        return profile
    return ClusterProfile(name=name, ssh_alias=alias)


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

        who = session.exec("whoami", category="ssh", check=False)
        if not who.skipped and who.returncode == 0:
            profile.remote_user = who.stdout.strip() or profile.remote_user

        home_probe = session.exec("printf %s \"$HOME\"", category="ssh", check=False)
        env_probe = session.exec(
            "env | grep -E '^(SCRATCH|SCRATCHDIR|SCRATCH_DIR|CSCRATCH)=' || true",
            category="ssh",
            check=False,
        )
        home = None
        scratch = None
        if not home_probe.skipped and home_probe.returncode == 0:
            home = infer_home(printenv_home=home_probe.stdout)
        if not env_probe.skipped and env_probe.returncode == 0:
            env = parse_env_assignments(env_probe.stdout)
            scratch, _source = infer_scratch(env=env)

        apply_generic_path_detection(profile, home=home, scratch=scratch)
        adapter = adapter_for(profile.name)
        adapter.apply_unresolved_paths(profile)
        adapter.derive_scratch_paths(profile)

        if self.runner.dry_run:
            self.runner.emit(f"[dry-run] would write profile {self.store.profile_path(profile.name)}")
            return profile

        self.store.save_profile_preserving_overrides(profile)
        self.store.set_active(profile.name)
        return profile
