"""vector-lab build"""

from __future__ import annotations

from pathlib import Path

from vector_lab.config.store import ConfigStore
from vector_lab.exec import CommandRunner
from vector_lab.images.build import BuildPipeline, require_cluster_profile, resolve_isaaclab
from vector_lab.images.state import ImageStateStore


class BuildCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore, *, start_dir: Path | None = None) -> None:
        self.runner = runner
        self.store = store
        self.start_dir = Path(start_dir or Path.cwd())

    def run(self, *, cluster: str | None, image_profile: str, force: bool = False) -> int:
        profile = require_cluster_profile(self.store, cluster)
        repo = resolve_isaaclab(profile.isaaclab_path, self.start_dir)
        outcome = BuildPipeline(self.runner, ImageStateStore(self.store)).run(
            repo=repo,
            image_profile=image_profile,
            force=force,
        )
        self.runner.emit(f"Docker inputs       {outcome.build_input_fingerprint[:12]}…")
        self.runner.emit(f"Docker image        {outcome.docker_image}")
        self.runner.emit(f"Image digest        {outcome.built_image_digest or 'unknown'}")
        self.runner.emit(f"Docker build        {outcome.action} ({outcome.reason})")
        if not self.runner.dry_run:
            profile.fingerprints.build_input_fingerprint = outcome.build_input_fingerprint
            profile.fingerprints.built_image_digest = outcome.built_image_digest
            self.store.save_profile(profile)
        return 0
