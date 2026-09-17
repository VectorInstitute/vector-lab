"""vector-lab push"""

from __future__ import annotations

from pathlib import Path

from vector_lab.config.store import ConfigStore
from vector_lab.exec import CommandRunner
from vector_lab.images.build import require_cluster_profile, resolve_isaaclab
from vector_lab.images.push import PushPipeline, format_pipeline_report
from vector_lab.images.state import ImageStateStore


class PushCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore, *, start_dir: Path | None = None) -> None:
        self.runner = runner
        self.store = store
        self.start_dir = Path(start_dir or Path.cwd())

    def run(
        self,
        *,
        cluster: str | None,
        image_profile: str,
        force: bool = False,
        force_convert: bool = False,
        force_upload: bool = False,
    ) -> int:
        profile = require_cluster_profile(self.store, cluster)
        repo = resolve_isaaclab(profile.isaaclab_path, self.start_dir)
        outcome = PushPipeline(self.runner, ImageStateStore(self.store)).run(
            profile=profile,
            repo=repo,
            image_profile=image_profile,
            force=force,
            force_convert=force_convert,
            force_upload=force_upload,
        )
        self.runner.emit(format_pipeline_report(outcome))
        if not self.runner.dry_run:
            self.store.save_profile(profile)
        return 0
