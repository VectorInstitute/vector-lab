"""vector-lab deploy — compose build → convert → push."""

from __future__ import annotations

from pathlib import Path

from vector_lab.commands.push import PushCommand
from vector_lab.config.store import ConfigStore
from vector_lab.exec import CommandRunner
from vector_lab.images.build import require_cluster_profile, resolve_isaaclab
from vector_lab.images.push import PushPipeline, format_pipeline_report
from vector_lab.images.state import ImageStateStore


class DeployCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore, *, start_dir: Path | None = None) -> None:
        self.runner = runner
        self.store = store
        self.start_dir = Path(start_dir or Path.cwd())

    def run(
        self,
        *,
        cluster: str | None,
        image_profile: str = "base",
        plan: bool = False,
        force: bool = False,
        force_convert: bool = False,
        force_upload: bool = False,
    ) -> int:
        profile = require_cluster_profile(self.store, cluster)
        repo = resolve_isaaclab(profile.isaaclab_path, self.start_dir)
        pipeline = PushPipeline(self.runner, ImageStateStore(self.store))
        preview = pipeline.preview(
            profile=profile,
            repo=repo,
            image_profile=image_profile,
            force=force,
            force_convert=force_convert,
            force_upload=force_upload,
        )
        self.runner.emit("Deploy plan (build → convert → push):")
        self.runner.emit(format_pipeline_report(preview))
        if plan or self.runner.dry_run:
            self.runner.emit("")
            self.runner.emit("plan only; no Docker build, conversion, or upload")
            return 0
        self.runner.emit("")
        self.runner.emit("Executing with existing cache logic...")
        return PushCommand(self.runner, self.store, start_dir=self.start_dir).run(
            cluster=cluster,
            image_profile=image_profile,
            force=force,
            force_convert=force_convert,
            force_upload=force_upload,
        )
