"""vector-lab videos / pull-video / shell."""

from __future__ import annotations

import shlex
from pathlib import Path

from vector_lab.config.store import ConfigStore
from vector_lab.errors import ConfigError, VectorLabError
from vector_lab.exec import CommandRunner
from vector_lab.images.build import require_cluster_profile
from vector_lab.jobs.videos import (
    find_videos_command,
    format_video_list,
    parse_find_videos,
    pull_video_args,
)
from vector_lab.ssh.session import SshSession


class VideosCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore) -> None:
        self.runner = runner
        self.store = store

    def run(self, *, cluster: str | None, job_id: str | None = None) -> int:
        profile = require_cluster_profile(self.store, cluster)
        logs_dir = profile.paths.logs
        if not logs_dir:
            raise ConfigError("profile.paths.logs is unset", suggestion="Run vector-lab setup")
        if job_id:
            record = self.store.find_job(job_id)
            if record is None:
                raise ConfigError(f"no local metadata for job {job_id}")
            # Job-scoped search still under persistent logs (training writes there)
            self.runner.emit(f"Searching persistent logs for job {job_id} task={record.task}")
        session = SshSession(
            profile.ssh_alias,
            self.runner,
            login_shell=profile.scheduler.remote_shell == "login",
        )
        if self.runner.dry_run:
            self.runner.emit(f"[dry-run] would find MP4s under {logs_dir}")
            return 0
        result = session.exec(find_videos_command(logs_dir), category="ssh", check=False)
        videos = parse_find_videos(result.stdout)
        self.runner.emit(format_video_list(videos))
        return 0


class PullVideoCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore) -> None:
        self.runner = runner
        self.store = store

    def run(
        self,
        *,
        cluster: str | None,
        latest: bool = False,
        job_id: str | None = None,
        remote_video: str | None = None,
        dest_dir: Path | None = None,
    ) -> int:
        profile = require_cluster_profile(self.store, cluster)
        logs_dir = profile.paths.logs
        if not logs_dir:
            raise ConfigError("profile.paths.logs is unset")
        session = SshSession(
            profile.ssh_alias,
            self.runner,
            login_shell=profile.scheduler.remote_shell == "login",
        )
        if remote_video:
            path = remote_video
        else:
            if self.runner.dry_run:
                self.runner.emit(f"[dry-run] would discover and pull latest MP4 from {logs_dir}")
                return 0
            result = session.exec(find_videos_command(logs_dir), category="ssh", check=False)
            videos = parse_find_videos(result.stdout)
            if job_id:
                record = self.store.find_job(job_id)
                if record:
                    task_hint = record.task.lower().replace("isaac-", "").split("-v")[0]
                    filtered = [v for v in videos if task_hint in v.path.lower()]
                    videos = filtered or videos
            if not videos:
                raise VectorLabError("no remote MP4 videos found", category="videos")
            if not latest and not job_id:
                raise ConfigError("specify --latest, --job, or a remote video path")
            path = videos[0].path

        local_dir = Path(dest_dir or Path.cwd())
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / Path(path).name
        args = pull_video_args(profile.ssh_alias, path, local_path)
        if self.runner.dry_run:
            self.runner.emit(f"[dry-run] {self.runner.format_args(args)}")
            return 0
        self.runner.run(args, category="rsync-video")
        self.runner.emit(f"Downloaded: {local_path}")
        return 0


class ShellCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore) -> None:
        self.runner = runner
        self.store = store

    def run(self, *, cluster: str | None) -> int:
        profile = require_cluster_profile(self.store, cluster)
        if self.runner.dry_run:
            self.runner.emit(f"[dry-run] would exec: ssh {profile.ssh_alias}")
            return 0
        import os
        import subprocess

        self.runner.emit(f"Opening SSH session to {profile.ssh_alias} (login node)")
        return subprocess.call(["ssh", profile.ssh_alias], env=os.environ.copy())
