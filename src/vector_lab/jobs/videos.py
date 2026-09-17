"""Remote MP4 discovery and pull helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RemoteVideo:
    path: str
    size: int | None = None
    mtime: str | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name


def parse_find_videos(text: str) -> list[RemoteVideo]:
    """Parse ``find … -printf '%T@ %s %p\\n'`` sorted newest first."""
    videos: list[RemoteVideo] = []
    rows: list[tuple[float, RemoteVideo]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or not line.endswith(".mp4"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            # fallback: path only
            videos.append(RemoteVideo(path=line))
            continue
        try:
            mtime = float(parts[0])
            size = int(parts[1])
        except ValueError:
            videos.append(RemoteVideo(path=line))
            continue
        rows.append((mtime, RemoteVideo(path=parts[2], size=size, mtime=parts[0])))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in rows] + videos


def find_videos_command(logs_dir: str) -> str:
    return (
        f"if [ -d {shlex.quote(logs_dir)} ]; then "
        f"find {shlex.quote(logs_dir)} -type f -name '*.mp4' "
        f"-printf '%T@ %s %p\\n' 2>/dev/null | sort -nr; "
        f"fi"
    )


def pull_video_args(ssh_alias: str, remote_path: str, local_path: Path) -> list[str]:
    return [
        "rsync",
        "-avh",
        "--progress",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=15",
        f"{ssh_alias}:{remote_path}",
        str(local_path),
    ]


def format_video_list(videos: list[RemoteVideo], *, limit: int = 20) -> str:
    if not videos:
        return "No MP4 files found under the configured logs directory."
    lines = []
    for idx, video in enumerate(videos[:limit], start=1):
        size = f"{video.size} B" if video.size is not None else "?"
        lines.append(f"{idx}. {video.name}  ({size})")
        lines.append(f"   {video.path}")
    if len(videos) > limit:
        lines.append(f"... and {len(videos) - limit} more")
    return "\n".join(lines)
