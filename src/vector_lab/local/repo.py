"""Locate and validate an Isaac Lab checkout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vector_lab.compat import ISAACLAB_COMMIT, ISAACLAB_REMOTE
from vector_lab.errors import DiscoveryError

MARKERS = (
    "docker/container.py",
    "docker/cluster/cluster_interface.sh",
    "source/isaaclab",
    "isaaclab.sh",
)


@dataclass
class IsaacLabRepo:
    root: Path

    def validate(self) -> list[str]:
        missing = [marker for marker in MARKERS if not (self.root / marker).exists()]
        return missing


def is_isaac_lab_repo(path: Path) -> bool:
    root = path.resolve()
    if not root.is_dir():
        return False
    return not IsaacLabRepo(root).validate()


def discover_isaac_lab(
    *,
    configured: str | Path | None = None,
    start: Path | None = None,
) -> IsaacLabRepo:
    """Search configured path, cwd, then parents. Does not clone."""
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.exists():
            raise DiscoveryError(
                f"configured Isaac Lab path does not exist: {path}",
                suggestion=(
                    "Run: vector-lab onboard bonecho  "
                    f"(clones pinned Isaac Lab {ISAACLAB_COMMIT[:12]} from {ISAACLAB_REMOTE})"
                ),
            )
        missing = IsaacLabRepo(path).validate()
        if missing:
            raise DiscoveryError(
                f"{path} does not look like Isaac Lab (missing {', '.join(missing)})",
                suggestion=(
                    "Point --isaaclab at a full Isaac Lab clone, or run: vector-lab onboard bonecho"
                ),
            )
        return IsaacLabRepo(path)

    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if is_isaac_lab_repo(candidate):
            return IsaacLabRepo(candidate)

    raise DiscoveryError(
        f"could not find an Isaac Lab repository from {here}",
        suggestion=(
            "Isaac Lab not found. Run: vector-lab onboard bonecho  "
            f"(clones {ISAACLAB_REMOTE} at {ISAACLAB_COMMIT[:12]}), "
            "or pass --isaaclab /path/to/IsaacLab"
        ),
    )
