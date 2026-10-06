"""Content-addressed artifact naming and safe cleanup."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from vector_sim.errors import VectorSimError


def docker_image_ref(image_profile: str) -> str:
    profile = "base" if image_profile in {"", "isaaclab"} else image_profile
    return f"isaac-lab-{profile}:latest"


def image_stem(image_profile: str) -> str:
    return docker_image_ref(image_profile).split(":")[0]


def short_digest(digest: str, length: int = 12) -> str:
    hexpart = digest.replace("sha256:", "").strip()
    return hexpart[:length] if hexpart else "unknown"


def artifact_filename(image_profile: str, digest: str) -> str:
    return f"{image_stem(image_profile)}-{short_digest(digest)}.tar"


def remote_tar_name(image_profile: str) -> str:
    return f"{image_stem(image_profile)}.tar"


def sandbox_dirname(image_profile: str) -> str:
    return f"{image_stem(image_profile)}.sif"


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_sha256sum_output(text: str) -> str | None:
    line = text.strip().splitlines()
    if not line:
        return None
    token = line[0].split()
    if not token:
        return None
    value = token[0].strip()
    if len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower()):
        return value.lower()
    return None


def parse_docker_inspect_digest(stdout: str) -> str:
    text = stdout.strip()
    if not text:
        raise VectorSimError("docker inspect returned empty digest", category="docker")
    if "sha256:" in text:
        start = text.index("sha256:")
        digest = text[start:].split()[0].split(",")[0].strip().strip("\"'")
        return digest
    return text.split()[0]


def safe_rmtree(path: Path, *, allowed_root: Path) -> None:
    """Remove a directory only if it is inside allowed_root."""
    resolved = path.resolve()
    allowed = allowed_root.resolve()
    if resolved == allowed:
        raise VectorSimError(
            f"refusing to delete allowed root {allowed}",
            category="cleanup",
        )
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise VectorSimError(
            f"refusing to delete {resolved}; not under {allowed}",
            category="cleanup",
        ) from exc
    if resolved.exists():
        shutil.rmtree(resolved)
