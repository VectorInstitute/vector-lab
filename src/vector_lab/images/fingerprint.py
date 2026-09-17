"""Fingerprint helpers. Build inputs never include a Docker image ID or digest."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from vector_lab.config.models import Fingerprints

_COPY_ADD = re.compile(r"^\s*(?:COPY|ADD)\s+(.+)$", re.IGNORECASE | re.MULTILINE)
_RUNTIME_ONLY_PREFIXES = (
    "source/isaaclab/",
    "source/isaaclab_tasks/",
    "source/isaaclab_rl/",
    "source/isaaclab_assets/",
    "source/isaaclab_mimic/",
    "source/isaaclab_contrib/",
    "scripts/",
)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    return hash_bytes(text.encode())


def build_input_fingerprint(files: Mapping[str, str] | Iterable[Path]) -> str:
    """Hash named file contents used as Docker build inputs.

    Callers must pass Dockerfiles, compose/env files, install tooling, and
    dependency manifests — not ordinary task/runtime Python and not image IDs.
    """
    if isinstance(files, Mapping):
        items = sorted((name, content) for name, content in files.items())
    else:
        items = []
        for path in files:
            p = Path(path)
            rel = str(p)
            items.append((rel, p.read_text() if p.is_file() else ""))
        items.sort()
    hasher = hashlib.sha256()
    for name, content in items:
        hasher.update(name.encode())
        hasher.update(b"\0")
        hasher.update(content.encode())
        hasher.update(b"\0")
    return hasher.hexdigest()


def conversion_fingerprint(
    *,
    built_image_digest: str,
    backend: str,
    version: str,
    options: str,
) -> str:
    """Fingerprint a conversion. Includes image digest, never build-input files."""
    return hash_text("\n".join([built_image_digest, backend, version, options]))


def _read_rel(repo: Path, rel: str, out: dict[str, str]) -> None:
    path = Path(repo) / rel
    if path.is_file():
        out[rel.replace("\\", "/")] = path.read_text(errors="replace")


def parse_dockerfile_copy_sources(dockerfile_text: str) -> list[str]:
    """Extract COPY/ADD source paths. Skips recursive parent copies like ``../``."""
    sources: list[str] = []
    for match in _COPY_ADD.finditer(dockerfile_text):
        rest = match.group(1).strip()
        # Drop flags like --from=...
        while rest.startswith("--"):
            parts = rest.split(None, 1)
            rest = parts[1] if len(parts) > 1 else ""
        tokens = rest.split()
        if len(tokens) < 2:
            continue
        for src in tokens[:-1]:
            if src in {".", "..", "../", "./"} or src.startswith("../"):
                continue
            sources.append(src)
    return sources


def _dependency_manifest_globs(repo: Path) -> list[str]:
    rels: list[str] = []
    source = Path(repo) / "source"
    if not source.is_dir():
        return rels
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in {"setup.py", "pyproject.toml", "extension.toml", "setup.cfg"}:
            rels.append(str(path.relative_to(repo)).replace("\\", "/"))
    return sorted(rels)


def collect_build_input_files(repo: Path, image_profile: str) -> dict[str, str]:
    """Collect Docker build-affecting inputs.

    Includes Dockerfiles/compose/env, .dockerignore, install tooling, dependency
    manifests, and explicit COPY/ADD sources. Excludes ordinary runtime/task
    Python under source/ and scripts/.
    """
    root = Path(repo)
    docker = root / "docker"
    files: dict[str, str] = {}

    names = ["Dockerfile.base", "docker-compose.yaml", ".env.base"]
    if image_profile not in {"", "base", "isaaclab"}:
        names.extend([f"Dockerfile.{image_profile}", f".env.{image_profile}"])
    for name in names:
        _read_rel(root, f"docker/{name}", files)

    _read_rel(root, ".dockerignore", files)
    _read_rel(root, "isaaclab.sh", files)
    _read_rel(root, "tools/install_deps.py", files)

    # Other install helpers under tools/ (not notebooks/docs)
    tools = root / "tools"
    if tools.is_dir():
        for path in tools.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".sh", ".toml", ".txt", ".cfg"}:
                continue
            rel = str(path.relative_to(root)).replace("\\", "/")
            if "/test" in rel or "/tests/" in rel:
                continue
            _read_rel(root, rel, files)

    for rel in _dependency_manifest_globs(root):
        _read_rel(root, rel, files)

    # Explicit COPY/ADD sources from Dockerfiles (skip COPY ../)
    for key, text in list(files.items()):
        if not key.startswith("docker/Dockerfile"):
            continue
        for src in parse_dockerfile_copy_sources(text):
            # Paths in Dockerfiles are often relative to docker/ context or repo
            candidates = [src, f"docker/{src}"]
            for cand in candidates:
                cand = cand.lstrip("./")
                if any(cand.startswith(prefix) for prefix in _RUNTIME_ONLY_PREFIXES):
                    continue
                if cand.endswith(".py") and cand.startswith("source/") and not cand.endswith(
                    ("setup.py",)
                ):
                    # Ordinary extension Python is runtime-mounted; skip unless
                    # already included as a dependency manifest.
                    continue
                path = root / cand
                if path.is_file():
                    _read_rel(root, cand, files)
                elif path.is_dir():
                    for child in path.rglob("*"):
                        if child.is_file() and child.suffix in {".py", ".sh", ".toml", ".txt", ".cfg", ".yaml"}:
                            rel = str(child.relative_to(root)).replace("\\", "/")
                            if any(rel.startswith(prefix) for prefix in _RUNTIME_ONLY_PREFIXES):
                                if not rel.endswith(("setup.py", "pyproject.toml", "extension.toml", "setup.cfg")):
                                    continue
                            _read_rel(root, rel, files)

    # Never keep ordinary runtime Python that slipped in
    filtered: dict[str, str] = {}
    for rel, content in files.items():
        if rel.endswith(".py") and any(rel.startswith(p) for p in _RUNTIME_ONLY_PREFIXES):
            if not rel.endswith("setup.py") and "/config/extension.toml" not in rel:
                continue
        filtered[rel] = content
    return filtered


def fingerprint_repo_build_inputs(repo: Path, image_profile: str) -> str:
    return build_input_fingerprint(collect_build_input_files(repo, image_profile))


def empty_fingerprints() -> Fingerprints:
    return Fingerprints()
