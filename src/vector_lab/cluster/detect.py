"""Generic home/scratch inference from remote environment probes.

The generic layer never assumes /scratch/<user>. Cluster adapters may supply
fallbacks after this function returns None.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PathProbe:
    home_dir: str | None
    scratch_dir: str | None
    source: str


def infer_home(*, printenv_home: str = "", pwd_home: str = "", whoami: str = "") -> str | None:
    home = (printenv_home or "").strip()
    if home:
        return home
    pwd = (pwd_home or "").strip()
    if pwd:
        return pwd
    return None


def infer_scratch(
    *,
    env: dict[str, str] | None = None,
    existing_directories: list[str] | None = None,
) -> tuple[str | None, str]:
    """Return (path, source_key) from environment or existing dirs.

    Environment keys checked: SCRATCH, SCRATCHDIR, SCRATCH_DIR, CSCRATCH.
    """
    env = env or {}
    for key in ("SCRATCH", "SCRATCHDIR", "SCRATCH_DIR", "CSCRATCH"):
        value = (env.get(key) or "").strip()
        if value:
            return value, key
    for path in existing_directories or []:
        cleaned = path.strip()
        if cleaned:
            return cleaned, "existing_directory"
    return None, "unresolved"


def parse_env_assignments(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            env[key] = value.strip()
    return env
