"""Generic home/scratch inference from remote environment probes.

The generic layer never assumes /scratch/<user>. Cluster adapters may supply
fallbacks after this function returns None.
"""

from __future__ import annotations

from dataclasses import dataclass


SCRATCH_ENV_KEYS = ("SCRATCH", "SCRATCHDIR", "SCRATCH_DIR", "CSCRATCH")


@dataclass
class PathProbe:
    home_dir: str | None
    scratch_dir: str | None
    source: str


def candidate_scratch_dirs(*, remote_user: str | None, home_dir: str | None) -> list[str]:
    """Conventional scratch locations worth testing when no scratch variable is exported."""
    candidates: list[str] = []
    if remote_user:
        candidates.append(f"/scratch/{remote_user}")
    if home_dir:
        candidates.append(f"{home_dir.rstrip('/')}/scratch")
    return list(dict.fromkeys(c for c in candidates if c))


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
    for key in SCRATCH_ENV_KEYS:
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
