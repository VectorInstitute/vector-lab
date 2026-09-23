"""YAML profile store and local job metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from vector_lab.config.models import ClusterProfile, JobRecord
from vector_lab.errors import ConfigError

PROFILES_DIRNAME = ".vector-lab"
STATE_DIRNAME = "state"
JOBS_FILENAME = "jobs.json"
ACTIVE_FILENAME = "active_cluster"


class ConfigStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.profiles_dir = self.root
        self.state_dir = self.root / STATE_DIRNAME
        self.generated_dir = self.root / "generated"
        self.artifacts_dir = self.root / "artifacts"
        self.work_dir = self.root / "work"

    @classmethod
    def locate(cls, start: Path | None = None, explicit: Path | None = None) -> ConfigStore:
        if explicit is not None:
            return cls(Path(explicit))
        here = Path(start or Path.cwd()).resolve()
        for candidate in [here, *here.parents]:
            cfg = candidate / PROFILES_DIRNAME
            if cfg.is_dir():
                return cls(cfg)
        return cls(here / PROFILES_DIRNAME)

    def profile_path(self, name: str) -> Path:
        return self.profiles_dir / f"{name}.yaml"

    def exists(self) -> bool:
        return self.root.is_dir()

    def ensure_local(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def load_profile(self, name: str) -> ClusterProfile:
        path = self.profile_path(name)
        if not path.is_file():
            raise ConfigError(
                f"cluster profile '{name}' not found at {path}",
                suggestion=f"Run: vector-lab onboard {name}",
            )
        with path.open() as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"profile {path} is not a mapping")
        data.setdefault("name", name)
        profile = ClusterProfile.from_dict(data)
        # Fill missing cluster_type from a built-in template only when the key was
        # absent on disk (legacy profiles). Adapter selection still uses the field.
        if "cluster_type" not in data:
            from vector_lab.config.models import BUILTIN_PROFILES

            factory = BUILTIN_PROFILES.get(profile.name.lower())
            if factory is not None:
                profile.cluster_type = factory().cluster_type
        return profile

    def save_profile(self, profile: ClusterProfile) -> Path:
        self.ensure_local()
        path = self.profile_path(profile.name)
        payload = profile.to_dict()
        with path.open("w") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        return path

    def save_profile_preserving_overrides(self, profile: ClusterProfile) -> Path:
        path = self.profile_path(profile.name)
        if path.is_file():
            existing = self.load_profile(profile.name)
            profile = existing.merge_detected(profile)
        return self.save_profile(profile)

    def set_active(self, name: str) -> None:
        self.ensure_local()
        (self.root / ACTIVE_FILENAME).write_text(name + "\n")

    def get_active(self) -> str | None:
        path = self.root / ACTIVE_FILENAME
        if not path.is_file():
            return None
        text = path.read_text().strip()
        return text or None

    def images_state_path(self) -> Path:
        return self.state_dir / "images.json"

    def jobs_path(self) -> Path:
        return self.state_dir / JOBS_FILENAME

    def load_jobs(self) -> list[JobRecord]:
        path = self.jobs_path()
        if not path.is_file():
            return []
        raw = json.loads(path.read_text())
        if isinstance(raw, dict) and "jobs" in raw:
            items = raw["jobs"]
        elif isinstance(raw, list):
            items = raw
        else:
            raise ConfigError(f"invalid jobs metadata at {path}")
        return [JobRecord(**{k: v for k, v in item.items() if k in JobRecord.__dataclass_fields__}) for item in items]

    def save_jobs(self, jobs: list[JobRecord]) -> Path:
        self.ensure_local()
        path = self.jobs_path()
        payload: dict[str, Any] = {"jobs": [job.__dict__ for job in jobs]}
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def upsert_job(self, record: JobRecord) -> None:
        jobs = [j for j in self.load_jobs() if j.job_id != record.job_id]
        jobs.append(record)
        self.save_jobs(jobs)

    def find_job(self, job_id: str) -> JobRecord | None:
        for job in self.load_jobs():
            if job.job_id == job_id:
                return job
        return None
