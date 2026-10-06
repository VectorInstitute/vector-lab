"""Persisted image/artifact fingerprints for idempotent build/push."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from vector_sim.config.store import ConfigStore


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ImageArtifactState:
    image_profile: str
    docker_image: str
    build_input_fingerprint: str | None = None
    built_image_digest: str | None = None
    conversion_fingerprint: str | None = None
    local_artifact: str | None = None
    local_artifact_size: int | None = None
    local_artifact_sha256: str | None = None
    conversion_timestamp: str | None = None
    remote_path: str | None = None
    remote_sha256: str | None = None
    pushed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ImageArtifactState:
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


class ImageStateStore:
    def __init__(self, store: ConfigStore) -> None:
        self.store = store

    @property
    def path(self) -> Path:
        return self.store.images_state_path()

    def load_all(self) -> dict[str, ImageArtifactState]:
        path = self.path
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text())
        items = raw.get("images", raw) if isinstance(raw, dict) else {}
        out: dict[str, ImageArtifactState] = {}
        if isinstance(items, dict):
            for key, value in items.items():
                if isinstance(value, dict):
                    out[key] = ImageArtifactState.from_dict(value)
        return out

    def get(self, image_profile: str) -> ImageArtifactState | None:
        return self.load_all().get(image_profile)

    def upsert(self, record: ImageArtifactState) -> None:
        self.store.ensure_local()
        all_records = self.load_all()
        all_records[record.image_profile] = record
        payload = {"images": {name: rec.to_dict() for name, rec in all_records.items()}}
        self.path.write_text(json.dumps(payload, indent=2) + "\n")


def stamp() -> str:
    return _utcnow()
