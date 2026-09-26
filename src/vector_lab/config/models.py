"""Profile, fingerprint, and job metadata models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SchedulerConfig:
    type: str = "slurm"
    remote_shell: str = "none"


@dataclass
class ApptainerConfig:
    """Container runtime settings. Vector SLURM sites set these on the cluster profile."""

    module: str | None = None
    command: str = "singularity"
    writable_mode: str | None = None
    extra_exec_args: list[str] = field(default_factory=list)


@dataclass
class ClusterPaths:
    cache: str | None = None
    logs: str | None = None
    containers: str | None = None
    workspace: str | None = None


@dataclass
class DefaultJob:
    partition: str | None = None
    gpu_type: str | None = None
    gpu_count: int = 1
    cpus: int = 8
    memory: str = "32G"
    time: str = "1h"


@dataclass
class GpuPartition:
    name: str
    gres: str | None = None
    state: str | None = None
    gpu_types: list[str] = field(default_factory=list)


@dataclass
class ImageProfile:
    name: str = "base"
    docker_image: str = "isaac-lab-base:latest"


@dataclass
class Fingerprints:
    """Separated so a Docker image ID is never mixed into pre-build inputs."""

    build_input_fingerprint: str | None = None
    built_image_digest: str | None = None
    conversion_fingerprint: str | None = None


@dataclass
class JobRecord:
    job_id: str
    cluster: str
    remote_run_dir: str
    slurm_log: str
    submitted_at: str
    task: str
    image_profile: str
    partition: str | None = None
    gres: str | None = None
    cpus: int | None = None
    memory: str | None = None
    time: str | None = None
    video: bool = False
    python_executable: str | None = None
    train_args: list[str] = field(default_factory=list)
    gui: bool = False
    gui_status: str | None = None
    hostname: str | None = None
    private_ip: str | None = None
    display: str | None = None
    vnc_port: int | None = None
    novnc_port: int | None = None

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        cluster: str,
        remote_run_dir: str,
        slurm_log: str,
        task: str,
        image_profile: str,
        submitted_at: str | None = None,
        partition: str | None = None,
        gres: str | None = None,
        cpus: int | None = None,
        memory: str | None = None,
        time: str | None = None,
        video: bool = False,
        python_executable: str | None = None,
        train_args: list[str] | None = None,
        gui: bool = False,
        gui_status: str | None = None,
        hostname: str | None = None,
        private_ip: str | None = None,
        display: str | None = None,
        vnc_port: int | None = None,
        novnc_port: int | None = None,
    ) -> JobRecord:
        return cls(
            job_id=job_id,
            cluster=cluster,
            remote_run_dir=remote_run_dir,
            slurm_log=slurm_log,
            submitted_at=submitted_at or _utcnow_iso(),
            task=task,
            image_profile=image_profile,
            partition=partition,
            gres=gres,
            cpus=cpus,
            memory=memory,
            time=time,
            video=video,
            python_executable=python_executable,
            train_args=list(train_args or []),
            gui=gui,
            gui_status=gui_status,
            hostname=hostname,
            private_ip=private_ip,
            display=display,
            vnc_port=vnc_port,
            novnc_port=novnc_port,
        )


@dataclass
class ClusterProfile:
    name: str
    ssh_alias: str  # OpenSSH Host alias / target used to connect
    cluster_type: str = "slurm"  # behavior implementation: slurm | vector-slurm | ...
    resolved_host: str | None = None
    remote_user: str | None = None
    home_dir: str | None = None
    scratch_dir: str | None = None
    isaaclab_path: str | None = None
    python_executable: str = "scripts/reinforcement_learning/rsl_rl/train.py"
    remove_code_copy_after_job: bool = False
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    apptainer: ApptainerConfig = field(default_factory=ApptainerConfig)
    paths: ClusterPaths = field(default_factory=ClusterPaths)
    default_job: DefaultJob = field(default_factory=DefaultJob)
    image: ImageProfile = field(default_factory=ImageProfile)
    fingerprints: Fingerprints = field(default_factory=Fingerprints)
    discovered_partitions: list[GpuPartition] = field(default_factory=list)
    user_set: list[str] = field(default_factory=list)

    def merge_detected(self, detected: ClusterProfile) -> ClusterProfile:
        """Fill from *detected* while keeping fields listed in ``user_set``."""
        current = self.to_dict()
        incoming = detected.to_dict()
        merged = _merge_preserving(incoming, current, self.user_set)
        merged["user_set"] = list(self.user_set)
        merged["name"] = self.name or detected.name
        return ClusterProfile.from_dict(merged)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClusterProfile:
        payload = dict(data)
        scheduler = SchedulerConfig(**_filter_fields(SchedulerConfig, payload.pop("scheduler", {}) or {}))
        apptainer_raw = dict(payload.pop("apptainer", {}) or {})
        extra = apptainer_raw.get("extra_exec_args") or []
        apptainer_raw["extra_exec_args"] = list(extra)
        apptainer = ApptainerConfig(**_filter_fields(ApptainerConfig, apptainer_raw))
        paths = ClusterPaths(**_filter_fields(ClusterPaths, payload.pop("paths", {}) or {}))
        default_job = DefaultJob(**_filter_fields(DefaultJob, payload.pop("default_job", {}) or {}))
        image = ImageProfile(**_filter_fields(ImageProfile, payload.pop("image", {}) or {}))
        fingerprints = Fingerprints(**_filter_fields(Fingerprints, payload.pop("fingerprints", {}) or {}))
        raw_parts = payload.pop("discovered_partitions", []) or []
        discovered_partitions = [
            GpuPartition(
                **_filter_fields(
                    GpuPartition,
                    {**p, "gpu_types": list(p.get("gpu_types") or [])} if isinstance(p, Mapping) else {},
                )
            )
            for p in raw_parts
            if isinstance(p, Mapping)
        ]
        user_set = list(payload.pop("user_set", []) or [])
        filtered = _filter_fields(cls, payload)
        if "name" not in filtered or "ssh_alias" not in filtered:
            raise ValueError("cluster profile requires name and ssh_alias")
        return cls(
            **filtered,
            scheduler=scheduler,
            apptainer=apptainer,
            paths=paths,
            default_job=default_job,
            image=image,
            fingerprints=fingerprints,
            discovered_partitions=discovered_partitions,
            user_set=user_set,
        )


def bonecho_defaults(*, ssh_alias: str = "bonecho") -> ClusterProfile:
    """Built-in Bonecho profile: name/SSH target independent of cluster_type=vector-slurm."""
    from vector_lab.cluster.adapters import CLUSTER_TYPE_VECTOR_SLURM, VECTOR_SLURM_APPTAINER

    return ClusterProfile(
        name="bonecho",
        ssh_alias=ssh_alias,
        cluster_type=CLUSTER_TYPE_VECTOR_SLURM,
        scheduler=SchedulerConfig(type="slurm", remote_shell="login"),
        apptainer=ApptainerConfig(
            module=VECTOR_SLURM_APPTAINER.module,
            command=VECTOR_SLURM_APPTAINER.command,
            writable_mode=VECTOR_SLURM_APPTAINER.writable_mode,
            extra_exec_args=list(VECTOR_SLURM_APPTAINER.extra_exec_args),
        ),
        default_job=DefaultJob(
            partition="a40_b1",
            gpu_type="a40",
            gpu_count=1,
            cpus=8,
            memory="32G",
            time="1h",
        ),
        image=ImageProfile(name="base", docker_image="isaac-lab-base:latest"),
    )


# Built-in profile factories keyed by user-facing profile name (not adapter selection).
BUILTIN_PROFILES: dict[str, Callable[..., ClusterProfile]] = {
    "bonecho": bonecho_defaults,
}


def _filter_fields(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def _set_path(tree: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor: dict[str, Any] = tree
    for part in parts[:-1]:
        next_val = cursor.get(part)
        if not isinstance(next_val, dict):
            next_val = {}
            cursor[part] = next_val
        cursor = next_val
    cursor[parts[-1]] = value


def _get_path(tree: Mapping[str, Any], dotted: str) -> Any:
    cursor: Any = tree
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _merge_preserving(detected: dict[str, Any], current: dict[str, Any], user_set: Iterable[str]) -> dict[str, Any]:
    merged = _deep_merge(detected, current, prefer="detected_then_current_fill")
    for path in user_set:
        value = _get_path(current, path)
        if value is not None:
            _set_path(merged, path, value)
    return merged


def _deep_merge(detected: dict[str, Any], current: dict[str, Any], *, prefer: str) -> dict[str, Any]:
    """Start from detected; keep current values only when detected is empty/None."""
    out: dict[str, Any] = dict(detected)
    for key, current_val in current.items():
        detected_val = out.get(key)
        if isinstance(current_val, dict) and isinstance(detected_val, dict):
            out[key] = _deep_merge(detected_val, current_val, prefer=prefer)
        elif detected_val in (None, "", [], {}) and current_val not in (None, "", [], {}):
            out[key] = current_val
    return out
