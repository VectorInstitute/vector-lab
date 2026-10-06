"""CLI / profile job argument resolution and Isaac Lab train.py flag expansion."""

from __future__ import annotations

from dataclasses import dataclass, field

from vector_sim.cluster.slurm import gres_request, slurm_time
from vector_sim.config.models import ClusterProfile, DefaultJob
from vector_sim.images.naming import image_stem


@dataclass
class JobSpec:
    task: str
    image_profile: str = "base"
    partition: str | None = None
    gpu_type: str | None = None
    gpu_count: int = 1
    gres: str | None = None
    cpus: int = 8
    memory: str = "32G"
    time: str = "1h"
    python_executable: str = "scripts/reinforcement_learning/rsl_rl/train.py"
    video: bool = False
    video_length: int = 200
    video_interval: int = 1000
    headless: bool | None = None
    extra_args: list[str] = field(default_factory=list)

    @property
    def container_name(self) -> str:
        return image_stem(self.image_profile)

    @property
    def resolved_gres(self) -> str | None:
        if self.gres:
            return self.gres
        return gres_request(
            DefaultJob(gpu_type=self.gpu_type, gpu_count=self.gpu_count)
        )

    @property
    def resolved_time(self) -> str:
        return slurm_time(self.time)


def resolve_job_spec(profile: ClusterProfile, **overrides) -> JobSpec:
    """Resolve JobSpec: cluster profile defaults, then CLI overrides."""
    defaults = profile.default_job
    task = overrides.get("task")
    if not task:
        raise ValueError("--task is required")
    image_profile = overrides.get("image_profile") or profile.image.name or "base"
    video = bool(overrides.get("video", False))
    headless = bool(overrides.get("headless")) or video
    return JobSpec(
        task=str(task),
        image_profile=str(image_profile),
        partition=overrides.get("partition") or defaults.partition,
        gpu_type=overrides.get("gpu") or overrides.get("gpu_type") or defaults.gpu_type,
        gpu_count=int(overrides.get("gpus") or overrides.get("gpu_count") or defaults.gpu_count or 1),
        gres=overrides.get("gres"),
        cpus=int(overrides.get("cpus") or defaults.cpus or 8),
        memory=str(overrides.get("memory") or defaults.memory or "32G"),
        time=str(overrides.get("time") or defaults.time or "1h"),
        python_executable=str(
            overrides.get("python_executable") or profile.python_executable
        ),
        video=video,
        video_length=int(overrides.get("video_length") or 200),
        video_interval=int(overrides.get("video_interval") or 1000),
        headless=headless,
        extra_args=list(overrides.get("extra_args") or []),
    )


def expand_train_args(spec: JobSpec) -> list[str]:
    """Build Isaac Lab train.py argv using underscore flag names."""
    args = ["--task", spec.task]
    if spec.headless:
        args.append("--headless")
    if spec.video:
        args.extend(
            [
                "--enable_cameras",
                "--video",
                "--video_length",
                str(spec.video_length),
                "--video_interval",
                str(spec.video_interval),
            ]
        )
    args.extend(spec.extra_args)
    return args
