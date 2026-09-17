"""Cluster adapters. Bonecho-specific runtime flags live here, not in generic SLURM."""

from __future__ import annotations

from dataclasses import dataclass

from vector_lab.config.models import ApptainerConfig, ClusterProfile, bonecho_defaults
from vector_lab.cluster.detect import infer_home, infer_scratch


@dataclass
class PathDefaults:
    home_dir: str | None = None
    scratch_dir: str | None = None


class ClusterAdapter:
    name = "generic"

    def path_defaults(self, remote_user: str | None) -> PathDefaults:
        del remote_user
        return PathDefaults()

    def apptainer_defaults(self) -> ApptainerConfig:
        return ApptainerConfig()

    def apply_unresolved_paths(self, profile: ClusterProfile) -> ClusterProfile:
        defaults = self.path_defaults(profile.remote_user)
        if not profile.home_dir and defaults.home_dir:
            profile.home_dir = defaults.home_dir
        if not profile.scratch_dir and defaults.scratch_dir:
            profile.scratch_dir = defaults.scratch_dir
        return profile

    def derive_scratch_paths(self, profile: ClusterProfile) -> ClusterProfile:
        scratch = profile.scratch_dir
        if not scratch:
            return profile
        if not profile.paths.cache:
            profile.paths.cache = f"{scratch}/docker-isaac-sim"
        if not profile.paths.logs:
            profile.paths.logs = f"{scratch}/isaaclab/logs"
        if not profile.paths.containers:
            profile.paths.containers = f"{scratch}/isaaclab-containers"
        if not profile.paths.workspace:
            profile.paths.workspace = f"{scratch}/isaaclab"
        return profile


class SlurmAdapter(ClusterAdapter):
    name = "slurm"


class BonechoAdapter(SlurmAdapter):
    name = "bonecho"

    def path_defaults(self, remote_user: str | None) -> PathDefaults:
        if not remote_user:
            return PathDefaults()
        return PathDefaults(home_dir=f"/h/{remote_user}", scratch_dir=f"/scratch/{remote_user}")

    def apptainer_defaults(self) -> ApptainerConfig:
        return bonecho_defaults().apptainer


def adapter_for(name: str) -> ClusterAdapter:
    key = name.lower()
    if key == "bonecho":
        return BonechoAdapter()
    return SlurmAdapter()


def apply_generic_path_detection(
    profile: ClusterProfile,
    *,
    home: str | None,
    scratch: str | None,
) -> ClusterProfile:
    if not profile.home_dir:
        profile.home_dir = infer_home(printenv_home=home or "")
    if not profile.scratch_dir:
        found, _source = infer_scratch(existing_directories=[scratch] if scratch else [])
        profile.scratch_dir = found
    return profile
