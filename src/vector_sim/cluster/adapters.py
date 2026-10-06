"""Cluster adapters. Behavior is selected by profile.cluster_type, never by profile name."""

from __future__ import annotations

from dataclasses import dataclass

from vector_sim.config.models import ApptainerConfig, ClusterProfile
from vector_sim.cluster.detect import infer_home, infer_scratch

CLUSTER_TYPE_SLURM = "slurm"
CLUSTER_TYPE_VECTOR_SLURM = "vector-slurm"

WRITABLE_TMPFS = "writable-tmpfs"
# Required by the Isaac Lab container itself, not by any particular site.
ISAAC_LAB_EXEC_ARGS = ("--nv", "--containall", "--writable-tmpfs")

# Shared Vector Institute Apptainer defaults (Bonecho today; other Vector SLURM sites later).
VECTOR_SLURM_APPTAINER = ApptainerConfig(
    module="apptainer",
    command="singularity",
    writable_mode=WRITABLE_TMPFS,
    extra_exec_args=list(ISAAC_LAB_EXEC_ARGS),
)


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
    """Generic SLURM adapter (no Vector login-shell / writable-tmpfs conventions)."""

    name = CLUSTER_TYPE_SLURM


class VectorSlurmAdapter(SlurmAdapter):
    """Vector Institute SLURM sites: login-shell SLURM + Apptainer module + writable-tmpfs.

    Selected only when ``profile.cluster_type == "vector-slurm"``.
    Profile name and SSH target are independent.
    """

    name = CLUSTER_TYPE_VECTOR_SLURM

    def path_defaults(self, remote_user: str | None) -> PathDefaults:
        if not remote_user:
            return PathDefaults()
        return PathDefaults(home_dir=f"/h/{remote_user}", scratch_dir=f"/scratch/{remote_user}")

    def apptainer_defaults(self) -> ApptainerConfig:
        return ApptainerConfig(
            module=VECTOR_SLURM_APPTAINER.module,
            command=VECTOR_SLURM_APPTAINER.command,
            writable_mode=VECTOR_SLURM_APPTAINER.writable_mode,
            extra_exec_args=list(VECTOR_SLURM_APPTAINER.extra_exec_args),
        )


def adapter_for(cluster_type: str | None) -> ClusterAdapter:
    """Resolve a behavior adapter from ``cluster_type`` only (never profile name)."""
    key = (cluster_type or CLUSTER_TYPE_SLURM).strip().lower() or CLUSTER_TYPE_SLURM
    if key == CLUSTER_TYPE_VECTOR_SLURM:
        return VectorSlurmAdapter()
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
