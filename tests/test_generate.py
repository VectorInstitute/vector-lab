import os
import re
import subprocess
from pathlib import Path

from vector_sim.cluster.adapters import VectorSlurmAdapter
from vector_sim.cluster.slurm import parse_sinfo_pipe_table, slurm_time
from vector_sim.config.models import ClusterProfile, SchedulerConfig, bonecho_defaults
from vector_sim.jobs.generate import generate_runtime


def _writable_sandbox(text: str) -> bool:
    return re.search(r"(^|[\s])--writable([\s]|$)", text) is not None


def _bonecho_profile() -> ClusterProfile:
    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.home_dir = "/h/alice"
    profile.scratch_dir = "/scratch/alice"
    VectorSlurmAdapter().derive_scratch_paths(profile)
    return profile


def test_generated_bonecho_submit_matches_working_behavior() -> None:
    generated = generate_runtime(_bonecho_profile())
    submit = generated.submit_job_slurm
    assert "#!/bin/bash -l" in submit
    assert "module load apptainer" in submit
    assert "bash -l -c 'sbatch < job.sh'" in submit
    assert "#SBATCH --partition=a40_b1" in submit
    assert "#SBATCH --gres=gpu:a40:1" in submit
    assert "#SBATCH --cpus-per-task=8" in submit
    assert "#SBATCH --mem=32G" in submit
    assert slurm_time("1h") in submit
    assert "run_singularity.sh" in submit
    assert "docker/cluster/run_singularity.sh" not in submit


def test_submit_wrapper_omits_empty_runner_argument(tmp_path: Path) -> None:
    profile = ClusterProfile(name="test", ssh_alias="test")
    submit = generate_runtime(profile).submit_job_slurm
    submit_path = tmp_path / "submit_job_slurm.sh"
    submit_path.write_text(submit)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "captured-job.sh"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text('#!/usr/bin/env bash\ncat > "$CAPTURED_JOB"\n')
    sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURED_JOB": str(captured),
    }

    subprocess.run(
        ["bash", str(submit_path), "/scratch/alice/run", "isaac-lab-base"],
        cwd=tmp_path,
        env=env,
        check=True,
    )

    runner_line = next(
        line for line in captured.read_text().splitlines() if "run_singularity.sh" in line
    )
    assert runner_line.endswith('"/scratch/alice/run" "isaac-lab-base"')
    assert '""' not in runner_line


def test_generated_bonecho_runner_apptainer_flags() -> None:
    runner = generate_runtime(_bonecho_profile()).run_singularity
    assert "set -e" in runner
    assert "--nv" in runner
    assert "--containall" in runner
    assert "--writable-tmpfs" in runner
    assert "--no-home" not in runner
    assert not _writable_sandbox(runner)
    assert "$CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw" in runner
    assert "${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache" in runner
    assert "${DOCKER_USER_HOME}/.cache/ov" in runner
    assert "${DOCKER_USER_HOME}/.cache/pip" in runner
    assert "${DOCKER_USER_HOME}/.cache/nvidia/GLCache" in runner
    assert "${DOCKER_USER_HOME}/.nv/ComputeCache" in runner
    assert "${DOCKER_USER_HOME}/.nvidia-omniverse/logs" in runner
    assert "${DOCKER_USER_HOME}/.local/share/ov/data" in runner
    assert "${DOCKER_USER_HOME}/Documents" in runner
    assert "/workspace/isaaclab:rw" in runner
    assert "mollysun" not in runner


def test_generated_bonecho_env_uses_alias_and_derived_paths() -> None:
    env = generate_runtime(_bonecho_profile()).env_cluster
    assert "CLUSTER_LOGIN=bonecho" in env
    assert "CLUSTER_ISAAC_SIM_CACHE_DIR=/scratch/alice/docker-isaac-sim" in env
    assert "CLUSTER_ISAACLAB_DIR=/scratch/alice/isaaclab" in env
    assert "CLUSTER_SIF_PATH=/scratch/alice/isaaclab-containers" in env
    assert "/scratch/alice/isaaclab" in env
    assert "<user>" not in env
    assert "<username>" not in env
    assert "<remote_user>" not in env
    assert "mollysun" not in env
    assert "DOCKER_ISAACSIM_ROOT_PATH=/isaac-sim" in env
    assert "DOCKER_USER_HOME=/root" in env


def test_env_cluster_rejects_username_placeholder() -> None:
    from vector_sim.errors import ConfigError
    from vector_sim.jobs.generate import render_env_cluster

    profile = bonecho_defaults()
    profile.remote_user = "<user>"
    profile.scratch_dir = "/scratch/<user>"
    profile.paths.workspace = "/scratch/<user>/isaaclab"
    try:
        render_env_cluster(profile)
        assert False, "expected ConfigError"
    except ConfigError as exc:
        assert "<user>" in str(exc)


def test_generic_slurm_does_not_inherit_bonecho_hacks() -> None:
    profile = ClusterProfile(name="other", ssh_alias="other")
    profile.scheduler = SchedulerConfig(type="slurm", remote_shell="none")
    profile.paths.logs = "/fast/alice/isaaclab/logs"
    generated = generate_runtime(profile)
    assert "module load apptainer" not in generated.submit_job_slurm
    assert "bash -l -c 'sbatch < job.sh'" not in generated.submit_job_slurm
    assert "--writable-tmpfs" not in generated.run_singularity
    assert "--containall" not in generated.run_singularity
    assert "$CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw" in generated.run_singularity
