"""Compare generated wrappers to the known-working Bonecho patches in Isaac Lab."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vector_lab.cluster.adapters import VectorSlurmAdapter
from vector_lab.config.models import bonecho_defaults
from vector_lab.jobs.generate import generate_runtime

_default_lab = Path(__file__).resolve().parents[2] / "IsaacLab"
PATCHED = Path(os.environ.get("ISAACLAB_PATH", str(_default_lab))) / "docker" / "cluster"


@pytest.mark.skipif(not (PATCHED / "run_singularity.sh").is_file(), reason="Isaac Lab clone not present")
def test_generated_covers_manual_bonecho_patches() -> None:
    patched_runner = (PATCHED / "run_singularity.sh").read_text()
    patched_submit = (PATCHED / "submit_job_slurm.sh").read_text()
    assert "set -e" in patched_runner
    assert "--writable-tmpfs" in patched_runner
    assert "--nv" in patched_runner
    assert "--containall" in patched_runner
    assert "bash -l -c 'sbatch < job.sh'" in patched_submit
    assert "module load apptainer" in patched_submit
    assert "#!/bin/bash -l" in patched_submit

    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.scratch_dir = "/scratch/alice"
    VectorSlurmAdapter().derive_scratch_paths(profile)
    generated = generate_runtime(profile)

    for token in ("set -e", "--nv", "--containall", "--writable-tmpfs", "/workspace/isaaclab/logs:rw"):
        assert token in patched_runner
        assert token in generated.run_singularity
    for token in ("#!/bin/bash -l", "module load apptainer", "bash -l -c 'sbatch < job.sh'", "--gres=gpu:a40:1"):
        assert token in patched_submit
        assert token in generated.submit_job_slurm
    assert "--no-home" not in generated.run_singularity
    assert "docker/cluster/run_singularity.sh" not in generated.submit_job_slurm
