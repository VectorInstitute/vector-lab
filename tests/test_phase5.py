from pathlib import Path

from vector_lab.cluster.adapters import VectorSlurmAdapter
from vector_lab.config.models import JobRecord, bonecho_defaults
from vector_lab.config.store import ConfigStore
from vector_lab.exec import CommandRunner
from vector_lab.jobs.generate import ENV_NAME, RUNNER_NAME, SUBMIT_NAME, generate_runtime
from vector_lab.jobs.status import parse_sacct_line, parse_squeue_line
from vector_lab.jobs.submit import parse_sbatch_job_id
from vector_lab.jobs.sync import RSYNC_EXCLUDES, remote_submit_path, rsync_source_args, stage_generated_into_repo
from vector_lab.jobs.videos import parse_find_videos, pull_video_args
from vector_lab.ssh.session import wrap_login_shell


def _profile():
    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.scratch_dir = "/scratch/alice"
    VectorSlurmAdapter().derive_scratch_paths(profile)
    return profile


def test_parse_sbatch_job_id() -> None:
    assert parse_sbatch_job_id("Submitted batch job 385225\n") == "385225"
    assert parse_sbatch_job_id("error") is None


def test_rsync_excludes_and_uses_alias() -> None:
    args = rsync_source_args(Path("/tmp/IsaacLab"), "bonecho", "/scratch/alice/isaaclab_1")
    assert args[0] == "rsync"
    assert "--delete" not in args
    assert "bonecho:/scratch/alice/isaaclab_1/" in args[-1]
    joined = " ".join(args)
    assert ".git/" in joined
    assert ".venv/" in joined
    assert "__pycache__/" in joined
    assert ".vector-lab/artifacts/" in joined
    for pattern in RSYNC_EXCLUDES[:5]:
        assert pattern in joined


def test_generated_wrappers_staged_not_upstream(tmp_path: Path) -> None:
    repo = tmp_path / "IsaacLab"
    repo.mkdir()
    generated = generate_runtime(_profile()).as_dict()
    dest = stage_generated_into_repo(repo, generated)
    assert (dest / SUBMIT_NAME).is_file()
    assert (dest / RUNNER_NAME).is_file()
    assert (dest / ENV_NAME).is_file()
    submit = (dest / SUBMIT_NAME).read_text()
    assert "bash -l -c 'sbatch < job.sh'" in submit
    assert "module load apptainer" in submit
    assert 'bash "$SCRIPT_DIR/run_singularity.sh"' in submit
    assert remote_submit_path("/scratch/alice/isaaclab_1").endswith(
        ".vector-lab/generated/submit_job_slurm.sh"
    )
    assert "/docker/cluster/submit_job_slurm.sh" not in remote_submit_path("/scratch/alice/isaaclab_1")


def test_jobs_json_roundtrip_extended(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path)
    record = JobRecord.create(
        job_id="385225",
        cluster="bonecho",
        remote_run_dir="/scratch/alice/isaaclab_20260101",
        slurm_log="/scratch/alice/isaaclab_20260101/slurm-385225.out",
        task="Isaac-Cartpole-v0",
        image_profile="base",
        partition="a40_b1",
        gres="gpu:a40:1",
        cpus=8,
        memory="32G",
        time="01:00:00",
        video=True,
        train_args=["--task", "Isaac-Cartpole-v0", "--video"],
    )
    store.upsert_job(record)
    loaded = store.find_job("385225")
    assert loaded is not None
    assert loaded.video is True
    assert loaded.gres == "gpu:a40:1"
    assert loaded.train_args[0] == "--task"


def test_status_squeue_and_sacct_parsing() -> None:
    running = parse_squeue_line("385225|training|RUNNING|00:10:00|a40_b1|gpu01\n", "385225")
    assert running is not None
    assert running.state == "RUNNING"
    done = parse_sacct_line(
        "385225|training|COMPLETED|01:00:00|0:0|a40_b1|gpu01\n"
        "385225.batch|batch|COMPLETED|01:00:00|0:0|a40_b1|gpu01\n",
        "385225",
    )
    assert done is not None
    assert done.state == "COMPLETED"
    failed = parse_sacct_line("99|x|OUT_OF_MEMORY|00:01:00|1:0|a40_b1|n\n", "99")
    assert failed is not None
    assert failed.state == "OUT_OF_MEMORY"


def test_video_sorting_newest_first() -> None:
    text = "\n".join(
        [
            "100 10 /scratch/alice/isaaclab/logs/old.mp4",
            "300 20 /scratch/alice/isaaclab/logs/rsl_rl/cartpole/run/videos/train/new.mp4",
            "200 15 /scratch/alice/isaaclab/logs/mid.mp4",
        ]
    )
    videos = parse_find_videos(text)
    assert videos[0].name == "new.mp4"
    assert videos[-1].name == "old.mp4"


def test_pull_video_uses_ssh_alias() -> None:
    args = pull_video_args("bonecho", "/scratch/alice/isaaclab/logs/a.mp4", Path("/tmp/a.mp4"))
    assert args[0] == "rsync"
    assert args[-2].startswith("bonecho:")


def test_login_shell_sbatch_wrapping() -> None:
    wrapped = wrap_login_shell("cd /scratch/x && bash submit.sh /scratch/x isaac-lab-base --task T")
    assert wrapped.startswith("bash -l -c ")
