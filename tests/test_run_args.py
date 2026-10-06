from vector_sim.cluster.adapters import VectorSlurmAdapter
from vector_sim.config.models import bonecho_defaults
from vector_sim.jobs.args import expand_train_args, resolve_job_spec


def _profile():
    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.scratch_dir = "/scratch/alice"
    VectorSlurmAdapter().derive_scratch_paths(profile)
    return profile


def test_video_cli_expansion_uses_underscore_flags() -> None:
    spec = resolve_job_spec(
        _profile(),
        task="Isaac-Cartpole-v0",
        video=True,
        video_length=200,
        video_interval=1000,
    )
    args = expand_train_args(spec)
    assert args == [
        "--task",
        "Isaac-Cartpole-v0",
        "--headless",
        "--enable_cameras",
        "--video",
        "--video_length",
        "200",
        "--video_interval",
        "1000",
    ]


def test_no_video_flags_without_video() -> None:
    args = expand_train_args(resolve_job_spec(_profile(), task="Isaac-Cartpole-v0"))
    assert "--video" not in args
    assert "--enable_cameras" not in args
    assert "--headless" not in args


def test_cli_override_precedence() -> None:
    spec = resolve_job_spec(
        _profile(),
        task="Isaac-Cartpole-v0",
        partition="a100_b1",
        gpu="a100",
        cpus=16,
        memory="64G",
        time="2h",
    )
    assert spec.partition == "a100_b1"
    assert spec.resolved_gres == "gpu:a100:1"
    assert spec.cpus == 16
    assert spec.memory == "64G"
    assert spec.resolved_time == "02:00:00"
    # defaults remain when not overridden
    default = resolve_job_spec(_profile(), task="Isaac-Cartpole-v0")
    assert default.partition == "a40_b1"
    assert default.resolved_gres == "gpu:a40:1"
