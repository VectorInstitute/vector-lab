from vector_lab.cli import build_parser, hoist_global_flags, parse_args


def test_parse_doctor_and_global_flags() -> None:
    args = parse_args(["--dry-run", "--verbose", "--cluster", "bonecho", "doctor"])
    assert args.command == "doctor"
    assert args.dry_run is True
    assert args.verbose is True
    assert args.cluster == "bonecho"


def test_parse_flags_after_subcommand() -> None:
    args = parse_args(["doctor", "--dry-run", "--verbose", "--cluster", "bonecho"])
    assert args.command == "doctor"
    assert args.dry_run is True
    assert args.verbose is True
    assert args.cluster == "bonecho"


def test_hoist_keeps_subcommand_flags() -> None:
    hoisted = hoist_global_flags(
        ["run", "--dry-run", "--task", "Isaac-Cartpole-v0", "--video", "--config-dir", "/tmp/cfg"]
    )
    assert hoisted[:4] == ["--dry-run", "--config-dir", "/tmp/cfg", "run"]
    assert "--task" in hoisted
    assert "--video" in hoisted


def test_parse_init() -> None:
    args = parse_args(["init", "bonecho", "--ssh-alias", "bonecho", "--isaaclab", "/opt/IsaacLab"])
    assert args.command == "init"
    assert args.name == "bonecho"
    assert args.ssh_alias == "bonecho"
    assert args.isaaclab == "/opt/IsaacLab"


def test_parse_setup() -> None:
    args = parse_args(["setup", "bonecho", "--isaaclab", "/opt/IsaacLab"])
    assert args.command == "setup"
    assert args.name == "bonecho"
    assert args.isaaclab == "/opt/IsaacLab"


def test_parse_run_video_flags() -> None:
    args = parse_args(
        ["run", "--task", "Isaac-Cartpole-v0", "--video", "--video-length", "200", "--video-interval", "1000"]
    )
    assert args.task == "Isaac-Cartpole-v0"
    assert args.video is True
    assert args.video_length == 200
    assert args.video_interval == 1000


def test_parse_bootstrap() -> None:
    args = parse_args(["bootstrap", "--install"])
    assert args.command == "bootstrap"
    assert args.install is True


def test_parse_build_and_push_flags() -> None:
    build = parse_args(["build", "--image-profile", "base", "--force"])
    assert build.command == "build"
    assert build.force is True
    push = parse_args(["push", "--force-convert", "--force-upload"])
    assert push.force_convert is True
    assert push.force_upload is True


def test_parse_run_requires_task() -> None:
    args = parse_args(
        [
            "run",
            "--task",
            "Isaac-Cartpole-v0",
            "--video",
            "--partition",
            "a40_b1",
            "--extra-arg",
            "seed=42",
            "--extra-arg",
            "num_envs=16",
        ]
    )
    assert args.task == "Isaac-Cartpole-v0"
    assert args.video is True
    assert args.partition == "a40_b1"
    assert args.extra_args == ["seed=42", "num_envs=16"]


def test_parse_logs_follow() -> None:
    args = parse_args(["logs", "123", "--follow"])
    assert args.follow is True
    assert args.job_id == "123"


def test_help_lists_commands() -> None:
    help_text = build_parser().format_help()
    for name in (
        "init",
        "setup",
        "doctor",
        "bootstrap",
        "onboard",
        "auth",
        "deploy",
        "smoke-test",
        "build",
        "push",
        "run",
        "status",
        "logs",
        "cancel",
        "videos",
        "pull-video",
        "shell",
    ):
        assert name in help_text
    assert "vector-lab" in help_text
    assert "isaac-cluster" not in help_text
