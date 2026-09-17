from pathlib import Path

from vector_lab.cli import build_parser, parse_args
from vector_lab.cluster.adapters import BonechoAdapter
from vector_lab.commands.auth import AuthCommand
from vector_lab.commands.bootstrap import BootstrapCommand
from vector_lab.commands.deploy import DeployCommand
from vector_lab.commands.doctor import Doctor, format_doctor_report
from vector_lab.commands.onboard import OnboardCommand
from vector_lab.commands.smoke import SmokeTestCommand, evaluate_job
from vector_lab.compat import ISAACLAB_COMMIT, classify_isaaclab_commit, classify_tool_version
from vector_lab.config.models import bonecho_defaults
from vector_lab.config.store import ConfigStore
from vector_lab.exec import CommandRunner
from vector_lab.images.fingerprint import conversion_fingerprint, fingerprint_repo_build_inputs
from vector_lab.images.naming import artifact_filename
from vector_lab.images.state import ImageArtifactState, ImageStateStore
from vector_lab.jobs.status import JobStatus
from vector_lab.onboard.isaaclab import find_or_clone, resolve_existing
from vector_lab.onboard.prereqs import STATUS_INSTALLED, inspect_platform, inspect_prereqs
from vector_lab.ssh.multiplex import multiplex_from_ssh_g
from tests.fakes.runner import FakeExecute


def _isaaclab(tmp_path: Path) -> Path:
    root = tmp_path / "IsaacLab"
    (root / "docker/cluster").mkdir(parents=True)
    (root / "source/isaaclab").mkdir(parents=True)
    (root / "docker/container.py").write_text("print('x')\n")
    (root / "docker/cluster/cluster_interface.sh").write_text("#!/bin/bash\n")
    (root / "isaaclab.sh").write_text("#!/bin/bash\n")
    (root / "docker/Dockerfile.base").write_text("FROM x\n")
    (root / "docker/docker-compose.yaml").write_text("services: {}\n")
    (root / "docker/.env.base").write_text("ACCEPT_EULA=Y\n")
    return root


def _profile(repo: Path):
    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.scratch_dir = "/scratch/alice"
    profile.home_dir = "/h/alice"
    BonechoAdapter().derive_scratch_paths(profile)
    profile.isaaclab_path = str(repo)
    return profile


def _ssh_fake() -> FakeExecute:
    fake = FakeExecute()
    fake.add(
        "ssh -G",
        stdout="user alice\nhostname bonecho.example.edu\nport 22\ncontrolmaster auto\ncontrolpersist 10m\ncontrolpath /tmp/cm-%C\n",
    )
    fake.add(lambda args: len(args) >= 2 and args[0] == "ssh" and args[1] == "-O", stdout="Master running\n")
    fake.add("whoami", stdout="alice\n")
    fake.add("printf %s", stdout="/h/alice\n")
    fake.add("sinfo", stdout="a40_b1|gpu:a40:8|up\n")
    fake.add("scontrol", stdout="PartitionName=a40_b1 Gres=gpu:a40:8\n")
    fake.add("command -v singularity", stdout="/usr/bin/singularity\n")
    fake.add("mkdir -p", stdout="")
    fake.add("rev-parse", stdout=ISAACLAB_COMMIT + "\n")
    fake.add("--version", stdout="apptainer version 1.3.4\n")
    fake.add("image inspect", stdout="sha256:abc\n")
    return fake


def test_cli_new_commands() -> None:
    help_text = build_parser().format_help()
    for name in ("onboard", "deploy", "smoke-test", "auth", "bootstrap"):
        assert name in help_text
    args = parse_args(["onboard", "bonecho", "--isaaclab", "/opt/IsaacLab", "--no-clone", "--no-auth"])
    assert args.no_clone is True
    assert args.no_auth is True
    deploy = parse_args(["deploy", "--plan"])
    assert deploy.plan is True
    smoke = parse_args(["smoke-test", "--video", "--max-iterations", "50"])
    assert smoke.video is True
    assert smoke.max_iterations == 50


def test_bootstrap_never_sudo(tmp_path: Path) -> None:
    fake = FakeExecute()
    runner = CommandRunner(execute=fake)
    store = ConfigStore(tmp_path)
    BootstrapCommand(runner, store, start_dir=tmp_path).run(install=True)
    assert not any(c.args and c.args[0] == "sudo" for c in runner.calls)
    assert fake.calls == []


def test_platform_ubuntu_and_other() -> None:
    ubuntu = inspect_platform(os_release={"ID": "ubuntu", "PRETTY_NAME": "Ubuntu 24.04"}, system="Linux")
    assert ubuntu.supported_bootstrap is True
    other = inspect_platform(os_release={"ID": "darwin"}, system="Darwin")
    assert other.supported_bootstrap is False


def test_prereq_detection_has_required_tools() -> None:
    report = inspect_prereqs()
    names = {i.name for i in report.items}
    assert {"git", "ssh", "rsync", "docker", "apptainer", "Python"} <= names
    installed = [i for i in report.items if i.name in {"git", "ssh", "rsync"}]
    assert all(i.status == STATUS_INSTALLED for i in installed)


def test_repo_existing_vs_clone_required(tmp_path: Path) -> None:
    runner = CommandRunner(execute=FakeExecute())
    missing = find_or_clone(runner, configured=None, start=tmp_path, dest=tmp_path / "IsaacLab", clone=False)
    assert missing.action in {"MISSING", "WOULD_CLONE"}
    assert not any("clone" in " ".join(c.args) for c in runner.calls)

    repo = _isaaclab(tmp_path)
    fake = FakeExecute()
    fake.add("rev-parse", stdout=ISAACLAB_COMMIT + "\n")
    found = find_or_clone(CommandRunner(execute=fake), configured=repo, start=tmp_path, clone=True)
    assert found.action == "READY"
    assert not any(c[:2] == ["git", "clone"] for c in fake.calls)


def test_repo_version_mismatch_warns_no_reset(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    fake = FakeExecute()
    fake.add("rev-parse", stdout="deadbeef" * 5 + "\n")
    runner = CommandRunner(execute=fake)
    result = resolve_existing(repo, runner)
    assert result.action == "WARN"
    assert result.warning
    assert "reset" in result.warning.lower() or "without reset" in result.warning
    assert not any("reset" in " ".join(c.args) for c in runner.calls)
    match = classify_isaaclab_commit("deadbeef")
    assert match.classification == "compatible"


def test_ssh_multiplex_and_auth_states() -> None:
    mux = multiplex_from_ssh_g(
        "bonecho",
        "user alice\nhostname bonecho.example.edu\ncontrolmaster auto\ncontrolpersist 10m\ncontrolpath /tmp/cm\n",
    )
    assert mux.multiplexing_configured is True
    assert mux.hostname == "bonecho.example.edu"
    none = multiplex_from_ssh_g("bonecho", "user alice\nhostname x\ncontrolmaster no\n")
    assert none.multiplexing_configured is False

    store = ConfigStore(Path("/tmp"))  # unused
    fake = FakeExecute()
    fake.add("ssh -G", stdout="user alice\nhostname bonecho.example.edu\ncontrolmaster auto\n")
    fake.add(lambda args: args[:2] == ["ssh", "-O"], stdout="Master running\n")
    fake.add("whoami", stdout="alice\n")
    runner = CommandRunner(execute=fake)
    ready = AuthCommand(runner, store).run("bonecho", interactive=True)
    assert ready.action == "READY"
    assert not any(c.category == "ssh-auth" and c.args == ["ssh", "bonecho"] for c in runner.calls)

    fake2 = FakeExecute()
    fake2.add("ssh -G", stdout="user alice\nhostname bonecho.example.edu\ncontrolmaster no\n")
    fake2.add(lambda args: args[:2] == ["ssh", "-O"], returncode=255, stderr="No ControlPath")
    runner2 = CommandRunner(dry_run=True, execute=fake2)
    required = AuthCommand(runner2, store).run("bonecho", interactive=True)
    assert required.action == "REQUIRED"


def test_onboard_reuses_setup_and_is_idempotent(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    fake = _ssh_fake()
    runner = CommandRunner(execute=fake)
    cmd = OnboardCommand(runner, store, start_dir=tmp_path)
    cmd.run("bonecho", isaaclab=repo, clone=False, authenticate=True)
    assert store.profile_path("bonecho").is_file()
    generated = list((store.generated_dir).glob("*"))
    assert generated
    assert not any(c.args and c.args[0] == "sudo" for c in runner.calls)
    assert not any(".ssh/config" in " ".join(c.args) for c in runner.calls)

    fake2 = _ssh_fake()
    runner2 = CommandRunner(execute=fake2)
    cmd2 = OnboardCommand(runner2, store, start_dir=tmp_path)
    cmd2.run("bonecho", isaaclab=repo, clone=False, authenticate=True)
    assert store.load_profile("bonecho").isaaclab_path == str(repo.resolve())


def test_deploy_cache_hit_and_first_run(tmp_path: Path, capsys) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    store.ensure_local()
    profile = _profile(repo)
    store.save_profile(profile)
    store.set_active("bonecho")

    fake = FakeExecute()
    fake.add("--version", stdout="apptainer version 1.3.4\n")
    fake.add("image inspect", stdout="sha256:abc123\n")
    runner = CommandRunner(execute=fake)
    first = DeployCommand(runner, store, start_dir=repo).run(cluster="bonecho", image_profile="base", plan=True)
    assert first == 0
    out1 = capsys.readouterr().out
    assert "Docker build        REQUIRED" in out1
    assert "Conversion          REQUIRED" in out1
    assert "Upload              REQUIRED" in out1
    # No conversion/upload on --plan
    assert not any(c.args and c.args[0] == "rsync" for c in runner.calls)

    digest = "sha256:abc123def4567890"
    blob = b"tar"
    artifact = store.artifacts_dir / artifact_filename("base", digest)
    artifact.write_bytes(blob)
    version = "apptainer version 1.3.4"
    conv = conversion_fingerprint(
        built_image_digest=digest,
        backend="local-apptainer",
        version=version,
        options="--sandbox --fakeroot tar",
    )
    ImageStateStore(store).upsert(
        ImageArtifactState(
            image_profile="base",
            docker_image="isaac-lab-base:latest",
            build_input_fingerprint=fingerprint_repo_build_inputs(repo, "base"),
            built_image_digest=digest,
            conversion_fingerprint=conv,
            local_artifact=str(artifact),
            local_artifact_sha256="abc",
            remote_sha256="abc",
            remote_path="/scratch/alice/isaaclab-containers/isaac-lab-base.tar",
        )
    )
    fake2 = FakeExecute()
    fake2.add("--version", stdout=version + "\n")
    fake2.add("image inspect", stdout=digest + "\n")
    runner2 = CommandRunner(execute=fake2)
    code = DeployCommand(runner2, store, start_dir=repo).run(cluster="bonecho", plan=True)
    assert code == 0
    out2 = capsys.readouterr().out
    assert "Docker build        SKIPPED" in out2
    assert "Conversion          cached" in out2
    assert "LIKELY SKIPPED" in out2


def test_smoke_state_machine() -> None:
    ok = evaluate_job(JobStatus(job_id="1", state="COMPLETED", exit_code="0:0"))
    assert ok.passed and ok.phase == "COMPLETED"
    bad = evaluate_job(JobStatus(job_id="1", state="FAILED", exit_code="1:0"))
    assert not bad.passed
    run = evaluate_job(JobStatus(job_id="1", state="RUNNING"))
    assert not run.passed and run.phase == "RUNNING"


def test_smoke_test_success_mocked(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    profile = _profile(repo)
    store.save_profile(profile)
    store.set_active("bonecho")
    fake = FakeExecute()
    fake.add("CONTAINER_OK", stdout="CONTAINER_OK\n")
    fake.add("command -v sbatch", stdout="/usr/bin/sbatch\n")
    fake.add("submit_job_slurm.sh", stdout="Submitted batch job 42\n")
    fake.add("squeue", stdout="")
    fake.add(
        "sacct",
        stdout="42|train|COMPLETED|00:01:00|0:0|a40_b1|bn081\n",
    )
    runner = CommandRunner(execute=fake)
    code = SmokeTestCommand(runner, store, start_dir=repo).run(
        cluster="bonecho",
        video=False,
        wait=True,
        timeout=5,
        poll_interval=0,
        max_iterations=2,
        sleep=lambda _s: None,
    )
    assert code == 0
    assert store.find_job("42")


def test_doctor_ready_summary(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    profile = _profile(repo)
    store.save_profile(profile)
    store.set_active("bonecho")
    ImageStateStore(store).upsert(
        ImageArtifactState(
            image_profile="base",
            docker_image="isaac-lab-base:latest",
            built_image_digest="sha256:abc",
            conversion_fingerprint="conv",
            local_artifact=str(tmp_path / "a.tar"),
            remote_sha256="dead",
            remote_path="/scratch/alice/isaaclab-containers/isaac-lab-base.tar",
        )
    )
    (tmp_path / "a.tar").write_bytes(b"x")
    fake = _ssh_fake()
    runner = CommandRunner(execute=fake)
    results = Doctor(runner, store, cluster="bonecho", start_dir=repo).run()
    report = format_doctor_report(results)
    assert "Overall:" in report
    assert "READY" in report
    assert "Deployment:" in report
    assert "Docker image" in report
    assert "Converted artifact" in report
    assert "Remote artifact" in report


def test_version_classify() -> None:
    tested = classify_tool_version(name="Docker", found="29.6.1", tested="29.6.1")
    assert tested.classification == "tested"
    compat = classify_tool_version(name="Docker", found="24.0.0", tested="29.6.1")
    assert compat.classification == "compatible"
    unknown = classify_tool_version(name="SLURM", found=None, tested="x")
    assert unknown.classification == "unknown"
