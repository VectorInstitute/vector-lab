from vector_lab.config.models import ClusterProfile, Fingerprints, bonecho_defaults
from vector_lab.config.store import ConfigStore
from vector_lab.exec import CommandRunner
from vector_lab.ssh.session import SshSession, wrap_login_shell
from tests.fakes.runner import FakeExecute


def test_ssh_session_uses_login_shell_wrapper() -> None:
    fake = FakeExecute()
    runner = CommandRunner(execute=fake)
    session = SshSession("bonecho", runner, login_shell=True, batch_mode=True)
    session.exec("squeue -u alice", category="slurm", dry_run_skip=False)
    args = fake.calls[0]
    assert args[0] == "ssh"
    assert "bonecho" in args
    assert args[-1] == wrap_login_shell("squeue -u alice")
    assert "BatchMode=yes" in args


def test_ssh_session_can_skip_login_shell() -> None:
    fake = FakeExecute()
    runner = CommandRunner(execute=fake)
    session = SshSession("bonecho", runner, login_shell=False)
    session.exec("whoami", login_shell=False)
    assert fake.calls[0][-1] == "whoami"


def test_profile_yaml_roundtrip(tmp_path) -> None:
    store = ConfigStore(tmp_path)
    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.home_dir = "/h/alice"
    profile.scratch_dir = "/scratch/alice"
    profile.fingerprints = Fingerprints(build_input_fingerprint="abc")
    store.save_profile(profile)
    loaded = store.load_profile("bonecho")
    assert loaded.remote_user == "alice"
    assert loaded.apptainer.writable_mode == "writable-tmpfs"
    assert loaded.apptainer.extra_exec_args == ["--nv", "--containall", "--writable-tmpfs"]
    assert loaded.fingerprints.build_input_fingerprint == "abc"
    assert loaded.scheduler.remote_shell == "login"
    assert loaded.cluster_type == "vector-slurm"


def test_override_preservation(tmp_path) -> None:
    store = ConfigStore(tmp_path)
    existing = bonecho_defaults()
    existing.scratch_dir = "/custom/scratch"
    existing.user_set = ["scratch_dir"]
    existing.remote_user = "olduser"
    store.save_profile(existing)

    detected = bonecho_defaults()
    detected.scratch_dir = "/scratch/alice"
    detected.remote_user = "alice"
    detected.home_dir = "/h/alice"
    store.save_profile_preserving_overrides(detected)

    loaded = store.load_profile("bonecho")
    assert loaded.scratch_dir == "/custom/scratch"
    assert loaded.remote_user == "alice"
    assert loaded.home_dir == "/h/alice"


def test_jobs_json_roundtrip(tmp_path) -> None:
    store = ConfigStore(tmp_path)
    from vector_lab.config.models import JobRecord

    record = JobRecord.create(
        job_id="123",
        cluster="bonecho",
        remote_run_dir="/scratch/alice/isaaclab_20260101",
        slurm_log="/scratch/alice/isaaclab/logs/slurm-123.out",
        task="Isaac-Cartpole-v0",
        image_profile="base",
    )
    store.upsert_job(record)
    found = store.find_job("123")
    assert found is not None
    assert found.task == "Isaac-Cartpole-v0"
    assert found.cluster == "bonecho"
