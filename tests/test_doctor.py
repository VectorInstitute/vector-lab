from vector_sim.cluster.detect import infer_home, infer_scratch, parse_env_assignments
from vector_sim.commands.doctor import Doctor, Severity, classify_result, format_doctor_report
from vector_sim.config.models import bonecho_defaults
from vector_sim.config.store import ConfigStore
from vector_sim.exec import CommandRunner
from tests.fakes.runner import FakeExecute


def test_classify_result() -> None:
    assert classify_result(passed=True, required=True) == Severity.INFO
    assert classify_result(passed=True, required=False) == Severity.INFO
    assert classify_result(passed=False, required=True) == Severity.ERROR
    assert classify_result(passed=False, required=False) == Severity.WARNING


def test_generic_scratch_not_hardcoded() -> None:
    path, source = infer_scratch(env={"SCRATCH": "/gpfs/scratch/alice"})
    assert path == "/gpfs/scratch/alice"
    assert source == "SCRATCH"
    path, source = infer_scratch(env={})
    assert path is None
    assert source == "unresolved"
    path, _source = infer_scratch(existing_directories=["/fast/alice"])
    assert path == "/fast/alice"
    assert infer_home(printenv_home="/u/alice") == "/u/alice"


def test_parse_env_assignments() -> None:
    env = parse_env_assignments("SCRATCH=/scratch/alice\nHOME=/h/alice\n")
    assert env["SCRATCH"] == "/scratch/alice"


def test_doctor_dry_run_with_profile(tmp_path) -> None:
    store = ConfigStore(tmp_path)
    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.scratch_dir = "/scratch/alice"
    profile.home_dir = "/h/alice"
    store.save_profile(profile)
    store.set_active("bonecho")

    fake = FakeExecute()
    fake.add(
        "ssh -G",
        stdout="user alice\nhostname bonecho.example.edu\nport 22\n",
    )
    runner = CommandRunner(dry_run=True, execute=fake)
    results = Doctor(runner, store, cluster="bonecho", start_dir=tmp_path).run()
    report = format_doctor_report(results)
    assert "SSH:" in report
    assert "bonecho" in report
    assert "[INFO]" in report
    assert "[ERROR]" in report or "Isaac Lab" in report
    connection = next(r for r in results if r.name == "connection")
    assert connection.status == "skipped"
    assert fake.calls  # ssh -G runs even in dry-run
    assert not any("whoami" in " ".join(c) for c in fake.calls)


def test_doctor_missing_profile(tmp_path) -> None:
    store = ConfigStore(tmp_path)
    runner = CommandRunner(dry_run=True, execute=FakeExecute())
    results = Doctor(runner, store, cluster=None, start_dir=tmp_path).run()
    assert any(r.section == "SSH" and r.severity.value == "ERROR" for r in results)
