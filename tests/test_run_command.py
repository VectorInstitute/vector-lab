from pathlib import Path

from vector_sim.cluster.adapters import VectorSlurmAdapter
from vector_sim.commands.run import RunCommand
from vector_sim.config.models import bonecho_defaults
from vector_sim.config.store import ConfigStore
from vector_sim.exec import CommandRunner
from vector_sim.errors import VectorSimError
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
    (root / "docker/.env.base").write_text("ACCEPT_EULA=Y\nDOCKER_ISAACSIM_ROOT_PATH=/isaac-sim\nDOCKER_USER_HOME=/root\n")
    return root


def _store(tmp_path: Path, repo: Path) -> ConfigStore:
    store = ConfigStore(tmp_path / "cfg")
    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.scratch_dir = "/scratch/alice"
    VectorSlurmAdapter().derive_scratch_paths(profile)
    profile.isaaclab_path = str(repo)
    store.save_profile(profile)
    store.set_active("bonecho")
    return store


def test_run_dry_run_does_not_submit(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = _store(tmp_path, repo)
    fake = FakeExecute()
    runner = CommandRunner(dry_run=True, execute=fake)
    code = RunCommand(runner, store, start_dir=repo).run(task="Isaac-Cartpole-v0", video=True)
    assert code == 0
    assert fake.calls == []
    assert not any(c.category == "sbatch" for c in runner.calls)
    assert store.load_jobs() == []
    # dry-run must not stage wrappers into the Isaac Lab checkout
    assert not (repo / ".vector-sim/generated/submit_job_slurm.sh").is_file()


def test_missing_remote_container_clear_error(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = _store(tmp_path, repo)
    fake = FakeExecute()
    fake.add("CONTAINER_OK", stdout="")  # won't match; default returns empty ok
    # Make the test -f probe fail
    fake.add(
        lambda args: "test -f" in " ".join(args),
        stdout="",
        returncode=0,
    )
    runner = CommandRunner(execute=fake)
    try:
        RunCommand(runner, store, start_dir=repo).run(task="Isaac-Cartpole-v0")
        assert False, "expected error"
    except VectorSimError as exc:
        assert "container not deployed" in str(exc)
        assert "vector-sim deploy" in (exc.suggestion or "")


def test_ssh_denial_is_not_reported_as_missing_container(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = _store(tmp_path, repo)
    fake = FakeExecute()
    fake.add(
        "test -f",
        stdout="",
        stderr="mollysun@bonecho.vectorinstitute.ai: Permission denied (keyboard-interactive).\n",
        returncode=255,
    )
    runner = CommandRunner(execute=fake)
    try:
        RunCommand(runner, store, start_dir=repo).run(task="Isaac-Cartpole-v0")
        assert False, "expected error"
    except VectorSimError as exc:
        assert exc.category == "ssh"
        assert "vector-sim auth" in (exc.suggestion or "")
        assert "container not deployed" not in str(exc)


def test_shell_dry_run_uses_alias(tmp_path: Path, capsys) -> None:
    from vector_sim.commands.videos import ShellCommand

    repo = _isaaclab(tmp_path)
    store = _store(tmp_path, repo)
    runner = CommandRunner(dry_run=True, execute=FakeExecute())
    ShellCommand(runner, store).run(cluster="bonecho")
    out = capsys.readouterr().out
    assert "ssh bonecho" in out
