from pathlib import Path

from vector_sim.config.store import ConfigStore
from vector_sim.exec import CommandRunner
from vector_sim.images.build import BuildPipeline
from vector_sim.images.fingerprint import fingerprint_repo_build_inputs
from vector_sim.images.naming import docker_image_ref
from vector_sim.images.state import ImageArtifactState, ImageStateStore
from tests.fakes.runner import FakeExecute


def _isaaclab(tmp_path: Path) -> Path:
    root = tmp_path / "IsaacLab"
    (root / "docker/cluster").mkdir(parents=True)
    (root / "source/isaaclab").mkdir(parents=True)
    (root / "docker/container.py").write_text("print('build')\n")
    (root / "docker/cluster/cluster_interface.sh").write_text("#!/bin/bash\n")
    (root / "isaaclab.sh").write_text("#!/bin/bash\n")
    (root / "docker/Dockerfile.base").write_text("FROM isaac-sim\n")
    (root / "docker/docker-compose.yaml").write_text("services: {}\n")
    (root / "docker/.env.base").write_text("ACCEPT_EULA=Y\n")
    return root


def test_build_cache_hit_skips_docker_build(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    store.ensure_local()
    fp = fingerprint_repo_build_inputs(repo, "base")
    ImageStateStore(store).upsert(
        ImageArtifactState(
            image_profile="base",
            docker_image=docker_image_ref("base"),
            build_input_fingerprint=fp,
            built_image_digest="sha256:abc123",
        )
    )
    fake = FakeExecute()
    fake.add("image inspect", stdout="sha256:abc123\n")
    runner = CommandRunner(execute=fake)
    outcome = BuildPipeline(runner, ImageStateStore(store)).run(
        repo=repo, image_profile="base", preflight=False
    )
    assert outcome.action == "SKIPPED"
    assert not any("container.py" in " ".join(c) for c in fake.calls)


def test_build_force_runs_container_py(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    store.ensure_local()
    fp = fingerprint_repo_build_inputs(repo, "base")
    ImageStateStore(store).upsert(
        ImageArtifactState(
            image_profile="base",
            docker_image="isaac-lab-base:latest",
            build_input_fingerprint=fp,
            built_image_digest="sha256:abc123",
        )
    )
    fake = FakeExecute()
    fake.add("image inspect", stdout="isaac-lab-base@sha256:abc123def456\n")
    fake.add("container.py", stdout="built\n")
    runner = CommandRunner(execute=fake)
    outcome = BuildPipeline(runner, ImageStateStore(store)).run(
        repo=repo, image_profile="base", force=True, preflight=False
    )
    assert outcome.action == "BUILT"
    assert any("container.py" in " ".join(c) and "build" in c for c in fake.calls)
    assert outcome.built_image_digest == "sha256:abc123def456"


def test_build_dry_run_does_not_execute_build(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    fake = FakeExecute()
    runner = CommandRunner(dry_run=True, execute=fake)
    outcome = BuildPipeline(runner, ImageStateStore(store)).run(
        repo=repo, image_profile="base", preflight=False
    )
    assert outcome.action == "REQUIRED"
    assert fake.calls == []
    assert any(call.category == "docker-build" for call in runner.calls) is False or True
    # dry-run returns before _build, so no docker-build category
    assert not any(call.category == "docker-build" for call in runner.calls)
