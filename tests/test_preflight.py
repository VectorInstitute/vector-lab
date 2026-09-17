from pathlib import Path

import pytest

from vector_lab.images.naming import safe_rmtree
from vector_lab.images.preflight import apptainer_preflight, disk_preflight
from vector_lab.local.docker_util import DOCKER_ACCESS_HINT, DOCKER_MISSING_HINT, fakeroot_primary_group_problem, permission_denied
from vector_lab.exec import CommandRunner
from tests.fakes.runner import FakeExecute


def test_docker_permission_diagnostic() -> None:
    assert permission_denied("Got permission denied while trying to connect to the Docker daemon")
    assert "newgrp docker" in DOCKER_ACCESS_HINT
    assert "usermod" in DOCKER_ACCESS_HINT
    assert "docs.docker.com" in DOCKER_MISSING_HINT


def test_fakeroot_preflight_when_docker_is_primary_group() -> None:
    assert fakeroot_primary_group_problem("docker")
    assert fakeroot_primary_group_problem("staff") is None
    fake = FakeExecute()
    fake.add("apptainer --version", stdout="apptainer version 1.3.4\n")
    issues = apptainer_preflight(CommandRunner(execute=fake), group_name="docker")
    assert any(i.severity == "ERROR" and "Primary group is docker" in i.message for i in issues)


def test_safe_rmtree_refuses_unrelated_paths(tmp_path: Path) -> None:
    allowed = tmp_path / "work"
    allowed.mkdir()
    other = tmp_path / "nope"
    other.mkdir()
    with pytest.raises(Exception):
        safe_rmtree(other, allowed_root=allowed)
    victim = allowed / "sandbox"
    victim.mkdir()
    safe_rmtree(victim, allowed_root=allowed)
    assert not victim.exists()


def test_disk_preflight_warning() -> None:
    issues = disk_preflight(Path("/"), min_bytes=10**18, free_bytes=100)
    assert issues and issues[0].severity == "WARNING"
