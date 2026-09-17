from pathlib import Path

from vector_lab.images.fingerprint import (
    collect_build_input_files,
    conversion_fingerprint,
    fingerprint_repo_build_inputs,
    parse_dockerfile_copy_sources,
)
from vector_lab.images.naming import parse_docker_inspect_digest


def _docker_tree(root: Path) -> None:
    docker = root / "docker"
    docker.mkdir(parents=True)
    (docker / "Dockerfile.base").write_text(
        "FROM isaac-sim:5.1.0\nCOPY docker/.ros/ /root/.ros/\n"
    )
    (docker / "docker-compose.yaml").write_text("services: {}\n")
    (docker / ".env.base").write_text("ACCEPT_EULA=Y\n")
    (root / ".dockerignore").write_text("*.pyc\n")
    (root / "isaaclab.sh").write_text("#!/bin/bash\n")
    (root / "tools").mkdir()
    (root / "tools/install_deps.py").write_text("print('deps')\n")
    (root / "source/isaaclab").mkdir(parents=True)
    (root / "source/isaaclab/setup.py").write_text("from setuptools import setup\nsetup()\n")
    (root / "source/isaaclab/module.py").write_text("x = 1\n")
    (root / "scripts").mkdir()
    (root / "scripts/task.py").write_text("print('task')\n")


def test_source_python_does_not_change_build_fingerprint(tmp_path: Path) -> None:
    _docker_tree(tmp_path)
    first = fingerprint_repo_build_inputs(tmp_path, "base")
    (tmp_path / "source/isaaclab/module.py").write_text("x = 999\n")
    (tmp_path / "scripts/task.py").write_text("print('changed')\n")
    second = fingerprint_repo_build_inputs(tmp_path, "base")
    assert first == second
    files = collect_build_input_files(tmp_path, "base")
    assert "source/isaaclab/module.py" not in files
    assert "scripts/task.py" not in files


def test_install_script_change_invalidates_fingerprint(tmp_path: Path) -> None:
    _docker_tree(tmp_path)
    first = fingerprint_repo_build_inputs(tmp_path, "base")
    (tmp_path / "tools/install_deps.py").write_text("print('deps changed')\n")
    second = fingerprint_repo_build_inputs(tmp_path, "base")
    assert first != second


def test_setup_py_change_invalidates_fingerprint(tmp_path: Path) -> None:
    _docker_tree(tmp_path)
    first = fingerprint_repo_build_inputs(tmp_path, "base")
    (tmp_path / "source/isaaclab/setup.py").write_text("from setuptools import setup\nsetup(name='x')\n")
    second = fingerprint_repo_build_inputs(tmp_path, "base")
    assert first != second


def test_docker_input_change_updates_fingerprint(tmp_path: Path) -> None:
    _docker_tree(tmp_path)
    first = fingerprint_repo_build_inputs(tmp_path, "base")
    (tmp_path / "docker/Dockerfile.base").write_text("FROM isaac-sim:5.1.0\nRUN echo deps\n")
    second = fingerprint_repo_build_inputs(tmp_path, "base")
    assert first != second


def test_parse_dockerfile_copy_skips_parent_tree() -> None:
    text = "COPY ../ /workspace/isaaclab\nCOPY docker/.ros/ /root/.ros/\n"
    assert parse_dockerfile_copy_sources(text) == ["docker/.ros/"]


def test_parse_docker_inspect_digest() -> None:
    assert (
        parse_docker_inspect_digest("isaac-lab-base@sha256:abcdef0123456789")
        == "sha256:abcdef0123456789"
    )
    assert parse_docker_inspect_digest("sha256:deadbeef") == "sha256:deadbeef"


def test_conversion_fingerprint_includes_digest_and_backend() -> None:
    a = conversion_fingerprint(
        built_image_digest="sha256:aaa",
        backend="local-apptainer",
        version="1.3.4",
        options="--sandbox --fakeroot tar",
    )
    b = conversion_fingerprint(
        built_image_digest="sha256:bbb",
        backend="local-apptainer",
        version="1.3.4",
        options="--sandbox --fakeroot tar",
    )
    assert a != b
