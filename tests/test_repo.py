from pathlib import Path

import pytest

from vector_lab.errors import DiscoveryError
from vector_lab.local.repo import discover_isaac_lab, is_isaac_lab_repo


def _touch_isaac_lab(root: Path) -> None:
    (root / "docker/cluster").mkdir(parents=True)
    (root / "source/isaaclab").mkdir(parents=True)
    (root / "docker/container.py").write_text("# container\n")
    (root / "docker/cluster/cluster_interface.sh").write_text("#!/bin/bash\n")
    (root / "isaaclab.sh").write_text("#!/bin/bash\n")


def test_configured_path(tmp_path: Path) -> None:
    repo = tmp_path / "IsaacLab"
    _touch_isaac_lab(repo)
    found = discover_isaac_lab(configured=repo)
    assert found.root == repo.resolve()


def test_search_parents(tmp_path: Path) -> None:
    repo = tmp_path / "IsaacLab"
    _touch_isaac_lab(repo)
    nested = repo / "source" / "isaaclab_tasks"
    nested.mkdir(parents=True)
    found = discover_isaac_lab(start=nested)
    assert found.root == repo.resolve()


def test_rejects_incomplete_tree(tmp_path: Path) -> None:
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker/container.py").write_text("x")
    assert is_isaac_lab_repo(tmp_path) is False
    with pytest.raises(DiscoveryError):
        discover_isaac_lab(configured=tmp_path)


def test_missing_configured_path(tmp_path: Path) -> None:
    with pytest.raises(DiscoveryError):
        discover_isaac_lab(configured=tmp_path / "nope")
