"""Regression tests for conversion packaging path handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from vector_sim.config.store import ConfigStore
from vector_sim.errors import CommandError, VectorSimError
from vector_sim.exec import CommandResult, CommandRunner
from vector_sim.images.conversion import ConversionSpec, LocalApptainerConversion
from vector_sim.images.state import ImageArtifactState, ImageStateStore


def test_tar_receives_absolute_output_path_with_work_root_cwd(tmp_path: Path) -> None:
    """Relative project-style artifact paths must become absolute before tar cwd=work_root."""
    project = tmp_path / "proj"
    work = project / ".vector-sim" / "work" / "base"
    artifacts = project / ".vector-sim" / "artifacts"
    work.mkdir(parents=True)
    # Simulate a completed sandbox (no apptainer rebuild).
    (work / "isaac-lab-base.sif").mkdir()
    (work / "isaac-lab-base.sif" / "bin").mkdir()
    (work / "isaac-lab-base.sif" / "bin" / "true").write_text("x")

    recorded: list[tuple[list[str], str | None]] = []

    def _exec(args, **kwargs):
        recorded.append((list(args), kwargs.get("cwd")))
        if args[:1] == ["tar"]:
            out = Path(args[args.index("-cf") + 1])
            assert out.is_absolute()
            assert not str(out).startswith(str(work))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"archive")
            return CommandResult(args=list(args), returncode=0, category="tar")
        if "--version" in " ".join(args):
            return CommandResult(args=list(args), returncode=0, stdout="apptainer version 1.3.4\n")
        return CommandResult(args=list(args), returncode=0, category="apptainer-build")

    runner = CommandRunner(execute=_exec)
    backend = LocalApptainerConversion(runner, work_root=work)
    # Intentionally relative to project root (the bug class from Step 3A).
    relative_out = Path(".vector-sim") / "artifacts" / "isaac-lab-base-deadbeef.tar"
    # Resolve relative to project by chdir simulation: pass path as project-relative Path
    # constructed under project, then pass as relative string-like Path from project cwd.
    # Use a path relative to tmp_path that would be wrong under work_root.
    rel = Path(".vector-sim/artifacts/isaac-lab-base-deadbeef.tar")
    # Create the relative path meaning from project: we pass Path that is not absolute.
    # package_sandbox resolves against process cwd, so run from project.
    import os

    old = os.getcwd()
    try:
        os.chdir(project)
        spec = ConversionSpec(
            docker_image="isaac-lab-base:latest",
            output_tar=rel,
            sandbox_name="isaac-lab-base.sif",
            image_digest="sha256:abc",
            apptainer_version="apptainer version 1.3.4",
        )
        result = backend.package_sandbox(spec, cleanup_sandbox=False)
    finally:
        os.chdir(old)

    expected = (artifacts / "isaac-lab-base-deadbeef.tar").resolve()
    assert result.output_tar == expected
    assert expected.is_file()
    tar_calls = [c for c in recorded if c[0][:1] == ["tar"]]
    assert len(tar_calls) == 1
    args, cwd = tar_calls[0]
    assert Path(args[args.index("-cf") + 1]).is_absolute()
    assert Path(args[args.index("-cf") + 1]) == expected
    assert Path(cwd).resolve() == work.resolve()
    # Parent was created by packaging.
    assert artifacts.is_dir()


def test_package_creates_missing_artifact_parent(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "box.sif").mkdir(parents=True)
    (work / "box.sif" / "x").write_text("1")
    dest = tmp_path / "nested" / "out" / "box.tar"
    assert not dest.parent.exists()

    def _exec(args, **kwargs):
        if args[:1] == ["tar"]:
            out = Path(args[args.index("-cf") + 1])
            out.write_bytes(b"ok")
            return CommandResult(args=list(args), returncode=0, category="tar")
        return CommandResult(args=list(args), returncode=0, stdout="apptainer version 1.0\n")

    backend = LocalApptainerConversion(CommandRunner(execute=_exec), work_root=work)
    backend.package_sandbox(
        ConversionSpec(
            docker_image="isaac-lab-base:latest",
            output_tar=dest,
            sandbox_name="box.sif",
            image_digest="sha256:1",
            apptainer_version="apptainer version 1.0",
        )
    )
    assert dest.is_file()
    assert dest.parent.is_dir()


def test_packaging_failure_does_not_write_conversion_state(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "cfg")
    store.ensure_local()
    work = store.work_dir / "base"
    sandbox = work / "isaac-lab-base.sif"
    sandbox.mkdir(parents=True)
    (sandbox / "marker").write_text("keep")
    state = ImageStateStore(store)
    state.upsert(
        ImageArtifactState(
            image_profile="base",
            docker_image="isaac-lab-base:latest",
            built_image_digest="sha256:abc",
            conversion_fingerprint=None,
        )
    )

    def _exec(args, **kwargs):
        if args[:1] == ["tar"]:
            return CommandResult(args=list(args), returncode=2, stderr="Cannot open\n", category="tar")
        return CommandResult(args=list(args), returncode=0, stdout="apptainer version 1.0\n")

    backend = LocalApptainerConversion(CommandRunner(execute=_exec), work_root=work)
    dest = store.artifacts_dir / "isaac-lab-base-abc.tar"
    with pytest.raises(CommandError):
        result = backend.package_sandbox(
            ConversionSpec(
                docker_image="isaac-lab-base:latest",
                output_tar=dest,
                sandbox_name="isaac-lab-base.sif",
                image_digest="sha256:abc",
                apptainer_version="apptainer version 1.0",
            )
        )
        # Callers must only persist state after success; simulate that guard:
        if False:  # pragma: no cover
            state.upsert(
                ImageArtifactState(
                    image_profile="base",
                    conversion_fingerprint=result.conversion_fingerprint,
                    local_artifact=str(result.output_tar),
                )
            )

    loaded = state.get("base")
    assert loaded is not None
    assert loaded.conversion_fingerprint is None
    assert loaded.local_artifact is None
    assert not dest.exists()


def test_packaging_failure_preserves_sandbox(tmp_path: Path) -> None:
    work = tmp_path / "work"
    sandbox = work / "isaac-lab-base.sif"
    sandbox.mkdir(parents=True)
    (sandbox / "keep").write_text("yes")

    def _exec(args, **kwargs):
        if args[:1] == ["tar"]:
            return CommandResult(args=list(args), returncode=2, stderr="fail\n", category="tar")
        return CommandResult(args=list(args), returncode=0, stdout="apptainer version 1.0\n")

    backend = LocalApptainerConversion(CommandRunner(execute=_exec), work_root=work)
    with pytest.raises(CommandError):
        backend.package_sandbox(
            ConversionSpec(
                docker_image="isaac-lab-base:latest",
                output_tar=tmp_path / "out.tar",
                sandbox_name="isaac-lab-base.sif",
                image_digest="sha256:abc",
                apptainer_version="apptainer version 1.0",
            ),
            cleanup_sandbox=True,
        )
    assert sandbox.is_dir()
    assert (sandbox / "keep").read_text() == "yes"


def test_paths_with_spaces_passed_as_argv_not_shell_string(tmp_path: Path) -> None:
    work = tmp_path / "work dir"
    sandbox = work / "my sandbox.sif"
    sandbox.mkdir(parents=True)
    (sandbox / "x").write_text("1")
    dest = tmp_path / "art ifacts" / "out file.tar"

    seen: list[list[str]] = []

    def _exec(args, **kwargs):
        seen.append(list(args))
        if args[:1] == ["tar"]:
            out = Path(args[2])
            assert " " in str(out)
            assert args == ["tar", "-cf", str(out), "my sandbox.sif"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"z")
            return CommandResult(args=list(args), returncode=0, category="tar")
        return CommandResult(args=list(args), returncode=0, stdout="apptainer version 1.0\n")

    backend = LocalApptainerConversion(CommandRunner(execute=_exec), work_root=work)
    backend.package_sandbox(
        ConversionSpec(
            docker_image="isaac-lab-base:latest",
            output_tar=dest,
            sandbox_name="my sandbox.sif",
            image_digest="sha256:abc",
            apptainer_version="apptainer version 1.0",
        )
    )
    tar_args = next(a for a in seen if a[:1] == ["tar"])
    assert tar_args[2] == str(dest.resolve())
    assert tar_args[3] == "my sandbox.sif"
    # Not a single shell-concatenated string
    assert all(isinstance(x, str) for x in tar_args)
    assert len(tar_args) == 4


def test_convert_full_flow_uses_absolute_tar_path(tmp_path: Path) -> None:
    work = tmp_path / "work"

    def _exec(args, **kwargs):
        if args[:1] == ["tar"]:
            out = Path(args[args.index("-cf") + 1])
            assert out.is_absolute()
            assert kwargs.get("cwd") == str(work.resolve())
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"data")
            return CommandResult(args=list(args), returncode=0, category="tar")
        if "build" in args and "apptainer" in args[0]:
            # Create sandbox as apptainer would
            (work / "isaac-lab-base.sif").mkdir(parents=True, exist_ok=True)
            return CommandResult(args=list(args), returncode=0, category="apptainer-build")
        return CommandResult(args=list(args), returncode=0, stdout="apptainer version 1.0\n")

    backend = LocalApptainerConversion(CommandRunner(execute=_exec), work_root=work)
    rel_style = tmp_path / "artifacts" / "out.tar"
    result = backend.convert(
        ConversionSpec(
            docker_image="isaac-lab-base:latest",
            output_tar=rel_style,
            sandbox_name="isaac-lab-base.sif",
            image_digest="sha256:abc",
            apptainer_version="apptainer version 1.0",
        )
    )
    assert result.output_tar.is_absolute()
    assert result.output_tar.is_file()


def test_package_missing_sandbox_errors(tmp_path: Path) -> None:
    backend = LocalApptainerConversion(CommandRunner(execute=lambda *a, **k: CommandResult(args=[], returncode=0)), work_root=tmp_path / "work")
    with pytest.raises(VectorSimError, match="sandbox missing"):
        backend.package_sandbox(
            ConversionSpec(
                docker_image="isaac-lab-base:latest",
                output_tar=tmp_path / "x.tar",
                sandbox_name="missing.sif",
                image_digest="sha256:abc",
                apptainer_version="apptainer version 1.0",
            )
        )
