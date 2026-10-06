import hashlib
from pathlib import Path

from vector_sim.cluster.adapters import VectorSlurmAdapter
from vector_sim.config.models import bonecho_defaults
from vector_sim.config.store import ConfigStore
from vector_sim.exec import CommandResult, CommandRunner
from vector_sim.images.conversion import ConversionSpec, LocalApptainerConversion
from vector_sim.images.fingerprint import conversion_fingerprint, fingerprint_repo_build_inputs
from vector_sim.images.naming import artifact_filename
from vector_sim.images.push import PushPipeline
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


def _profile(isaaclab: Path):
    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.scratch_dir = "/scratch/alice"
    VectorSlurmAdapter().derive_scratch_paths(profile)
    profile.isaaclab_path = str(isaaclab)
    return profile


def test_conversion_creates_tar_and_fingerprint(tmp_path: Path) -> None:
    fake = FakeExecute()
    work = tmp_path / "work"

    def _exec(args, **kwargs):
        joined = " ".join(args)
        if args[:1] == ["tar"] or (len(args) > 1 and args[0] == "tar"):
            out = Path(args[args.index("-cf") + 1])
            assert out.is_absolute()
            out.write_bytes(b"sif-archive")
            return CommandResult(args=list(args), returncode=0, category="tar")
        if "--version" in joined:
            return CommandResult(args=list(args), returncode=0, stdout="apptainer version 1.3.4\n")
        # Simulate apptainer creating the sandbox directory.
        (work / "isaac-lab-base.sif").mkdir(parents=True, exist_ok=True)
        return CommandResult(args=list(args), returncode=0, category="apptainer-build")

    runner = CommandRunner(execute=_exec)
    backend = LocalApptainerConversion(runner, work_root=work)
    dest = tmp_path / "out" / "isaac-lab-base.tar"
    spec = ConversionSpec(
        docker_image="isaac-lab-base:latest",
        output_tar=dest,
        sandbox_name="isaac-lab-base.sif",
        image_digest="sha256:abc",
        apptainer_version="apptainer version 1.3.4",
    )
    result = backend.convert(spec)
    assert dest.is_file()
    assert result.output_tar == dest.resolve()
    assert result.conversion_fingerprint == backend.compute_fingerprint(spec)
    assert any("apptainer" in " ".join(c.args) and "build" in c.args for c in runner.calls)
    tar_call = next(c for c in runner.calls if c.args[:1] == ["tar"])
    assert Path(tar_call.args[tar_call.args.index("-cf") + 1]).is_absolute()
    assert Path(tar_call.cwd).resolve() == work.resolve()


def test_push_skips_upload_on_checksum_match(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    store.ensure_local()
    blob = b"container-bytes"
    sha = hashlib.sha256(blob).hexdigest()
    digest = "sha256:abc123def4567890"
    artifact = store.artifacts_dir / artifact_filename("base", digest)
    artifact.write_bytes(blob)
    conv = conversion_fingerprint(
        built_image_digest=digest,
        backend="local-apptainer",
        version="unknown",
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
            local_artifact_size=len(blob),
            local_artifact_sha256=sha,
        )
    )
    fake = FakeExecute()
    fake.add("image inspect", stdout=digest + "\n")
    fake.add("sha256sum", stdout=f"{sha}  /scratch/alice/isaaclab-containers/isaac-lab-base.tar\n")
    runner = CommandRunner(execute=fake)
    backend = LocalApptainerConversion(runner, work_root=store.work_dir / "base")
    outcome = PushPipeline(runner, ImageStateStore(store), backend=backend).run(
        profile=_profile(repo),
        repo=repo,
        image_profile="base",
        preflight=False,
    )
    assert outcome.upload_action == "SKIPPED"
    assert not any(c[0] == "rsync" for c in fake.calls)


def test_push_uploads_to_partial_then_renames(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    store.ensure_local()
    blob = b"container-bytes"
    sha = hashlib.sha256(blob).hexdigest()
    digest = "sha256:abc123def4567890"
    artifact = store.artifacts_dir / artifact_filename("base", digest)
    artifact.write_bytes(blob)
    conv = conversion_fingerprint(
        built_image_digest=digest,
        backend="local-apptainer",
        version="unknown",
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
            local_artifact_sha256=sha,
        )
    )
    fake = FakeExecute()
    fake.add("image inspect", stdout=digest + "\n")
    fake.add(lambda args: "sha256sum" in " ".join(args) and ".partial" in " ".join(args), stdout=f"{sha}  file\n")
    fake.add("sha256sum", stdout="")  # final artifact missing
    fake.add("rsync", stdout="sent\n")
    fake.add("mv -f", stdout="")
    runner = CommandRunner(execute=fake)
    backend = LocalApptainerConversion(runner, work_root=store.work_dir / "base")
    outcome = PushPipeline(runner, ImageStateStore(store), backend=backend).run(
        profile=_profile(repo),
        repo=repo,
        image_profile="base",
        preflight=False,
    )
    assert outcome.upload_action == "UPLOADED"
    joined = [" ".join(c) for c in fake.calls]
    assert any(item.startswith("rsync") and "bonecho:" in item and ".partial" in item for item in joined)
    assert any("mv -f" in item and "isaac-lab-base.tar" in item for item in joined)


def test_push_dry_run_does_not_rsync_or_convert(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    fake = FakeExecute()
    runner = CommandRunner(dry_run=True, execute=fake)
    backend = LocalApptainerConversion(runner, work_root=store.work_dir / "base")
    outcome = PushPipeline(runner, ImageStateStore(store), backend=backend).run(
        profile=_profile(repo),
        repo=repo,
        image_profile="base",
        preflight=False,
    )
    assert outcome.build_action == "REQUIRED"
    assert outcome.conversion_action == "REQUIRED"
    assert outcome.upload_action == "REQUIRED"
    # Version probe may run under dry-run for fingerprint stability; no rsync/tar mutate.
    assert all("--version" in " ".join(c) for c in fake.calls)
    assert not any(call.category == "rsync" for call in runner.calls)
    assert not any(call.category == "tar" for call in runner.calls)
    assert any("rsync" in note for note in outcome.notes)


def test_push_dry_run_likely_skipped_when_only_local_cache_matches(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = ConfigStore(tmp_path / "cfg")
    store.ensure_local()
    blob = b"container-bytes"
    sha = hashlib.sha256(blob).hexdigest()
    digest = "sha256:abc123def4567890"
    artifact = store.artifacts_dir / artifact_filename("base", digest)
    artifact.write_bytes(blob)
    conv = conversion_fingerprint(
        built_image_digest=digest,
        backend="local-apptainer",
        version="unknown",
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
            local_artifact_sha256=sha,
            remote_sha256=sha,
            remote_path="/scratch/alice/isaaclab-containers/isaac-lab-base.tar",
        )
    )
    runner = CommandRunner(dry_run=True, execute=FakeExecute())
    backend = LocalApptainerConversion(runner, work_root=store.work_dir / "base")
    outcome = PushPipeline(runner, ImageStateStore(store), backend=backend).run(
        profile=_profile(repo),
        repo=repo,
        image_profile="base",
        preflight=False,
    )
    assert outcome.upload_action == "LIKELY SKIPPED"
    assert "previously matched" in (outcome.remote_path or "")
    assert not any(call.category == "rsync" for call in runner.calls)


def test_rsync_uses_ssh_alias(tmp_path: Path) -> None:
    pipeline = PushPipeline(CommandRunner(), ImageStateStore(ConfigStore(tmp_path)))
    args = pipeline._rsync_args(Path("/tmp/a.tar"), "bonecho", "/scratch/alice/isaaclab-containers/x.tar.partial")
    assert args[0] == "rsync"
    assert "--partial" in args
    assert "--progress" in args
    assert args[-1].startswith("bonecho:")
    assert "-e" in args
