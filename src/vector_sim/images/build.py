"""Isaac Lab Docker image build with fingerprint cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vector_sim.errors import ConfigError, DiscoveryError, VectorSimError
from vector_sim.exec import CommandRunner
from vector_sim.images.fingerprint import fingerprint_repo_build_inputs
from vector_sim.images.naming import docker_image_ref, parse_docker_inspect_digest
from vector_sim.images.preflight import docker_preflight, raise_if_errors
from vector_sim.images.state import ImageArtifactState, ImageStateStore, stamp
from vector_sim.local.docker_util import which_docker
from vector_sim.local.repo import discover_isaac_lab


@dataclass
class BuildOutcome:
    image_profile: str
    docker_image: str
    build_input_fingerprint: str
    built_image_digest: str | None
    action: str  # SKIPPED, REQUIRED, BUILT
    reason: str


class BuildPipeline:
    def __init__(self, runner: CommandRunner, state: ImageStateStore) -> None:
        self.runner = runner
        self.state = state

    def plan(
        self,
        *,
        repo: Path,
        image_profile: str,
        force: bool = False,
    ) -> tuple[str, str, ImageArtifactState | None]:
        fingerprint = fingerprint_repo_build_inputs(repo, image_profile)
        current = self.state.get(image_profile)
        image = docker_image_ref(image_profile)
        exists = self._image_exists(image)
        if self.runner.dry_run and current and current.built_image_digest:
            exists = True
        if force:
            return "REQUIRED", "forced rebuild", current
        if (
            current
            and current.build_input_fingerprint == fingerprint
            and exists
        ):
            return "SKIPPED", "Docker inputs unchanged and image exists", current
        if not exists:
            return "REQUIRED", "Docker image is missing", current
        if current is None or current.build_input_fingerprint != fingerprint:
            return "REQUIRED", "Docker inputs changed or no cached fingerprint", current
        return "REQUIRED", "rebuild needed", current

    def run(
        self,
        *,
        repo: Path,
        image_profile: str,
        force: bool = False,
        preflight: bool = True,
    ) -> BuildOutcome:
        if preflight:
            raise_if_errors(docker_preflight(self.runner))
        docker_image = docker_image_ref(image_profile)
        fingerprint = fingerprint_repo_build_inputs(repo, image_profile)
        action, reason, previous = self.plan(repo=repo, image_profile=image_profile, force=force)
        digest = previous.built_image_digest if previous else None
        if action == "SKIPPED":
            digest = digest or self._inspect_digest(docker_image)
            record = self._record(image_profile, docker_image, fingerprint, digest, previous)
            if not self.runner.dry_run:
                self.state.upsert(record)
            return BuildOutcome(
                image_profile=image_profile,
                docker_image=docker_image,
                build_input_fingerprint=fingerprint,
                built_image_digest=digest,
                action="SKIPPED",
                reason=reason,
            )

        if self.runner.dry_run:
            return BuildOutcome(
                image_profile=image_profile,
                docker_image=docker_image,
                build_input_fingerprint=fingerprint,
                built_image_digest=digest,
                action="REQUIRED",
                reason=reason,
            )

        self._build(repo, image_profile)
        if not self._image_exists(docker_image):
            raise VectorSimError(
                f"Docker build finished but image {docker_image} was not found",
                category="docker-build",
                suggestion="Re-run with --verbose and inspect docker compose output.",
            )
        digest = self._inspect_digest(docker_image)
        record = self._record(image_profile, docker_image, fingerprint, digest, previous)
        self.state.upsert(record)
        return BuildOutcome(
            image_profile=image_profile,
            docker_image=docker_image,
            build_input_fingerprint=fingerprint,
            built_image_digest=digest,
            action="BUILT",
            reason=reason,
        )

    def _docker(self) -> str:
        return which_docker() or "docker"

    def _build(self, repo: Path, image_profile: str) -> None:
        container_py = Path(repo) / "docker" / "container.py"
        if not container_py.is_file():
            raise DiscoveryError(f"missing {container_py}")
        self.runner.run(
            ["python3", str(container_py), "build", image_profile],
            category="docker-build",
            cwd=str(Path(repo) / "docker"),
            suggestion="Fix Docker access, then retry. Do not use sudo or newgrp docker.",
        )

    def _image_exists(self, image: str) -> bool:
        result = self.runner.run(
            [self._docker(), "image", "inspect", image],
            category="docker-inspect",
            check=False,
            dry_run_skip=True,
        )
        if result.skipped:
            return False
        return result.returncode == 0

    def _inspect_digest(self, image: str) -> str | None:
        result = self.runner.run(
            [
                self._docker(),
                "image",
                "inspect",
                "--format",
                "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}",
                image,
            ],
            category="docker-inspect",
            check=False,
            dry_run_skip=True,
        )
        if result.skipped or result.returncode != 0 or not result.stdout.strip():
            return None
        return parse_docker_inspect_digest(result.stdout)

    def _record(
        self,
        image_profile: str,
        docker_image: str,
        fingerprint: str,
        digest: str | None,
        previous: ImageArtifactState | None,
    ) -> ImageArtifactState:
        record = previous or ImageArtifactState(image_profile=image_profile, docker_image=docker_image)
        record.image_profile = image_profile
        record.docker_image = docker_image
        record.build_input_fingerprint = fingerprint
        if digest:
            record.built_image_digest = digest
        if previous is None:
            record.conversion_timestamp = record.conversion_timestamp or stamp()
        return record


def resolve_isaaclab(profile_isaaclab: str | None, start: Path | None = None) -> Path:
    repo = discover_isaac_lab(configured=profile_isaaclab, start=start)
    return repo.root


def require_cluster_profile(store, cluster: str | None):
    from vector_sim.config.store import ConfigStore

    cfg: ConfigStore = store
    name = cluster or cfg.get_active()
    if not name:
        raise ConfigError(
            "no cluster profile",
            suggestion="Run: vector-sim onboard bonecho",
        )
    return cfg.load_profile(name)
