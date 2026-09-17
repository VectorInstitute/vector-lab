"""Container conversion backends. Push depends on this interface, not a specific tool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import shutil

from vector_lab.errors import VectorLabError
from vector_lab.exec import CommandRunner
from vector_lab.images.fingerprint import conversion_fingerprint
from vector_lab.images.naming import sandbox_dirname, safe_rmtree

CONVERSION_OPTIONS = "--sandbox --fakeroot tar"


@dataclass
class ConversionSpec:
    docker_image: str
    output_tar: Path
    sandbox_name: str
    image_digest: str = ""
    apptainer_version: str = ""


@dataclass
class ConversionResult:
    output_tar: Path
    conversion_fingerprint: str
    skipped: bool = False


class ConversionBackend(ABC):
    """Convert a Docker image into a cluster-ready Apptainer/Singularity archive."""

    name: str

    @abstractmethod
    def convert(self, spec: ConversionSpec) -> ConversionResult:
        raise NotImplementedError

    @abstractmethod
    def compute_fingerprint(self, spec: ConversionSpec) -> str:
        raise NotImplementedError


class LocalApptainerConversion(ConversionBackend):
    """v1 backend: convert locally with Apptainer, matching Isaac Lab's sandbox+tar flow."""

    name = "local-apptainer"

    def __init__(
        self,
        runner: CommandRunner,
        *,
        work_root: Path,
        apptainer_bin: str | None = None,
    ) -> None:
        self.runner = runner
        self.work_root = Path(work_root)
        self.apptainer_bin = apptainer_bin or shutil.which("apptainer") or "apptainer"
        self._version_cache: str | None = None

    def compute_fingerprint(self, spec: ConversionSpec) -> str:
        version = spec.apptainer_version or self.version()
        return conversion_fingerprint(
            built_image_digest=spec.image_digest,
            backend=self.name,
            version=version,
            options=CONVERSION_OPTIONS,
        )

    def version(self) -> str:
        if self._version_cache:
            return self._version_cache
        result = self.runner.run(
            [self.apptainer_bin, "--version"],
            category="apptainer-preflight",
            check=False,
            # Always probe version so dry-run fingerprints match live conversion state.
            dry_run_skip=False,
        )
        if result.skipped or result.returncode != 0:
            self._version_cache = "unknown"
        else:
            self._version_cache = result.stdout.strip() or "unknown"
        return self._version_cache

    def convert(self, spec: ConversionSpec) -> ConversionResult:
        fingerprint = self.compute_fingerprint(spec)
        work_root = self.work_root.expanduser().resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        self.work_root = work_root
        sandbox = spec.sandbox_name or sandbox_dirname("base")
        sandbox_path = work_root / sandbox
        if sandbox_path.exists() and not self.runner.dry_run:
            safe_rmtree(sandbox_path, allowed_root=work_root)

        build = self.runner.run(
            [
                self.apptainer_bin,
                "build",
                "--sandbox",
                "--fakeroot",
                sandbox,
                f"docker-daemon://{spec.docker_image}",
            ],
            category="apptainer-build",
            cwd=str(work_root),
            env={"APPTAINER_NOHTTPS": "1"},
            suggestion="If fakeroot fails, check that docker is only a supplementary group.",
        )
        if build.skipped:
            return ConversionResult(
                output_tar=Path(spec.output_tar).expanduser().resolve(),
                conversion_fingerprint=fingerprint,
                skipped=True,
            )
        return self.package_sandbox(spec, cleanup_sandbox=True)

    def package_sandbox(self, spec: ConversionSpec, *, cleanup_sandbox: bool = False) -> ConversionResult:
        """Tar an existing sandbox. Resolves output_tar absolutely for cwd=work_root.

        Does not delete the sandbox when packaging fails. When ``cleanup_sandbox`` is
        true and packaging succeeds, the sandbox directory is removed.
        """
        fingerprint = self.compute_fingerprint(spec)
        work_root = self.work_root.expanduser().resolve()
        self.work_root = work_root
        output_tar = Path(spec.output_tar).expanduser().resolve()
        spec.output_tar = output_tar
        output_tar.parent.mkdir(parents=True, exist_ok=True)

        sandbox = spec.sandbox_name or sandbox_dirname("base")
        sandbox_path = work_root / sandbox
        if not self.runner.dry_run and not sandbox_path.is_dir():
            raise VectorLabError(
                f"sandbox missing for packaging: {sandbox_path}",
                category="tar",
                suggestion="Run Apptainer conversion before packaging, or restore the sandbox directory.",
            )

        tar = self.runner.run(
            ["tar", "-cf", str(output_tar), sandbox],
            category="tar",
            cwd=str(work_root),
        )
        if tar.skipped:
            return ConversionResult(
                output_tar=output_tar,
                conversion_fingerprint=fingerprint,
                skipped=True,
            )
        if not output_tar.is_file():
            raise VectorLabError(
                f"conversion did not produce {output_tar}",
                category="tar",
            )
        if cleanup_sandbox and sandbox_path.exists():
            safe_rmtree(sandbox_path, allowed_root=work_root)
        return ConversionResult(output_tar=output_tar, conversion_fingerprint=fingerprint)


class RemoteConversion(ConversionBackend):
    """Future backend: convert on the cluster. Must not require push() API changes."""

    name = "remote"

    def convert(self, spec: ConversionSpec) -> ConversionResult:
        raise NotImplementedError("remote conversion is not implemented")

    def compute_fingerprint(self, spec: ConversionSpec) -> str:
        raise NotImplementedError("remote conversion is not implemented")
