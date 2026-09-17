"""Upload a converted container artifact with checksum-gated rsync."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

from vector_lab.config.models import ClusterProfile
from vector_lab.errors import ConfigError, VectorLabError
from vector_lab.exec import CommandRunner
from vector_lab.images.build import BuildPipeline
from vector_lab.images.conversion import ConversionSpec, LocalApptainerConversion
from vector_lab.images.naming import (
    artifact_filename,
    docker_image_ref,
    file_sha256,
    parse_sha256sum_output,
    remote_tar_name,
    sandbox_dirname,
)
from vector_lab.images.preflight import (
    apptainer_preflight,
    disk_preflight,
    docker_preflight,
    raise_if_errors,
    remote_push_preflight,
)
from vector_lab.images.state import ImageArtifactState, ImageStateStore, stamp
from vector_lab.ssh.session import SshSession


@dataclass
class PushOutcome:
    image_profile: str
    docker_image: str
    build_action: str
    conversion_action: str
    upload_action: str
    build_input_fingerprint: str | None = None
    built_image_digest: str | None = None
    conversion_fingerprint: str | None = None
    local_artifact: str | None = None
    local_sha256: str | None = None
    remote_path: str | None = None
    notes: list[str] = field(default_factory=list)


class PushPipeline:
    def __init__(
        self,
        runner: CommandRunner,
        state: ImageStateStore,
        *,
        backend: LocalApptainerConversion | None = None,
    ) -> None:
        self.runner = runner
        self.state = state
        self.backend = backend

    def run(
        self,
        *,
        profile: ClusterProfile,
        repo: Path,
        image_profile: str,
        force: bool = False,
        force_convert: bool = False,
        force_upload: bool = False,
        group_name: str | None = None,
        preflight: bool = True,
    ) -> PushOutcome:
        self.state.store.ensure_local()
        if preflight:
            raise_if_errors(
                docker_preflight(self.runner)
                + disk_preflight(self.state.store.artifacts_dir)
                + apptainer_preflight(self.runner, group_name=group_name)
            )

        built = BuildPipeline(self.runner, self.state).run(
            repo=repo,
            image_profile=image_profile,
            force=force,
            preflight=False,
        )
        record = self.state.get(image_profile) or ImageArtifactState(
            image_profile=image_profile,
            docker_image=built.docker_image,
        )
        record.build_input_fingerprint = built.build_input_fingerprint
        record.built_image_digest = built.built_image_digest or record.built_image_digest
        record.docker_image = built.docker_image

        backend = self.backend or LocalApptainerConversion(
            self.runner,
            work_root=self.state.store.work_dir / image_profile,
        )

        digest = record.built_image_digest or "unknown"
        # Resolve before conversion: tar runs with cwd=work_root, so relative
        # project paths would otherwise open under the wrong directory.
        output_tar = (self.state.store.artifacts_dir / artifact_filename(image_profile, digest)).resolve()
        spec = ConversionSpec(
            docker_image=docker_image_ref(image_profile),
            output_tar=output_tar,
            sandbox_name=sandbox_dirname(image_profile),
            image_digest=digest,
        )
        conv_fp = backend.compute_fingerprint(spec)
        artifact_ok = _artifact_exists(record, spec.output_tar)
        if self.runner.dry_run and record.conversion_fingerprint == conv_fp and record.local_artifact_sha256:
            artifact_ok = True
        need_convert = force or force_convert or record.conversion_fingerprint != conv_fp or not artifact_ok
        convert_action = "SKIPPED"
        if need_convert:
            convert_action = "REQUIRED"
            result = backend.convert(spec)
            if not result.skipped and spec.output_tar.is_file():
                convert_action = "CONVERTED"
                record.local_artifact = str(result.output_tar)
                record.local_artifact_size = result.output_tar.stat().st_size
                record.local_artifact_sha256 = file_sha256(result.output_tar)
                record.conversion_fingerprint = result.conversion_fingerprint
                record.conversion_timestamp = stamp()
            else:
                record.conversion_fingerprint = conv_fp
                record.local_artifact = str(spec.output_tar)

        local_path = Path(record.local_artifact) if record.local_artifact else spec.output_tar
        if local_path.is_file():
            record.local_artifact = str(local_path)
            record.local_artifact_size = local_path.stat().st_size
            record.local_artifact_sha256 = record.local_artifact_sha256 or file_sha256(local_path)

        remote_dir = profile.paths.containers
        if not remote_dir:
            raise ConfigError(
                "profile is missing paths.containers",
                suggestion="Run: vector-lab onboard bonecho, then vector-lab deploy",
            )
        remote_final = f"{remote_dir.rstrip('/')}/{remote_tar_name(image_profile)}"
        remote_partial = f"{remote_final}.partial"

        session = SshSession(
            profile.ssh_alias,
            self.runner,
            login_shell=profile.scheduler.remote_shell == "login",
        )
        if preflight:
            raise_if_errors(remote_push_preflight(session, containers_dir=remote_dir))

        local_sha = record.local_artifact_sha256
        notes: list[str] = []
        upload_action = "REQUIRED"
        outcome_remote = remote_final
        probed_remote_sha, probe_skipped = self._remote_checksum_with_status(session, remote_final)
        remote_sha = None if force_upload else probed_remote_sha

        if self.runner.dry_run and (probe_skipped or remote_sha is None):
            # Never claim a definitive remote skip from cached local state alone.
            if record.remote_sha256 and local_sha and record.remote_sha256 == local_sha and not force_upload:
                upload_action = "LIKELY SKIPPED"
                outcome_remote = "previously matched / would verify"
                notes.append("dry-run did not probe the live remote checksum")
            else:
                upload_action = "REQUIRED"
                notes.append(self._rsync_preview(local_path, profile.ssh_alias, remote_partial))
        elif remote_sha and local_sha and remote_sha == local_sha and not force_upload:
            upload_action = "SKIPPED"
            outcome_remote = "checksum match"
            record.remote_path = remote_final
            record.remote_sha256 = remote_sha
        elif self.runner.dry_run:
            upload_action = "REQUIRED"
            notes.append(self._rsync_preview(local_path, profile.ssh_alias, remote_partial))
        else:
            if not local_path.is_file():
                raise VectorLabError(
                    f"local artifact missing: {local_path}",
                    category="push",
                    suggestion="Conversion must produce a tar before upload.",
                )
            self._upload_atomic(session, profile, local_path, remote_partial, remote_final, local_sha or "")
            upload_action = "UPLOADED"
            outcome_remote = remote_final
            record.remote_path = remote_final
            record.remote_sha256 = local_sha
            record.pushed_at = stamp()

        if not self.runner.dry_run:
            self.state.upsert(record)
            profile.fingerprints.build_input_fingerprint = record.build_input_fingerprint
            profile.fingerprints.built_image_digest = record.built_image_digest
            profile.fingerprints.conversion_fingerprint = record.conversion_fingerprint

        return PushOutcome(
            image_profile=image_profile,
            docker_image=built.docker_image,
            build_action=built.action,
            conversion_action=convert_action,
            upload_action=upload_action,
            build_input_fingerprint=record.build_input_fingerprint,
            built_image_digest=record.built_image_digest,
            conversion_fingerprint=record.conversion_fingerprint or conv_fp,
            local_artifact=record.local_artifact,
            local_sha256=record.local_artifact_sha256,
            remote_path=outcome_remote,
            notes=notes,
        )

    def preview(
        self,
        *,
        profile: ClusterProfile,
        repo: Path,
        image_profile: str,
        force: bool = False,
        force_convert: bool = False,
        force_upload: bool = False,
    ) -> PushOutcome:
        """Cheap plan: fingerprints and local state only. No convert/upload/rebuild."""
        from vector_lab.images.build import BuildPipeline
        from vector_lab.images.conversion import ConversionSpec, LocalApptainerConversion
        from vector_lab.images.fingerprint import fingerprint_repo_build_inputs
        from vector_lab.images.naming import artifact_filename, docker_image_ref, sandbox_dirname

        builder = BuildPipeline(self.runner, self.state)
        fingerprint_needed = fingerprint_repo_build_inputs(repo, image_profile)
        action, reason, previous = builder.plan(repo=repo, image_profile=image_profile, force=force)
        digest = previous.built_image_digest if previous else None
        backend = self.backend or LocalApptainerConversion(
            self.runner,
            work_root=self.state.store.work_dir / image_profile,
        )
        output_tar = (self.state.store.artifacts_dir / artifact_filename(image_profile, digest or "unknown")).resolve()
        spec = ConversionSpec(
            docker_image=docker_image_ref(image_profile),
            output_tar=output_tar,
            sandbox_name=sandbox_dirname(image_profile),
            image_digest=digest or "unknown",
        )
        conv_fp = backend.compute_fingerprint(spec)
        record = previous
        artifact_ok = _artifact_exists(record, spec.output_tar) if record else output_tar.is_file()
        need_convert = (
            force
            or force_convert
            or record is None
            or record.conversion_fingerprint != conv_fp
            or not artifact_ok
        )
        local_sha = record.local_artifact_sha256 if record else None
        if force_upload:
            upload = "REQUIRED"
        elif record and record.remote_sha256 and local_sha and record.remote_sha256 == local_sha:
            upload = "LIKELY SKIPPED"
        else:
            upload = "REQUIRED"
        return PushOutcome(
            image_profile=image_profile,
            docker_image=docker_image_ref(image_profile),
            build_action=action,
            conversion_action="REQUIRED" if need_convert else "SKIPPED",
            upload_action=upload,
            build_input_fingerprint=fingerprint_needed,
            built_image_digest=digest,
            conversion_fingerprint=(record.conversion_fingerprint if record else conv_fp),
            local_artifact=record.local_artifact if record else None,
            local_sha256=local_sha,
            remote_path=record.remote_path if record else None,
            notes=[reason] if reason else [],
        )

    def _remote_checksum(self, session: SshSession, remote_path: str) -> str | None:
        sha, _skipped = self._remote_checksum_with_status(session, remote_path)
        return sha

    def _remote_checksum_with_status(
        self, session: SshSession, remote_path: str
    ) -> tuple[str | None, bool]:
        result = session.exec(
            f"if [ -f {shlex.quote(remote_path)} ]; then sha256sum {shlex.quote(remote_path)}; fi",
            category="ssh-checksum",
            check=False,
        )
        if result.skipped:
            return None, True
        if result.returncode != 0:
            return None, False
        return parse_sha256sum_output(result.stdout), False

    def _rsync_preview(self, local: Path, alias: str, remote_partial: str) -> str:
        return shlex.join(self._rsync_args(local, alias, remote_partial))

    def _rsync_args(self, local: Path, alias: str, remote_partial: str) -> list[str]:
        return [
            "rsync",
            "-avh",
            "--progress",
            "--partial",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=15",
            str(local),
            f"{alias}:{remote_partial}",
        ]

    def _upload_atomic(
        self,
        session: SshSession,
        profile: ClusterProfile,
        local: Path,
        remote_partial: str,
        remote_final: str,
        expected_sha: str,
    ) -> None:
        self.runner.run(
            self._rsync_args(local, profile.ssh_alias, remote_partial),
            category="rsync",
            suggestion="Retry push; rsync --partial can resume an interrupted transfer.",
        )
        checksum = session.exec(
            f"sha256sum {shlex.quote(remote_partial)}",
            category="ssh-checksum",
        )
        actual = parse_sha256sum_output(checksum.stdout)
        if actual != expected_sha:
            session.exec(
                f"rm -f {shlex.quote(remote_partial)}",
                category="remote-fs",
                check=False,
            )
            raise VectorLabError(
                f"remote checksum mismatch after upload (got {actual}, expected {expected_sha})",
                category="push",
                suggestion="Retry vector-lab deploy. An incomplete remote partial can be resumed.",
            )
        session.exec(
            f"mv -f {shlex.quote(remote_partial)} {shlex.quote(remote_final)}",
            category="remote-fs",
        )


def _artifact_exists(record: ImageArtifactState, fallback: Path) -> bool:
    if record.local_artifact and Path(record.local_artifact).is_file():
        return True
    return fallback.is_file()


def format_pipeline_report(outcome: PushOutcome) -> str:
    inputs = "unchanged" if outcome.build_action == "SKIPPED" else "changed or missing image"
    conv = {
        "SKIPPED": "cached",
        "REQUIRED": "REQUIRED",
        "CONVERTED": "CONVERTED",
    }.get(outcome.conversion_action, outcome.conversion_action)
    remote = outcome.remote_path or "(none)"
    if outcome.upload_action == "SKIPPED":
        remote = "checksum match"
    elif outcome.upload_action == "LIKELY SKIPPED":
        remote = outcome.remote_path or "previously matched / would verify"
    lines = [
        f"Docker inputs       {inputs}",
        f"Docker image        {outcome.docker_image}",
        f"Image digest        {outcome.built_image_digest or 'unknown'}",
        f"Docker build        {outcome.build_action}",
        "",
        f"Conversion          {conv}",
        f"Artifact            {outcome.local_artifact or '(none)'}",
        f"SHA256              {outcome.local_sha256 or '(none)'}",
        "",
        f"Remote artifact     {remote}",
        f"Upload              {outcome.upload_action}",
    ]
    for note in outcome.notes:
        lines.append(f"rsync               {note}")
    return "\n".join(lines)
