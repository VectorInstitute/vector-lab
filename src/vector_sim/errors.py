"""Error types with command context for debugging cluster operations."""

from __future__ import annotations

from typing import Sequence


class VectorSimError(Exception):
    """Base error for the vector-sim CLI."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "general",
        command: Sequence[str] | None = None,
        exit_status: int | None = None,
        stderr: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.command = list(command) if command is not None else None
        self.exit_status = exit_status
        self.stderr = stderr
        self.suggestion = suggestion

    def format_report(self) -> str:
        lines = [f"error ({self.category}): {self}"]
        if self.command:
            lines.append(f"  command: {self.command}")
        if self.exit_status is not None:
            lines.append(f"  exit status: {self.exit_status}")
        if self.stderr:
            lines.append(f"  stderr: {self.stderr.strip()}")
        if self.suggestion:
            lines.append(f"  next: {self.suggestion}")
        return "\n".join(lines)


class CommandError(VectorSimError):
    """A subprocess command failed."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        command: Sequence[str],
        exit_status: int,
        stderr: str | None = None,
        stdout: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            message,
            category=category,
            command=command,
            exit_status=exit_status,
            stderr=stderr,
            suggestion=suggestion,
        )
        self.stdout = stdout


class ConfigError(VectorSimError):
    """Profile or configuration is invalid or missing."""

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        super().__init__(message, category="config", suggestion=suggestion)


class DiscoveryError(VectorSimError):
    """Failed to discover Isaac Lab, SSH, or cluster details."""

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        super().__init__(message, category="discovery", suggestion=suggestion)


class NotImplementedPhaseError(VectorSimError):
    """Command exists in the CLI but is not implemented in this phase."""

    def __init__(self, command: str, phase: str) -> None:
        super().__init__(
            f"'{command}' is not implemented yet ({phase}).",
            category="cli",
            suggestion=f"Wait for {phase} or use --help to see currently supported commands.",
        )
