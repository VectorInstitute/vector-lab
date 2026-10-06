"""Mockable subprocess execution with dry-run and verbose support."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TextIO


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    category: str = "general"
    skipped: bool = False


ExecuteFn = Callable[..., CommandResult]


@dataclass
class RecordedCall:
    args: list[str]
    category: str
    cwd: str | None = None


class CommandRunner:
    """Run local commands. Inject ``execute`` in tests to avoid real subprocesses."""

    def __init__(
        self,
        *,
        dry_run: bool = False,
        verbose: bool = False,
        execute: ExecuteFn | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.dry_run = dry_run
        self.verbose = verbose
        self._execute = execute
        self._interactive: Callable[[list[str]], int] | None = None
        self._stdout = stdout
        self.calls: list[RecordedCall] = []

    def set_interactive(self, fn: Callable[[list[str]], int] | None) -> None:
        """Tests inject a stub so MFA never runs."""
        self._interactive = fn

    def emit(self, message: str) -> None:
        if self._stdout is not None:
            self._stdout.write(message + "\n")
        else:
            print(message)

    def format_args(self, args: Sequence[str]) -> str:
        return shlex.join(args)

    def run(
        self,
        args: Sequence[str],
        *,
        category: str,
        check: bool = True,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        dry_run_skip: bool = True,
        suggestion: str | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        argv = [str(a) for a in args]
        self.calls.append(RecordedCall(args=argv, category=category, cwd=cwd))
        if self.verbose:
            self.emit(f"+ ({category}) {self.format_args(argv)}")

        if self.dry_run and dry_run_skip:
            self.emit(f"[dry-run] ({category}) {self.format_args(argv)}")
            return CommandResult(args=argv, returncode=0, category=category, skipped=True)

        if self._execute is not None:
            result = self._execute(
                argv,
                category=category,
                cwd=cwd,
                env=env,
                timeout=timeout,
            )
        else:
            result = self._run_subprocess(argv, category=category, cwd=cwd, env=env, timeout=timeout)

        if self.verbose and result.stdout:
            self.emit(result.stdout.rstrip())
        if self.verbose and result.stderr:
            self.emit(result.stderr.rstrip())

        if check and result.returncode != 0:
            from vector_sim.errors import CommandError

            raise CommandError(
                f"{category} command failed with exit status {result.returncode}",
                category=category,
                command=argv,
                exit_status=result.returncode,
                stderr=result.stderr,
                stdout=result.stdout,
                suggestion=suggestion,
            )
        return result

    def _run_subprocess(
        self,
        argv: list[str],
        *,
        category: str,
        cwd: str | None,
        env: Mapping[str, str] | None,
        timeout: float | None,
    ) -> CommandResult:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env={**os.environ, **dict(env)} if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            args=argv,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            category=category,
        )

    def run_interactive(self, args: Sequence[str], *, category: str = "ssh-auth") -> int:
        """Run a command with inherited stdio (MFA). Never captures passwords."""
        argv = [str(a) for a in args]
        self.calls.append(RecordedCall(args=argv, category=category))
        if self.dry_run:
            self.emit(f"[dry-run] ({category}) {self.format_args(argv)}")
            return 0
        if self._interactive is not None:
            return int(self._interactive(argv))
        completed = subprocess.run(argv, check=False)
        return int(completed.returncode)
