"""SSH config resolution and remote execution, including login-shell wrapping."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from vector_lab.exec import CommandResult, CommandRunner


def wrap_login_shell(command: str) -> str:
    """Wrap a remote command so SLURM/module env from login shells is available.

    Produces: bash -l -c <shlex-quoted-command>
    """
    return "bash -l -c " + shlex.quote(command)


def parse_ssh_g(output: str) -> dict[str, str | list[str]]:
    """Parse ``ssh -G <alias>`` into a dict. Repeated keys (identityfile) become lists."""
    parsed: dict[str, str | list[str]] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or " " not in line:
            continue
        key, value = line.split(None, 1)
        key = key.lower()
        if key in parsed:
            existing = parsed[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                parsed[key] = [existing, value]
        else:
            parsed[key] = value
    return parsed


@dataclass
class ResolvedSsh:
    alias: str
    hostname: str | None
    user: str | None
    port: str | None
    identity_files: list[str] = field(default_factory=list)
    raw: dict[str, str | list[str]] = field(default_factory=dict)

    @property
    def resolved_host(self) -> str | None:
        return self.hostname


def resolved_ssh_from_g(alias: str, output: str) -> ResolvedSsh:
    raw = parse_ssh_g(output)
    identity = raw.get("identityfile", [])
    if isinstance(identity, str):
        identity_files = [identity]
    else:
        identity_files = list(identity)
    user = raw.get("user")
    host = raw.get("hostname")
    port = raw.get("port")
    return ResolvedSsh(
        alias=alias,
        hostname=str(host) if isinstance(host, str) else None,
        user=str(user) if isinstance(user, str) else None,
        port=str(port) if isinstance(port, str) else None,
        identity_files=identity_files,
        raw=raw,
    )


class SshSession:
    """Run commands on a remote host via the SSH alias. Authentication stays in SSH config."""

    def __init__(
        self,
        alias: str,
        runner: CommandRunner,
        *,
        login_shell: bool = False,
        batch_mode: bool = True,
        connect_timeout: int = 15,
    ) -> None:
        self.alias = alias
        self.runner = runner
        self.login_shell = login_shell
        self.batch_mode = batch_mode
        self.connect_timeout = connect_timeout

    def ssh_prefix(self) -> list[str]:
        args = ["ssh"]
        if self.batch_mode:
            args.extend(["-o", "BatchMode=yes"])
        args.extend(["-o", f"ConnectTimeout={self.connect_timeout}"])
        args.append(self.alias)
        return args

    def resolve_config(self) -> ResolvedSsh:
        result = self.runner.run(
            ["ssh", "-G", self.alias],
            category="ssh-config",
            dry_run_skip=False,
            suggestion=f"Add a Host {self.alias} entry to ~/.ssh/config",
        )
        if result.skipped:
            return ResolvedSsh(alias=self.alias, hostname=None, user=None, port=None)
        return resolved_ssh_from_g(self.alias, result.stdout)

    def exec(
        self,
        command: str,
        *,
        category: str = "ssh",
        login_shell: bool | None = None,
        check: bool = True,
        suggestion: str | None = None,
        dry_run_skip: bool = True,
    ) -> CommandResult:
        use_login = self.login_shell if login_shell is None else login_shell
        remote = wrap_login_shell(command) if use_login else command
        args = self.ssh_prefix() + [remote]
        return self.runner.run(
            args,
            category=category,
            check=check,
            suggestion=suggestion,
            dry_run_skip=dry_run_skip,
        )
