"""Inspect SSH ControlMaster/ControlPersist without rewriting ~/.ssh/config."""

from __future__ import annotations

from dataclasses import dataclass

from vector_sim.exec import CommandRunner
from vector_sim.ssh.session import ResolvedSsh, parse_ssh_g, resolved_ssh_from_g

_MUX_ON = {"yes", "true", "auto", "ask"}

RECOMMENDED_SSH_SNIPPET = """\
Host bonecho
    HostName <login-node>
    User <your-user>
    ControlMaster auto
    ControlPersist 10m
    ControlPath ~/.ssh/cm-%C
"""


@dataclass
class MultiplexStatus:
    alias: str
    hostname: str | None
    user: str | None
    control_master: str | None
    control_persist: str | None
    control_path: str | None
    multiplexing_configured: bool
    master_alive: bool | None
    resolved: ResolvedSsh | None = None

    @property
    def authentication_likely(self) -> bool:
        return bool(self.master_alive)


def _scalar(raw: dict, key: str) -> str | None:
    value = raw.get(key)
    if isinstance(value, list):
        return str(value[-1]) if value else None
    if value is None:
        return None
    return str(value)


def multiplex_from_ssh_g(alias: str, output: str) -> MultiplexStatus:
    resolved = resolved_ssh_from_g(alias, output)
    raw = parse_ssh_g(output)
    master = (_scalar(raw, "controlmaster") or "").lower() or None
    persist = _scalar(raw, "controlpersist")
    path = _scalar(raw, "controlpath")
    configured = bool(master and master in _MUX_ON)
    return MultiplexStatus(
        alias=alias,
        hostname=resolved.hostname,
        user=resolved.user,
        control_master=master,
        control_persist=persist,
        control_path=path,
        multiplexing_configured=configured,
        master_alive=None,
        resolved=resolved,
    )


def check_master(runner: CommandRunner, alias: str) -> bool:
    result = runner.run(
        ["ssh", "-O", "check", alias],
        category="ssh-mux",
        check=False,
        dry_run_skip=False,
    )
    if result.skipped:
        return False
    return result.returncode == 0


def inspect_multiplex(runner: CommandRunner, alias: str, *, check_alive: bool = True) -> MultiplexStatus:
    g = runner.run(
        ["ssh", "-G", alias],
        category="ssh-config",
        check=False,
        dry_run_skip=False,
        suggestion=f"Add Host {alias} to ~/.ssh/config. vector-sim will not rewrite that file.",
    )
    status = multiplex_from_ssh_g(alias, g.stdout if g.returncode == 0 else "")
    if g.returncode != 0:
        status.hostname = status.hostname
    if check_alive and g.returncode == 0:
        status.master_alive = check_master(runner, alias)
    return status
