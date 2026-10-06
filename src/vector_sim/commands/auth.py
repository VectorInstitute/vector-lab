"""vector-sim auth — interactive MFA without storing secrets."""

from __future__ import annotations

from dataclasses import dataclass

from vector_sim.config.store import ConfigStore
from vector_sim.exec import CommandRunner
from vector_sim.ssh.multiplex import RECOMMENDED_SSH_SNIPPET, MultiplexStatus, inspect_multiplex
from vector_sim.ssh.session import SshSession


@dataclass
class AuthOutcome:
    alias: str
    multiplex: MultiplexStatus
    authenticated: bool
    action: str  # READY | AUTHENTICATED | REQUIRED | SKIPPED
    detail: str = ""


def batch_whoami(runner: CommandRunner, alias: str, *, login_shell: bool = True) -> bool:
    session = SshSession(alias, runner, login_shell=login_shell, batch_mode=True)
    result = session.exec("whoami", category="ssh-auth", check=False, dry_run_skip=True)
    if result.skipped:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


class AuthCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore) -> None:
        self.runner = runner
        self.store = store

    def inspect(self, alias: str) -> MultiplexStatus:
        return inspect_multiplex(self.runner, alias, check_alive=True)

    def run(self, alias: str, *, interactive: bool = True, login_shell: bool = True) -> AuthOutcome:
        mux = self.inspect(alias)
        self._emit_mux(mux)
        if mux.master_alive and batch_whoami(self.runner, alias, login_shell=login_shell):
            return AuthOutcome(alias, mux, True, "READY", "ControlMaster is alive; BatchMode works.")

        if self.runner.dry_run or not interactive:
            return AuthOutcome(
                alias,
                mux,
                False,
                "REQUIRED",
                "SSH authentication required. Run: vector-sim auth " + alias,
            )

        self.runner.emit("SSH authentication required.")
        self.runner.emit("Opening interactive SSH session (complete MFA in the terminal).")
        self.runner.emit("vector-sim does not store passwords or MFA tokens.")
        code = self.runner.run_interactive(["ssh", alias], category="ssh-auth")
        if code != 0:
            return AuthOutcome(alias, mux, False, "REQUIRED", f"interactive ssh exited {code}")

        mux = inspect_multiplex(self.runner, alias, check_alive=True)
        ok = batch_whoami(self.runner, alias, login_shell=login_shell)
        if ok:
            return AuthOutcome(alias, mux, True, "AUTHENTICATED", "non-interactive reuse succeeded")
        detail = "Interactive SSH finished but BatchMode still fails. Configure ControlMaster (see snippet)."
        if not mux.multiplexing_configured:
            self.runner.emit("SSH multiplexing is not configured. Add this to ~/.ssh/config yourself:")
            self.runner.emit(RECOMMENDED_SSH_SNIPPET.replace("bonecho", alias))
            self.runner.emit("vector-sim will not rewrite ~/.ssh/config.")
        return AuthOutcome(alias, mux, False, "REQUIRED", detail)

    def _emit_mux(self, mux: MultiplexStatus) -> None:
        self.runner.emit(f"SSH alias            {mux.alias}")
        self.runner.emit(f"Resolved host        {mux.hostname or '(unresolved)'}")
        self.runner.emit(f"SSH user             {mux.user or '(from config / default)'}")
        self.runner.emit(f"ControlMaster        {mux.control_master or 'unset'}")
        self.runner.emit(f"ControlPersist       {mux.control_persist or 'unset'}")
        self.runner.emit(f"ControlPath          {mux.control_path or 'unset'}")
        self.runner.emit(f"Multiplexing         {'configured' if mux.multiplexing_configured else 'not configured'}")
        alive = "yes" if mux.master_alive else "no" if mux.master_alive is False else "unknown"
        self.runner.emit(f"Master connection    {alive}")
