"""vector-sim bootstrap — inspect (and never silently sudo)."""

from __future__ import annotations

from pathlib import Path

from vector_sim.config.store import ConfigStore
from vector_sim.exec import CommandRunner
from vector_sim.onboard.prereqs import format_prereq_report, inspect_prereqs


class BootstrapCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore, *, start_dir: Path | None = None) -> None:
        self.runner = runner
        self.store = store
        self.start_dir = Path(start_dir or Path.cwd())

    def run(self, *, install: bool = False) -> int:
        report = inspect_prereqs(start=self.start_dir)
        self.runner.emit(format_prereq_report(report))
        if install:
            self.runner.emit("")
            self.runner.emit("`--install` does not run sudo.")
            pending = [i for i in report.items if i.install_command and i.status != "INSTALLED"]
            if not pending:
                self.runner.emit("No install commands to print.")
            else:
                self.runner.emit("Run these yourself if you accept the privilege change:")
                for item in pending:
                    self.runner.emit(f"  {item.install_command}")
        if any(c.args and c.args[0] == "sudo" for c in self.runner.calls):
            raise RuntimeError("bootstrap must never invoke sudo")
        return 0 if report.ready else 1
