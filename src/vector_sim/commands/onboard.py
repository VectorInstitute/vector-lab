"""vector-sim onboard — first-time setup orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vector_sim.commands.auth import AuthCommand
from vector_sim.commands.doctor import Doctor, format_doctor_report
from vector_sim.commands.setup import SetupCommand
from vector_sim.config.store import ConfigStore
from vector_sim.exec import CommandRunner
from vector_sim.onboard.isaaclab import find_or_clone
from vector_sim.onboard.prereqs import format_prereq_report, inspect_prereqs
from vector_sim.ssh.multiplex import RECOMMENDED_SSH_SNIPPET


@dataclass
class OnboardStep:
    name: str
    status: str
    detail: str = ""


@dataclass
class OnboardReport:
    steps: list[OnboardStep] = field(default_factory=list)
    ready: bool = False
    actions: list[str] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.steps.append(OnboardStep(name, status, detail))


class OnboardCommand:
    def __init__(self, runner: CommandRunner, store: ConfigStore, *, start_dir: Path | None = None) -> None:
        self.runner = runner
        self.store = store
        self.start_dir = Path(start_dir or Path.cwd())

    def run(
        self,
        name: str,
        *,
        isaaclab: str | Path | None = None,
        ssh_alias: str | None = None,
        dest: str | Path | None = None,
        clone: bool = True,
        authenticate: bool = True,
    ) -> int:
        report = OnboardReport()
        alias = ssh_alias or name

        self.runner.emit(f"== onboard {name} ==")
        prereqs = inspect_prereqs(start=self.start_dir)
        self.runner.emit(format_prereq_report(prereqs))
        prereq_status = "READY" if prereqs.ready else "USER ACTION REQUIRED"
        report.add("prerequisites", prereq_status, prereqs.platform.pretty)
        if not prereqs.ready:
            report.actions.append("Install missing local tools (see commands above). Do not use sudo via vector-sim.")

        repo = find_or_clone(
            self.runner,
            configured=isaaclab,
            start=self.start_dir,
            dest=Path(dest) if dest else None,
            clone=clone,
        )
        report.add("isaaclab", repo.action, repo.detail)
        if repo.warning:
            self.runner.emit(f"[WARN] {repo.warning}")
            report.add("isaaclab-version", "WARN", repo.warning)
        if repo.action in {"MISSING", "WOULD_CLONE"}:
            report.actions.append(repo.detail or "Clone Isaac Lab at the pinned commit.")
        isaaclab_path = str(repo.path) if repo.path and repo.action in {"READY", "CLONED", "WARN"} else isaaclab

        auth = AuthCommand(self.runner, self.store)
        if authenticate:
            outcome = auth.run(alias, interactive=not self.runner.dry_run)
            report.add("ssh", outcome.action, outcome.detail)
            if not outcome.authenticated:
                report.actions.append(outcome.detail)
                if not outcome.multiplex.multiplexing_configured:
                    report.actions.append("Add ControlMaster to ~/.ssh/config yourself (vector-sim will not rewrite it).")
                    self.runner.emit(RECOMMENDED_SSH_SNIPPET.replace("bonecho", alias))
        else:
            report.add("ssh", "SKIPPED", "--no-auth")

        if isaaclab_path and Path(str(isaaclab_path)).exists():
            profile = SetupCommand(self.runner, self.store, start_dir=self.start_dir).run(
                name,
                isaaclab=isaaclab_path,
                ssh_alias=alias,
            )
            report.add("setup", "READY", f"profile {profile.name}")
        else:
            report.add("setup", "SKIPPED", "Isaac Lab path not ready")
            report.actions.append("Re-run onboard after Isaac Lab is cloned.")

        cluster = name
        try:
            results = Doctor(self.runner, self.store, cluster=cluster, start_dir=self.start_dir).run()
            self.runner.emit("")
            self.runner.emit(format_doctor_report(results))
            doctor_ok = all(item.ok for item in results)
            report.add("doctor", "READY" if doctor_ok else "USER ACTION REQUIRED")
            report.ready = doctor_ok and prereqs.ready and not report.actions
        except Exception as exc:
            report.add("doctor", "USER ACTION REQUIRED", str(exc))
            report.ready = False

        self.runner.emit("")
        self.runner.emit("Onboard steps:")
        for step in report.steps:
            extra = f"  {step.detail}" if step.detail else ""
            self.runner.emit(f"  {step.name:<16} {step.status:<22}{extra}")
        if report.actions:
            self.runner.emit("")
            self.runner.emit("Still needs you:")
            for action in report.actions:
                self.runner.emit(f"  - {action}")
        else:
            self.runner.emit("")
            self.runner.emit("Overall: READY")
            self.runner.emit("Next: vector-sim deploy")
        if any(c.args and c.args[0] == "sudo" for c in self.runner.calls):
            raise RuntimeError("onboard must never invoke sudo")
        if any(".ssh/config" in " ".join(c.args) for c in self.runner.calls):
            raise RuntimeError("onboard must never rewrite ~/.ssh/config")
        return 0 if report.ready else 1
