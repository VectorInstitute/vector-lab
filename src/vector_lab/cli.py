"""vector-lab command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vector_lab import __version__
from vector_lab.commands.auth import AuthCommand
from vector_lab.commands.bootstrap import BootstrapCommand
from vector_lab.commands.build import BuildCommand
from vector_lab.commands.deploy import DeployCommand
from vector_lab.commands.doctor import Doctor, format_doctor_report
from vector_lab.commands.init import InitCommand
from vector_lab.commands.onboard import OnboardCommand
from vector_lab.commands.ops import CancelCommand, LogsCommand, StatusCommand
from vector_lab.commands.push import PushCommand
from vector_lab.commands.run import RunCommand
from vector_lab.commands.setup import SetupCommand
from vector_lab.commands.smoke import SmokeTestCommand
from vector_lab.commands.videos import PullVideoCommand, ShellCommand, VideosCommand
from vector_lab.config.store import ConfigStore
from vector_lab.errors import VectorLabError
from vector_lab.exec import CommandRunner
from vector_lab.images.naming import image_stem


_GLOBAL_FLAGS_WITH_VALUE = {"--cluster", "--config-dir"}
_GLOBAL_FLAGS_BOOL = {"--dry-run", "--verbose"}


def hoist_global_flags(argv: list[str]) -> list[str]:
    """Allow global flags before or after the subcommand."""
    globals_: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        name = arg.split("=", 1)[0]
        if name in _GLOBAL_FLAGS_BOOL:
            globals_.append(arg)
            i += 1
            continue
        if name in _GLOBAL_FLAGS_WITH_VALUE:
            if "=" in arg:
                globals_.append(arg)
                i += 1
            else:
                globals_.append(arg)
                if i + 1 < len(argv):
                    globals_.append(argv[i + 1])
                    i += 2
                else:
                    i += 1
            continue
        rest.append(arg)
        i += 1
    return globals_ + rest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vector-lab",
        description="Deploy Isaac Lab workloads to HPC clusters over SSH, SLURM, and Apptainer.",
    )
    parser.add_argument("--version", action="version", version=f"vector-lab {__version__}")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without expensive or remote mutations.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show subprocess commands and output.")
    parser.add_argument("--cluster", help="Cluster profile name (default: active profile).")
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="Directory for profiles and state (default: ./.vector-lab).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Inspect local/SSH/cluster details and write a profile.")
    init.add_argument("name", help="Cluster profile name, e.g. bonecho")
    init.add_argument("--ssh-alias", help="SSH config Host alias (default: profile name).")
    init.add_argument("--isaaclab", help="Path to an Isaac Lab checkout.")

    setup = sub.add_parser(
        "setup",
        help="Discover the cluster and generate the reproducible runtime wrappers.",
    )
    setup.add_argument("name", nargs="?", help="Cluster profile name (default: active profile).")
    setup.add_argument("--ssh-alias", help="SSH config Host alias (default: profile name).")
    setup.add_argument("--isaaclab", help="Path to an Isaac Lab checkout.")

    sub.add_parser("doctor", help="Run local, SSH, cluster, and container checks.")
    bootstrap = sub.add_parser("bootstrap", help="Check local dependencies (never runs sudo).")
    bootstrap.add_argument(
        "--install",
        action="store_true",
        help="Print exact install commands; does not execute sudo.",
    )
    onboard = sub.add_parser("onboard", help="First-time setup: tools, Isaac Lab, SSH, profile, doctor.")
    onboard.add_argument("name", help="Cluster profile name, e.g. bonecho")
    onboard.add_argument("--ssh-alias", help="SSH config Host alias (default: profile name).")
    onboard.add_argument("--isaaclab", help="Existing Isaac Lab checkout.")
    onboard.add_argument("--dest", help="Clone destination if Isaac Lab is missing.")
    onboard.add_argument(
        "--no-clone",
        action="store_true",
        help="Do not clone Isaac Lab; print the pinned git command instead.",
    )
    onboard.add_argument("--no-auth", action="store_true", help="Skip interactive SSH/MFA.")
    auth = sub.add_parser("auth", help="Open interactive SSH so MFA can establish ControlMaster.")
    auth.add_argument("name", nargs="?", help="SSH alias or profile name (default: active cluster).")
    deploy = sub.add_parser("deploy", help="Build, convert, and push using existing cache logic.")
    deploy.add_argument("--image-profile", default="base")
    deploy.add_argument("--plan", action="store_true", help="Show the plan only.")
    deploy.add_argument("--force", action="store_true")
    deploy.add_argument("--force-convert", action="store_true")
    deploy.add_argument("--force-upload", action="store_true")
    smoke = sub.add_parser("smoke-test", help="Submit the Cartpole acceptance workload and wait.")
    smoke.add_argument("--video", action="store_true", help="Enable headless camera video.")
    smoke.add_argument("--no-wait", action="store_true", help="Submit only.")
    smoke.add_argument("--timeout", type=float, default=3600)
    smoke.add_argument("--max-iterations", type=int, default=50)
    build = sub.add_parser("build", help="Build the configured Docker image profile.")
    build.add_argument("--image-profile", default="base", help="Isaac Lab container profile (default: base).")
    build.add_argument("--force", action="store_true", help="Rebuild even if the input fingerprint is unchanged.")
    push = sub.add_parser("push", help="Convert and upload the container if fingerprints changed.")
    push.add_argument("--image-profile", default="base", help="Isaac Lab container profile (default: base).")
    push.add_argument("--force", action="store_true", help="Force rebuild, reconvert, and reupload.")
    push.add_argument("--force-convert", action="store_true", help="Re-run Apptainer conversion even if cached.")
    push.add_argument("--force-upload", action="store_true", help="Upload even if remote checksum matches.")

    run = sub.add_parser("run", help="Rsync code and submit a SLURM job.")
    run.add_argument("--task", required=True, help="Isaac Lab task id, e.g. Isaac-Cartpole-v0")
    run.add_argument("--image-profile", default=None, help="Container profile (default: from cluster profile).")
    run.add_argument("--partition", help="SLURM partition override")
    run.add_argument("--gpu", help="GPU type, e.g. a40")
    run.add_argument("--gpus", type=int, help="GPU count")
    run.add_argument("--gres", help="Full SLURM GRES string, e.g. gpu:a40:1")
    run.add_argument("--cpus", type=int, help="CPUs per task")
    run.add_argument("--memory", help="Memory, e.g. 32G")
    run.add_argument("--time", help="Walltime, e.g. 1h")
    run.add_argument("--python-executable", dest="python_executable", help="Path under Isaac Lab to train script")
    run.add_argument("--video", action="store_true", help="Enable headless camera video recording.")
    run.add_argument("--video-length", type=int, default=200)
    run.add_argument("--video-interval", type=int, default=1000)
    run.add_argument("--headless", action="store_true", help="Force --headless")
    run.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        dest="extra_args",
        help="Extra argument passed through to train.py (repeatable).",
    )

    status = sub.add_parser("status", help="Show recent or running cluster jobs.")
    status.add_argument("job_id", nargs="?", help="Optional SLURM job id")
    logs = sub.add_parser("logs", help="Show the SLURM log for a job.")
    logs.add_argument("job_id")
    logs.add_argument("--follow", action="store_true", help="Follow the log (Ctrl+C stops follow only).")
    cancel = sub.add_parser("cancel", help="Cancel a SLURM job.")
    cancel.add_argument("job_id")
    videos = sub.add_parser("videos", help="List remote training videos, newest first.")
    videos.add_argument("--job", dest="job_id", help="Optional job id hint")
    pull = sub.add_parser("pull-video", help="Copy a remote MP4 to the local machine.")
    pull.add_argument("--latest", action="store_true", help="Fetch the newest remote MP4.")
    pull.add_argument("--job", dest="job_id", help="Prefer videos associated with this job.")
    pull.add_argument("remote_video", nargs="?", help="Explicit remote MP4 path")
    pull.add_argument("--dest", type=Path, help="Local destination directory")
    sub.add_parser("shell", help="Open an interactive SSH session to the login node.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    return build_parser().parse_args(hoist_global_flags(raw))


_COMMAND_GERUNDS = {
    "init": "initializing",
    "setup": "setting up",
    "doctor": "checking",
    "bootstrap": "bootstrapping",
    "onboard": "onboarding",
    "auth": "authenticating",
    "deploy": "deploying",
    "smoke-test": "smoke-testing",
    "build": "building",
    "push": "pushing",
    "run": "running",
    "status": "checking status",
    "logs": "fetching logs",
    "cancel": "canceling",
    "videos": "listing videos",
    "pull-video": "pulling video",
    "shell": "opening shell",
}


def resolve_banner_cluster(args: argparse.Namespace, store: ConfigStore) -> str:
    """Cluster name for the command banner: --cluster, positional name, then active profile."""
    command = getattr(args, "command", None)
    if command == "bootstrap":
        return "this machine"
    for attr in ("cluster", "name"):
        value = getattr(args, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return store.get_active() or "(no cluster selected)"


def format_command_banner(args: argparse.Namespace, store: ConfigStore) -> str:
    """Loud first-line banner: what is running, and on which cluster."""
    command = args.command or "command"
    verb = _COMMAND_GERUNDS.get(command, command)
    cluster = resolve_banner_cluster(args, store)
    if cluster[0].isupper() or cluster.startswith("(") or " " in cluster:
        display_cluster = cluster
    else:
        display_cluster = cluster.title()
    subject = _banner_subject(args)
    if subject:
        body = f"{verb} {subject} on {display_cluster}"
    else:
        body = f"{verb} on {display_cluster}"
    width = max(72, len(body) + 8)
    bar = "=" * width
    return f"{bar}\n>>> {body} <<<\n{bar}"


def _banner_subject(args: argparse.Namespace) -> str | None:
    command = args.command
    if command in {"deploy", "push", "build"}:
        return image_stem(getattr(args, "image_profile", None) or "base")
    if command == "run":
        return getattr(args, "task", None)
    if command == "logs":
        return getattr(args, "job_id", None)
    if command == "cancel":
        return getattr(args, "job_id", None)
    if command == "status":
        return getattr(args, "job_id", None)
    return None


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        runner = CommandRunner(dry_run=args.dry_run, verbose=args.verbose)
        store = ConfigStore.locate(explicit=args.config_dir)
        runner.emit(format_command_banner(args, store))
        return _dispatch(args, runner, store)
    except VectorLabError as exc:
        print(exc.format_report(), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error (cli): {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace, runner: CommandRunner, store: ConfigStore) -> int:
    command = args.command
    if command == "init":
        profile = InitCommand(runner, store).run(
            args.name,
            ssh_alias=args.ssh_alias,
            isaaclab_path=args.isaaclab,
        )
        runner.emit(f"profile: {profile.name}")
        runner.emit(f"  ssh_alias: {profile.ssh_alias}")
        runner.emit(f"  cluster_type: {profile.cluster_type}")
        runner.emit(f"  resolved_host: {profile.resolved_host or '(unresolved)'}")
        runner.emit(f"  remote_user: {profile.remote_user or '(probe skipped or failed)'}")
        runner.emit(f"  home_dir: {profile.home_dir or '(unresolved)'}")
        runner.emit(f"  scratch_dir: {profile.scratch_dir or '(unresolved)'}")
        runner.emit(f"  isaaclab_path: {profile.isaaclab_path or '(not found)'}")
        runner.emit("remaining: run vector-lab setup, then doctor / build / push.")
        return 0
    if command == "setup":
        SetupCommand(runner, store).run(
            args.name,
            isaaclab=args.isaaclab,
            ssh_alias=args.ssh_alias,
        )
        return 0
    if command == "doctor":
        cluster = args.cluster or store.get_active()
        results = Doctor(runner, store, cluster=cluster).run()
        runner.emit(format_doctor_report(results))
        return 1 if any(not item.ok for item in results) else 0
    if command == "build":
        return BuildCommand(runner, store).run(
            cluster=args.cluster,
            image_profile=args.image_profile,
            force=args.force,
        )
    if command == "push":
        return PushCommand(runner, store).run(
            cluster=args.cluster,
            image_profile=args.image_profile,
            force=args.force,
            force_convert=args.force_convert,
            force_upload=args.force_upload,
        )
    if command == "run":
        return RunCommand(runner, store).run(
            cluster=args.cluster,
            task=args.task,
            image_profile=args.image_profile,
            partition=args.partition,
            gpu=args.gpu,
            gpus=args.gpus,
            gres=args.gres,
            cpus=args.cpus,
            memory=args.memory,
            time=args.time,
            python_executable=args.python_executable,
            video=args.video,
            video_length=args.video_length,
            video_interval=args.video_interval,
            headless=args.headless,
            extra_args=args.extra_args,
        )
    if command == "status":
        return StatusCommand(runner, store).run(cluster=args.cluster, job_id=args.job_id)
    if command == "logs":
        return LogsCommand(runner, store).run(
            cluster=args.cluster, job_id=args.job_id, follow=args.follow
        )
    if command == "cancel":
        return CancelCommand(runner, store).run(cluster=args.cluster, job_id=args.job_id)
    if command == "videos":
        return VideosCommand(runner, store).run(cluster=args.cluster, job_id=args.job_id)
    if command == "pull-video":
        return PullVideoCommand(runner, store).run(
            cluster=args.cluster,
            latest=args.latest,
            job_id=args.job_id,
            remote_video=args.remote_video,
            dest_dir=args.dest,
        )
    if command == "shell":
        return ShellCommand(runner, store).run(cluster=args.cluster)
    if command == "bootstrap":
        return BootstrapCommand(runner, store).run(install=args.install)
    if command == "onboard":
        return OnboardCommand(runner, store).run(
            args.name,
            isaaclab=args.isaaclab,
            ssh_alias=args.ssh_alias,
            dest=args.dest,
            clone=not args.no_clone,
            authenticate=not args.no_auth,
        )
    if command == "auth":
        alias = args.name
        if not alias:
            alias = args.cluster or store.get_active() or "bonecho"
        outcome = AuthCommand(runner, store).run(alias)
        runner.emit(f"auth {outcome.action}: {outcome.detail}")
        return 0 if outcome.authenticated else 1
    if command == "deploy":
        return DeployCommand(runner, store).run(
            cluster=args.cluster,
            image_profile=args.image_profile,
            plan=args.plan,
            force=args.force,
            force_convert=args.force_convert,
            force_upload=args.force_upload,
        )
    if command == "smoke-test":
        return SmokeTestCommand(runner, store).run(
            cluster=args.cluster,
            video=args.video,
            wait=not args.no_wait,
            timeout=args.timeout,
            max_iterations=args.max_iterations,
        )
    raise VectorLabError(f"unknown command {command}", category="cli")


if __name__ == "__main__":
    sys.exit(main())
