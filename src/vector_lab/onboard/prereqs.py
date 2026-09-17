"""Local prerequisite inspection. Never runs sudo."""

from __future__ import annotations

import os
import pwd
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from vector_lab.local.docker_util import (
    DOCKER_ACCESS_HINT,
    docker_socket_accessible,
    fakeroot_primary_group_problem,
    primary_group_name,
    which_docker,
)

STATUS_INSTALLED = "INSTALLED"
STATUS_REQUIRED = "REQUIRED"
STATUS_USER_ACTION = "USER ACTION REQUIRED"
STATUS_UNSUPPORTED = "UNSUPPORTED"

UBUNTU_HINT = (
    "v1 bootstrap supports Ubuntu-like Linux. Other platforms: install git, ssh, "
    "rsync, Docker, and Apptainer yourself, then re-run vector-lab bootstrap."
)


@dataclass
class Prereq:
    name: str
    status: str
    detail: str = ""
    install_command: str | None = None
    required: bool = True

    @property
    def ok(self) -> bool:
        return self.status == STATUS_INSTALLED


@dataclass
class PlatformInfo:
    system: str
    like: str  # ubuntu | linux | other
    pretty: str
    supported_bootstrap: bool
    detail: str = ""


@dataclass
class PrereqReport:
    platform: PlatformInfo
    items: list[Prereq] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(item.ok or not item.required for item in self.items)


def read_os_release(path: Path | None = None) -> dict[str, str]:
    target = path or Path("/etc/os-release")
    data: dict[str, str] = {}
    if not target.is_file():
        return data
    for line in target.read_text(errors="replace").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value.strip().strip('"')
    return data


def inspect_platform(*, os_release: dict[str, str] | None = None, system: str | None = None) -> PlatformInfo:
    sysname = (system or os.uname().sysname).lower()
    release = os_release if os_release is not None else read_os_release()
    ident = (release.get("ID") or "").lower()
    like = (release.get("ID_LIKE") or "").lower()
    pretty = release.get("PRETTY_NAME") or sysname
    ubuntu_like = ident == "ubuntu" or "ubuntu" in like or ident == "debian" or "debian" in like
    if sysname != "linux":
        return PlatformInfo(sysname, "other", pretty, False, "vector-lab v1 targets Linux.")
    if ubuntu_like:
        return PlatformInfo(sysname, "ubuntu", pretty, True, "Ubuntu-like Linux; apt install commands apply.")
    return PlatformInfo(sysname, "linux", pretty, False, UBUNTU_HINT)


def supplementary_groups() -> list[str]:
    user = pwd.getpwuid(os.getuid()).pw_name
    gids = os.getgrouplist(user, os.getgid())
    names: list[str] = []
    import grp

    for gid in gids:
        try:
            names.append(grp.getgrgid(gid).gr_name)
        except KeyError:
            continue
    return names


def inspect_prereqs(*, min_free_bytes: int = 40 * 1024**3, start: Path | None = None) -> PrereqReport:
    platform = inspect_platform()
    items: list[Prereq] = []
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    py_ok = sys.version_info >= (3, 10)
    items.append(
        Prereq(
            "Python",
            STATUS_INSTALLED if py_ok else STATUS_REQUIRED,
            f"{sys.version.split()[0]} (need >= 3.10)",
            required=True,
        )
    )

    apt = {
        "git": "sudo apt-get update && sudo apt-get install -y git",
        "ssh": "sudo apt-get update && sudo apt-get install -y openssh-client",
        "rsync": "sudo apt-get update && sudo apt-get install -y rsync",
    }
    for tool, cmd in apt.items():
        path = shutil.which(tool)
        items.append(
            Prereq(
                tool,
                STATUS_INSTALLED if path else STATUS_USER_ACTION if platform.supported_bootstrap else STATUS_REQUIRED,
                path or f"{tool} not found on PATH",
                install_command=None if path else cmd,
                required=True,
            )
        )

    docker = which_docker()
    if docker:
        items.append(Prereq("docker", STATUS_INSTALLED, docker, required=True))
        sock_ok = docker_socket_accessible()
        items.append(
            Prereq(
                "Docker access",
                STATUS_INSTALLED if sock_ok else STATUS_USER_ACTION,
                docker if sock_ok else DOCKER_ACCESS_HINT,
                install_command=None if sock_ok else f"sudo usermod -aG docker {pwd.getpwuid(os.getuid()).pw_name}",
                required=True,
            )
        )
    else:
        items.append(
            Prereq(
                "docker",
                STATUS_USER_ACTION if platform.supported_bootstrap else STATUS_REQUIRED,
                "Docker not on PATH (Snap's /snap/bin/docker is fine)",
                install_command="Install Docker Engine; then: sudo usermod -aG docker $USER  # logout/login required",
                required=True,
            )
        )

    apptainer = shutil.which("apptainer") or shutil.which("singularity")
    items.append(
        Prereq(
            "apptainer",
            STATUS_INSTALLED if apptainer else STATUS_USER_ACTION,
            apptainer or "needed for local Docker→Apptainer conversion",
            install_command=None
            if apptainer
            else "See https://apptainer.org/docs/admin/main/installation.html (do not chmod 666 docker.sock)",
            required=True,
        )
    )

    groups = supplementary_groups()
    primary = primary_group_name()
    group_status = STATUS_INSTALLED
    group_required = False
    group_cmd = None
    if primary == "docker":
        group_status = STATUS_USER_ACTION
        group_required = True
        group_cmd = fakeroot_primary_group_problem(primary)
    elif docker and "docker" not in groups:
        group_status = STATUS_USER_ACTION
        group_cmd = f"sudo usermod -aG docker {pwd.getpwuid(os.getuid()).pw_name}  # then logout/login"
    items.append(
        Prereq(
            "groups",
            group_status,
            f"primary={primary} supplementary={','.join(groups)}",
            install_command=group_cmd,
            required=group_required,
        )
    )

    root = Path(start or Path.cwd())
    try:
        free = shutil.disk_usage(root).free
        gb = free / 1024**3
        enough = free >= min_free_bytes
        items.append(
            Prereq(
                "disk",
                STATUS_INSTALLED if enough else STATUS_USER_ACTION,
                f"{gb:.1f} GiB free at {root} (conversion may need ~{min_free_bytes / 1024**3:.0f} GiB)",
                required=False,
            )
        )
    except OSError as exc:
        items.append(Prereq("disk", STATUS_USER_ACTION, str(exc), required=False))

    if not platform.supported_bootstrap:
        items.append(
            Prereq(
                "platform",
                STATUS_UNSUPPORTED,
                platform.detail,
                required=False,
            )
        )
    else:
        items.append(Prereq("platform", STATUS_INSTALLED, platform.pretty, required=False))

    return PrereqReport(platform=platform, items=items)


def format_prereq_report(report: PrereqReport) -> str:
    lines = [
        f"Platform            {report.platform.pretty}",
        f"Bootstrap support   {'Ubuntu-like Linux' if report.platform.supported_bootstrap else 'not auto-installed (v1)'}",
        "",
    ]
    for item in report.items:
        lines.append(f"{item.name:<20} {item.status:<22} {item.detail}")
        if item.install_command and item.status != STATUS_INSTALLED:
            lines.append(f"{'':<20} command: {item.install_command}")
    lines.append("")
    lines.append("vector-lab never runs sudo unless you copy-paste a command yourself.")
    lines.append("Do not use newgrp docker, chmod 666 /var/run/docker.sock, or sudo for the whole workflow.")
    return "\n".join(lines)
