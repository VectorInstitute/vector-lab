"""Docker PATH lookup, permission diagnostics, and primary-group checks."""

from __future__ import annotations

import grp
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from vector_lab.errors import DiscoveryError

DOCKER_MISSING_HINT = (
    "Install Docker Engine and ensure `docker` is on PATH "
    "(/snap/bin/docker is fine). Ubuntu: https://docs.docker.com/engine/install/ubuntu/"
)

DOCKER_ACCESS_HINT = (
    "Docker cannot access the daemon. Recommended: sudo usermod -aG docker $USER  "
    "then log out and back in. Keep docker as a supplementary group; do not use "
    "newgrp docker, chmod 666 on docker.sock, or sudo for the whole workflow."
)

FAKEROOT_HINT = (
    "Primary group is docker. Apptainer fakeroot/newgidmap often fails in that state. "
    "Keep your normal primary group; docker should only be supplementary. "
    "Do not use newgrp docker."
)


def which_docker() -> str | None:
    return shutil.which("docker")


def require_docker() -> str:
    path = which_docker()
    if not path:
        raise DiscoveryError(
            "Docker is not on PATH",
            suggestion=DOCKER_MISSING_HINT,
        )
    return path


def docker_socket_accessible(sock: Path | None = None) -> bool:
    path = sock or Path("/var/run/docker.sock")
    return path.exists() and os.access(path, os.R_OK | os.W_OK)


def permission_denied(stderr: str) -> bool:
    text = stderr.lower()
    return "permission denied" in text or "dial unix /var/run/docker.sock" in text


def primary_group_name(gid: int | None = None) -> str:
    return grp.getgrgid(os.getgid() if gid is None else gid).gr_name


def fakeroot_primary_group_problem(group_name: str | None = None) -> str | None:
    name = group_name if group_name is not None else primary_group_name()
    if name == "docker":
        return FAKEROOT_HINT
    return None


@dataclass
class DockerAccess:
    binary: str | None
    socket_ok: bool
    daemon_ok: bool | None = None
    stderr: str = ""

    @property
    def ok(self) -> bool:
        if not self.binary:
            return False
        if self.daemon_ok is not None:
            return self.daemon_ok
        return self.socket_ok
