"""Known-good versions from the Bonecho acceptance tests.

Artifact hashes are evidence, not runtime requirements.
"""

from __future__ import annotations

from dataclasses import dataclass

# Generated wrapper contract. Increment when wrapper semantics change.
WRAPPER_SCHEMA_VERSION = "1"

# Pinned Isaac Lab checkout that passed Docker + Cartpole acceptance.
ISAACLAB_REMOTE = "https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_COMMIT = "b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8"
ISAACLAB_REF_NAME = "tested Bonecho acceptance commit"

# Local tools observed on the machine that produced the accepted deployment.
TESTED_DOCKER_VERSION = "29.6.1"
TESTED_LOCAL_APPTAINER = "1.5.3"
TESTED_PYTHON = "3.12"

# Cluster-side: exact remote Apptainer/SLURM strings were not pinned as
# requirements. Bonecho uses `module load apptainer` and login-shell SLURM.
TESTED_REMOTE_APPTAINER = "module:apptainer (Bonecho)"
TESTED_SLURM = "Bonecho login-shell SLURM (sbatch via bash -l)"

# Acceptance-test evidence only — do not gate deploy on these hashes.
EVIDENCE_CARTPOLE_JOB_ID = "390208"
EVIDENCE_DOCKER_DIGEST = (
    "sha256:0448e7e5b8b13bb9e3f304e115f77353b0dd19976b51d72eb991cd1fb99ecd4d"
)
EVIDENCE_ARTIFACT_SHA256 = "98ba9b235957a1d843759c4bb7aa55cf56f747e84240ccfd16dcc3e8ea113e7d"
EVIDENCE_PARTITION = "a40_b1"
EVIDENCE_GRES = "gpu:a40:1"
EVIDENCE_CPUS = 8
EVIDENCE_MEMORY = "32G"
EVIDENCE_TIME = "1h"
EVIDENCE_EXIT = "0:0"
EVIDENCE_STATE = "COMPLETED"


@dataclass(frozen=True)
class VersionMatch:
    name: str
    found: str | None
    tested: str | None
    classification: str  # tested | compatible | unknown
    detail: str = ""


def classify_tool_version(*, name: str, found: str | None, tested: str | None) -> VersionMatch:
    """Classify a found version against a tested string.

    Exact match → tested. Any present version for Docker/SLURM/Apptainer → compatible.
    Missing → unknown.
    """
    if not found:
        return VersionMatch(name, found, tested, "unknown", "not detected")
    if tested and _version_token(found) == _version_token(tested):
        return VersionMatch(name, found, tested, "tested", "matches known-good")
    if tested and _version_token(tested) in found:
        return VersionMatch(name, found, tested, "tested", "contains known-good version")
    return VersionMatch(name, found, tested, "compatible", "present; exact version not required")


def classify_isaaclab_commit(found: str | None) -> VersionMatch:
    if not found:
        return VersionMatch("Isaac Lab", found, ISAACLAB_COMMIT, "unknown", "HEAD not available")
    if found.startswith(ISAACLAB_COMMIT) or ISAACLAB_COMMIT.startswith(found):
        return VersionMatch("Isaac Lab", found, ISAACLAB_COMMIT, "tested", "matches pinned commit")
    return VersionMatch(
        "Isaac Lab",
        found,
        ISAACLAB_COMMIT,
        "compatible",
        f"checkout differs from pinned {ISAACLAB_COMMIT[:12]}",
    )


def _version_token(text: str) -> str:
    return text.strip().split()[-1].lstrip("v")
