"""Find or clone a pinned Isaac Lab checkout. Never reset user work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vector_sim.compat import ISAACLAB_COMMIT, ISAACLAB_REMOTE, classify_isaaclab_commit
from vector_sim.exec import CommandRunner
from vector_sim.local.repo import IsaacLabRepo, discover_isaac_lab, is_isaac_lab_repo


@dataclass
class RepoResolution:
    path: Path | None
    action: str  # READY | CLONED | WOULD_CLONE | MISSING | WARN
    commit: str | None = None
    warning: str | None = None
    detail: str = ""


def default_clone_dest(start: Path | None = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    sibling = here.parent / "IsaacLab"
    # Recognize the current checkout name as well as the renamed project directory.
    if here.name in {"vector-sim", "vector-lab"}:
        return sibling
    return here / "IsaacLab"


def git_rev_parse(runner: CommandRunner, repo: Path) -> str | None:
    result = runner.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        category="git",
        check=False,
        dry_run_skip=False,
    )
    if result.skipped or result.returncode != 0:
        return None
    return result.stdout.strip() or None


def resolve_existing(path: Path, runner: CommandRunner) -> RepoResolution:
    root = path.expanduser().resolve()
    if not root.exists():
        return RepoResolution(root, "MISSING", detail="path does not exist")
    if not is_isaac_lab_repo(root):
        missing = IsaacLabRepo(root).validate()
        return RepoResolution(
            root,
            "MISSING",
            warning=f"{root} exists but is not Isaac Lab (missing {', '.join(missing)})",
            detail="will not overwrite",
        )
    commit = git_rev_parse(runner, root)
    match = classify_isaaclab_commit(commit)
    warning = None
    action = "READY"
    if match.classification != "tested":
        action = "WARN"
        warning = (
            f"Isaac Lab HEAD {commit or 'unknown'} differs from pinned "
            f"{ISAACLAB_COMMIT}. Continuing without reset."
        )
    return RepoResolution(root, action, commit=commit, warning=warning, detail=str(root))


def find_or_clone(
    runner: CommandRunner,
    *,
    configured: str | Path | None,
    start: Path | None = None,
    dest: Path | None = None,
    remote: str = ISAACLAB_REMOTE,
    commit: str = ISAACLAB_COMMIT,
    clone: bool = True,
) -> RepoResolution:
    if configured:
        existing = Path(configured).expanduser()
        if existing.exists():
            return resolve_existing(existing, runner)
        dest_path = existing
    else:
        try:
            found = discover_isaac_lab(start=start)
            return resolve_existing(found.root, runner)
        except Exception:
            dest_path = Path(dest) if dest else default_clone_dest(start)

    dest_path = dest_path.expanduser().resolve()
    if dest_path.exists():
        return resolve_existing(dest_path, runner)

    clone_cmd = ["git", "clone", remote, str(dest_path)]
    checkout_cmd = ["git", "-C", str(dest_path), "checkout", "--detach", commit]
    if runner.dry_run or not clone:
        action = "WOULD_CLONE" if clone else "MISSING"
        return RepoResolution(
            dest_path,
            action,
            commit=commit,
            detail=f"{' '.join(clone_cmd)} && {' '.join(checkout_cmd)}",
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    runner.run(clone_cmd, category="git-clone", suggestion="Check network and git credentials.")
    runner.run(
        checkout_cmd,
        category="git-checkout",
        suggestion=f"Pinned commit {commit} should exist on {remote}.",
    )
    return RepoResolution(dest_path, "CLONED", commit=commit, detail=str(dest_path))
