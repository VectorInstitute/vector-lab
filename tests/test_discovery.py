"""Discovery must work on clusters that expose nothing to a non-login shell."""

from vector_lab.cluster.detect import candidate_scratch_dirs
from vector_lab.cluster.discover import (
    discover_apptainer,
    discover_remote_shell,
    discover_scratch,
    parse_capabilities,
    parse_module_names,
    resolve_remote_shell,
)
from vector_lab.cluster.slurm import gpu_capacity, select_default_partition
from vector_lab.config.models import ClusterProfile, GpuPartition
from vector_lab.exec import CommandRunner
from vector_lab.ssh.session import SshSession
from tests.fakes.runner import FakeExecute


LOGIN_ONLY_CAPS = "sinfo\nsbatch\nmodule\nSCRATCH\n"


def _session(fake: FakeExecute, *, login_shell: bool = False) -> tuple[SshSession, CommandRunner]:
    runner = CommandRunner(dry_run=False, execute=fake)
    return SshSession("cluster", runner, login_shell=login_shell), runner


def _is_login_call(args: list[str]) -> bool:
    return any("bash -l -c" in a for a in args)


def test_parse_capabilities_ignores_blank_lines() -> None:
    assert parse_capabilities("sinfo\n\n  module \n") == frozenset({"sinfo", "module"})


def test_remote_shell_upgrades_when_only_login_shell_has_tools() -> None:
    fake = FakeExecute()
    fake.add(lambda args: _is_login_call(args), stdout=LOGIN_ONLY_CAPS)
    fake.add(lambda args: True, stdout="")
    session, _ = _session(fake)

    probe = discover_remote_shell(session)

    assert probe.remote_shell == "login"
    assert {"sinfo", "sbatch", "module", "SCRATCH"} <= probe.capabilities


def test_remote_shell_stays_plain_when_both_shells_match() -> None:
    fake = FakeExecute()
    fake.add(lambda args: True, stdout="sinfo\nsbatch\n")
    session, _ = _session(fake)

    assert discover_remote_shell(session).remote_shell == "none"


def test_resolve_remote_shell_never_downgrades_a_login_profile() -> None:
    fake = FakeExecute()
    fake.add(lambda args: True, stdout="")
    profile = ClusterProfile(name="c", ssh_alias="c")
    profile.scheduler.remote_shell = "login"
    session, runner = _session(fake, login_shell=True)

    session, _probe = resolve_remote_shell(profile, session, runner)

    assert profile.scheduler.remote_shell == "login"
    assert session.login_shell


def test_resolve_remote_shell_switches_session_to_login() -> None:
    fake = FakeExecute()
    fake.add(lambda args: _is_login_call(args), stdout=LOGIN_ONLY_CAPS)
    fake.add(lambda args: True, stdout="")
    profile = ClusterProfile(name="c", ssh_alias="c")
    session, runner = _session(fake)

    session, _probe = resolve_remote_shell(profile, session, runner)

    assert profile.scheduler.remote_shell == "login"
    assert session.login_shell


def test_scratch_falls_back_to_an_existing_directory() -> None:
    fake = FakeExecute()
    fake.add("grep -E", stdout="")
    fake.add("for d in", stdout="/scratch/alice\n")
    session, _ = _session(fake)

    found, source = discover_scratch(session, remote_user="alice", home_dir="/home/alice")

    assert found == "/scratch/alice"
    assert source == "existing_directory"


def test_scratch_prefers_an_exported_variable() -> None:
    fake = FakeExecute()
    fake.add("grep -E", stdout="SCRATCH=/gpfs/scratch/alice\n")
    session, _ = _session(fake)

    found, source = discover_scratch(session, remote_user="alice", home_dir="/home/alice")

    assert found == "/gpfs/scratch/alice"
    assert source == "SCRATCH"


def test_candidate_scratch_dirs_are_deduplicated() -> None:
    assert candidate_scratch_dirs(remote_user="alice", home_dir="/home/alice") == [
        "/scratch/alice",
        "/home/alice/scratch",
    ]
    assert candidate_scratch_dirs(remote_user=None, home_dir=None) == []


def test_parse_module_names_drops_paths_versions_and_aliases() -> None:
    text = (
        "/cvmfs/soft/modules/Core:\n"
        "apptainer/1(@apptainer/1.2.4)\n"
        "apptainer/1.3.5\n"
        "apptainer/1.4.5\n"
        "openmpi/4.1.5\n"
    )
    assert parse_module_names(text, keywords=("apptainer", "singularity")) == ["apptainer"]


def test_apptainer_is_found_through_the_module_system() -> None:
    fake = FakeExecute()
    fake.add(
        lambda args: "module load apptainer" in " ".join(args),
        stdout="/opt/apptainer/bin/singularity\n",
    )
    fake.add("module -t avail", stdout="/cvmfs/Core:\napptainer/1.3.5\n")
    fake.add("command -v singularity", stdout="")
    session, _ = _session(fake)

    probe = discover_apptainer(session, module=None)

    assert probe.available
    assert probe.module == "apptainer"
    assert probe.command == "singularity"


def test_apptainer_probe_reports_unavailable_without_a_module() -> None:
    fake = FakeExecute()
    fake.add(lambda args: True, stdout="")
    session, _ = _session(fake)

    assert not discover_apptainer(session, module=None).available


def test_gpu_capacity_reads_typed_counts() -> None:
    assert gpu_capacity(GpuPartition(name="p", gpu_types=["l40s", "l40s=672"])) == ("l40s", 672)
    assert gpu_capacity(GpuPartition(name="p", gpu_types=["l40s"])) == (None, 0)


def test_default_partition_picks_the_largest_usable_gpu_partition() -> None:
    partitions = [
        GpuPartition(name="cpu_b1", gres="(null)", state="up", gpu_types=[]),
        GpuPartition(name="h100_b1", state="up", gpu_types=["h100", "h100=80"]),
        GpuPartition(name="l40s_b1", state="up", gpu_types=["l40s", "l40s=672"]),
        GpuPartition(name="l40s_down", state="down", gpu_types=["l40s", "l40s=999"]),
    ]

    assert select_default_partition(partitions) == ("l40s_b1", "l40s")


def test_default_partition_is_none_without_gpu_partitions() -> None:
    assert select_default_partition([GpuPartition(name="cpu", gpu_types=[])]) == (None, None)
