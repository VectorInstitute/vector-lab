from pathlib import Path

from vector_lab.cluster.adapters import (
    CLUSTER_TYPE_SLURM,
    CLUSTER_TYPE_VECTOR_SLURM,
    SlurmAdapter,
    VectorSlurmAdapter,
    adapter_for,
)
from vector_lab.commands.init import build_initial_profile
from vector_lab.config.models import BUILTIN_PROFILES, ClusterProfile, bonecho_defaults
from vector_lab.exec import CommandRunner
from vector_lab.images.conversion import ConversionBackend, LocalApptainerConversion, RemoteConversion
from vector_lab.jobs.generate import generate_runtime


def test_conversion_backends_are_pluggable(tmp_path: Path) -> None:
    local: ConversionBackend = LocalApptainerConversion(CommandRunner(), work_root=tmp_path)
    remote: ConversionBackend = RemoteConversion()
    assert local.name == "local-apptainer"
    assert remote.name == "remote"


def test_bonecho_defaults_uses_vector_slurm() -> None:
    profile = bonecho_defaults()
    assert profile.name == "bonecho"
    assert profile.ssh_alias == "bonecho"
    assert profile.cluster_type == CLUSTER_TYPE_VECTOR_SLURM
    assert "bonecho" in BUILTIN_PROFILES


def test_bonecho_resolves_vector_slurm_adapter() -> None:
    profile = build_initial_profile("bonecho")
    assert profile.cluster_type == CLUSTER_TYPE_VECTOR_SLURM
    adapter = adapter_for(profile.cluster_type)
    assert isinstance(adapter, VectorSlurmAdapter)
    assert adapter.name == CLUSTER_TYPE_VECTOR_SLURM


def test_arbitrary_profile_with_vector_slurm_resolves_adapter() -> None:
    profile = ClusterProfile(
        name="my-custom-lab",
        ssh_alias="my-custom-lab",
        cluster_type=CLUSTER_TYPE_VECTOR_SLURM,
    )
    assert profile.name not in BUILTIN_PROFILES
    adapter = adapter_for(profile.cluster_type)
    assert isinstance(adapter, VectorSlurmAdapter)


def test_ssh_target_can_differ_from_profile_name() -> None:
    profile = bonecho_defaults(ssh_alias="bonecho-login")
    profile.name = "lab-a"
    profile.remote_user = "alice"
    profile.scratch_dir = "/scratch/alice"
    VectorSlurmAdapter().derive_scratch_paths(profile)
    assert profile.name == "lab-a"
    assert profile.ssh_alias == "bonecho-login"
    assert profile.cluster_type == CLUSTER_TYPE_VECTOR_SLURM
    env = generate_runtime(profile).env_cluster
    assert "CLUSTER_LOGIN=bonecho-login" in env
    assert isinstance(adapter_for(profile.cluster_type), VectorSlurmAdapter)


def test_adapter_selection_does_not_inspect_profile_name() -> None:
    # Same cluster_type, wildly different names → same adapter.
    a = adapter_for(
        ClusterProfile(name="bonecho", ssh_alias="x", cluster_type=CLUSTER_TYPE_VECTOR_SLURM).cluster_type
    )
    b = adapter_for(
        ClusterProfile(name="totally-unrelated", ssh_alias="y", cluster_type=CLUSTER_TYPE_VECTOR_SLURM).cluster_type
    )
    assert type(a) is type(b) is VectorSlurmAdapter
    # Name alone never selects VectorSlurmAdapter.
    bare = ClusterProfile(name="bonecho", ssh_alias="bonecho", cluster_type=CLUSTER_TYPE_SLURM)
    assert isinstance(adapter_for(bare.cluster_type), SlurmAdapter)
    assert not isinstance(adapter_for(bare.cluster_type), VectorSlurmAdapter)


def test_generic_slurm_adapter_remains_separate() -> None:
    generic = adapter_for(CLUSTER_TYPE_SLURM)
    assert isinstance(generic, SlurmAdapter)
    assert not isinstance(generic, VectorSlurmAdapter)
    cfg = generic.apptainer_defaults()
    assert cfg.module is None
    assert cfg.writable_mode is None
    assert "--writable-tmpfs" not in cfg.extra_exec_args


def test_vector_slurm_adapter_owns_apptainer_and_path_defaults() -> None:
    adapter = VectorSlurmAdapter()
    cfg = adapter.apptainer_defaults()
    assert cfg.module == "apptainer"
    assert cfg.writable_mode == "writable-tmpfs"
    assert cfg.extra_exec_args == ["--nv", "--containall", "--writable-tmpfs"]
    paths = adapter.path_defaults("alice")
    assert paths.home_dir == "/h/alice"
    assert paths.scratch_dir == "/scratch/alice"


def test_legacy_profile_yaml_fills_missing_cluster_type_from_builtin(tmp_path) -> None:
    from vector_lab.config.store import ConfigStore

    store = ConfigStore(tmp_path)
    # Simulate a pre-cluster_type bonecho.yaml (key absent on disk).
    path = store.profile_path("bonecho")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "name: bonecho\n"
        "ssh_alias: bonecho\n"
        "scheduler:\n"
        "  type: slurm\n"
        "  remote_shell: login\n"
    )
    loaded = store.load_profile("bonecho")
    assert loaded.cluster_type == CLUSTER_TYPE_VECTOR_SLURM
    assert isinstance(adapter_for(loaded.cluster_type), VectorSlurmAdapter)


def test_adapter_for_ignores_case_and_whitespace() -> None:
    assert isinstance(adapter_for(" Vector-Slurm "), VectorSlurmAdapter)
    assert isinstance(adapter_for(None), SlurmAdapter)
