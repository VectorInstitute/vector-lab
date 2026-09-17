from pathlib import Path

from vector_lab.exec import CommandRunner
from vector_lab.cluster.adapters import BonechoAdapter, SlurmAdapter
from vector_lab.images.conversion import ConversionBackend, LocalApptainerConversion, RemoteConversion


def test_conversion_backends_are_pluggable(tmp_path: Path) -> None:
    local: ConversionBackend = LocalApptainerConversion(CommandRunner(), work_root=tmp_path)
    remote: ConversionBackend = RemoteConversion()
    assert local.name == "local-apptainer"
    assert remote.name == "remote"


def test_generic_slurm_adapter_has_no_bonecho_apptainer_flags() -> None:
    adapter = SlurmAdapter()
    cfg = adapter.apptainer_defaults()
    assert cfg.module is None
    assert cfg.writable_mode is None
    assert "--writable-tmpfs" not in cfg.extra_exec_args
    assert "--containall" not in cfg.extra_exec_args


def test_bonecho_adapter_owns_apptainer_and_path_defaults() -> None:
    adapter = BonechoAdapter()
    cfg = adapter.apptainer_defaults()
    assert cfg.module == "apptainer"
    assert cfg.writable_mode == "writable-tmpfs"
    assert cfg.extra_exec_args == ["--nv", "--containall", "--writable-tmpfs"]
    paths = adapter.path_defaults("alice")
    assert paths.home_dir == "/h/alice"
    assert paths.scratch_dir == "/scratch/alice"
