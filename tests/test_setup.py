from vector_lab.cluster.adapters import BonechoAdapter
from vector_lab.commands.setup import SetupCommand
from vector_lab.config.models import bonecho_defaults
from vector_lab.config.store import ConfigStore
from vector_lab.exec import CommandRunner
from vector_lab.jobs.generate import ENV_NAME, RUNNER_NAME, SUBMIT_NAME
from tests.fakes.runner import FakeExecute


def test_setup_generates_files_and_skips_remote_mkdir_in_dry_run(tmp_path) -> None:
    store = ConfigStore(tmp_path)
    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.scratch_dir = "/scratch/alice"
    BonechoAdapter().derive_scratch_paths(profile)
    store.save_profile(profile)
    store.set_active("bonecho")

    fake = FakeExecute()
    fake.add("sinfo", stdout="a40_b1|gpu:a40:8|up\na100_b1|gpu:a100:4|up\n")
    fake.add("scontrol", stdout="PartitionName=a40_b1 Gres=gpu:a40:8\n")
    fake.add("command -v singularity", stdout="/usr/bin/singularity\n")
    runner = CommandRunner(dry_run=True, execute=fake)
    SetupCommand(runner, store, start_dir=tmp_path).run("bonecho")

    generated = store.generated_dir
    assert (generated / ENV_NAME).is_file()
    assert (generated / SUBMIT_NAME).is_file()
    assert (generated / RUNNER_NAME).is_file()
    submit = (generated / SUBMIT_NAME).read_text()
    assert "module load apptainer" in submit
    env = (generated / ENV_NAME).read_text()
    assert "/scratch/alice/isaaclab-containers" in env
    joined = [" ".join(call.args) for call in runner.calls]
    assert any("mkdir -p" in item for item in joined)
    assert fake.calls == []
