from vector_sim.cluster.slurm import (
    gpu_types_from_gres,
    gres_request,
    parse_scontrol_partitions,
    parse_sinfo_pipe_table,
    slurm_time,
)
from vector_sim.config.models import DefaultJob


def test_slurm_time_and_gres() -> None:
    assert slurm_time("1h") == "01:00:00"
    assert slurm_time("23:00:00") == "23:00:00"
    assert slurm_time("1:00:00") == "01:00:00"
    job = DefaultJob(gpu_type="a40", gpu_count=1)
    assert gres_request(job) == "gpu:a40:1"


def test_parse_sinfo_gpu_partitions() -> None:
    text = """
a40_b1*|gpu:a40:8|up
a100_b1|gpu:a100:4|up
rtx6000|gpu:rtx6000:4|up
cpu| (null)|up
"""
    parts = parse_sinfo_pipe_table(text)
    names = [p.name for p in parts]
    assert names == ["a40_b1", "a100_b1", "rtx6000", "cpu"]
    a40 = parts[0]
    assert a40.gpu_types == ["a40"]
    assert a40.gres == "gpu:a40:8"
    assert parts[1].gpu_types == ["a100"]
    assert parts[2].gpu_types == ["rtx6000"]


def test_parse_scontrol_partitions() -> None:
    text = """
PartitionName=a40_b1
   Gres=gpu:a40:8 MaxTime=1:00:00
PartitionName=a100_b1
   Gres=gpu:a100:4
"""
    parts = parse_scontrol_partitions(text)
    assert {p.name for p in parts} == {"a40_b1", "a100_b1"}
    assert gpu_types_from_gres("gpu:a40:8,gpu:a100:2") == ["a40", "a100"]
