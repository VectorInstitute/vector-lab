"""GUI mode: parsing, generated wrappers, and readiness formatting. No GPU or X server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vector_sim.cluster.adapters import VectorSlurmAdapter
from vector_sim.cli import parse_args
from vector_sim.commands.gui import GuiCommand
from vector_sim.config.models import JobRecord, bonecho_defaults
from vector_sim.config.store import ConfigStore
from vector_sim.errors import VectorSimError
from vector_sim.exec import CommandRunner
from vector_sim.jobs.generate import GUI_BOOTSTRAP_NAME, generate_runtime
from vector_sim.jobs.gui import (
    DISPLAY_MAX,
    DISPLAY_MIN,
    NOVNC_PORT_MAX,
    NOVNC_PORT_MIN,
    VNC_PORT_MAX,
    VNC_PORT_MIN,
    classification_from_log,
    display_is_taken,
    format_browser_url,
    format_gui_failure,
    format_gui_ready,
    format_tunnel_command,
    gui_cache_paths,
    isaac_gui_log_ok,
    parse_gui_result,
    select_free_display,
    select_free_port,
)
from tests.fakes.runner import FakeExecute


def _profile():
    profile = bonecho_defaults()
    profile.remote_user = "alice"
    profile.scratch_dir = "/scratch/alice"
    VectorSlurmAdapter().derive_scratch_paths(profile)
    return profile


def _isaaclab(tmp_path: Path) -> Path:
    root = tmp_path / "IsaacLab"
    (root / "docker/cluster").mkdir(parents=True)
    (root / "source/isaaclab").mkdir(parents=True)
    (root / "docker/container.py").write_text("print('x')\n")
    (root / "docker/cluster/cluster_interface.sh").write_text("#!/bin/bash\n")
    (root / "isaaclab.sh").write_text("#!/bin/bash\n")
    (root / "docker/Dockerfile.base").write_text("FROM x\n")
    (root / "docker/docker-compose.yaml").write_text("services: {}\n")
    (root / "docker/.env.base").write_text(
        "ACCEPT_EULA=Y\nDOCKER_ISAACSIM_ROOT_PATH=/isaac-sim\nDOCKER_USER_HOME=/root\n"
    )
    return root


def _store(tmp_path: Path, repo: Path) -> ConfigStore:
    store = ConfigStore(tmp_path / "cfg")
    profile = _profile()
    profile.isaaclab_path = str(repo)
    store.save_profile(profile)
    store.set_active("bonecho")
    return store


def _ready_payload() -> dict:
    return {
        "classification": "GUI_READY",
        "status": "ready",
        "job_id": "42",
        "hostname": "bn070",
        "private_ip": "172.17.8.10",
        "display": ":97",
        "vnc_port": 5903,
        "novnc_port": 6084,
        "detail": "Isaac GUI is reachable through noVNC",
        "logs": {
            "isaac": "/scratch/alice/run/logs/gui/isaac.log",
            "xvfb": "/scratch/alice/run/logs/gui/xvfb.log",
            "x11vnc": "/scratch/alice/run/logs/gui/x11vnc.log",
            "websockify": "/scratch/alice/run/logs/gui/websockify.log",
            "cuda": "/scratch/alice/run/logs/gui/cuda.log",
        },
    }


def test_parse_gui_cli() -> None:
    args = parse_args(
        ["gui", "--time", "3h", "--resolution", "1920x1080x24", "--partition", "a40_b1", "--gpu", "a40"]
    )
    assert args.command == "gui"
    assert args.time == "3h"
    assert args.resolution == "1920x1080x24"
    assert args.partition == "a40_b1"
    assert args.gpu == "a40"
    assert args.ready_timeout == 2400
    assert args.python_executable is None


def test_headless_wrappers_unchanged_and_gui_adds_startup() -> None:
    profile = _profile()
    headless = generate_runtime(profile)
    gui = generate_runtime(profile, gui=True, resolution="1920x1080x24")

    assert "set -e" in headless.run_singularity
    assert "--nv" in headless.run_singularity
    assert "--containall" in headless.run_singularity
    assert "Xvfb" not in headless.run_singularity
    assert "x11vnc" not in headless.run_singularity
    assert "websockify" not in headless.run_singularity
    assert "VECTOR_SIM_GUI" not in headless.submit_job_slurm
    assert GUI_BOOTSTRAP_NAME not in headless.as_dict()
    assert "scripts/reinforcement_learning/rsl_rl/train.py" in headless.env_cluster
    assert "#SBATCH --job-name=\"training-" in headless.submit_job_slurm

    script = gui.run_singularity
    assert "Xvfb" in script
    assert "x11vnc" in script
    assert "websockify" in script
    assert "-screen 0 1920x1080x24 +extension GLX +render -noreset -ac" in script
    assert f"seq {DISPLAY_MIN} {DISPLAY_MAX}" in script
    assert f"seq {VNC_PORT_MIN} {VNC_PORT_MAX}" in script
    assert f"seq {NOVNC_PORT_MIN} {NOVNC_PORT_MAX}" in script
    assert "-localhost" in script
    assert "-rfbport" in script
    assert "127.0.0.1:$VNC_PORT" in script
    assert "0.0.0.0:$port" in script
    assert "unset LIVESTREAM" in script
    assert "--livestream" not in script
    assert "49100" not in script
    assert "47998" not in script
    assert script.index("CUDA_MICROTEST PASS") < script.index("/tmp/.X11-unix:/tmp/.X11-unix")
    assert script.index("/tmp/.X11-unix:/tmp/.X11-unix") < script.index("-localhost")
    assert "VECTOR_SIM_GUI=1" in gui.submit_job_slurm
    assert "module load apptainer" in gui.submit_job_slurm
    assert "bash -l -c 'sbatch < job.sh'" in gui.submit_job_slurm
    assert "run_singularity.sh" in gui.submit_job_slurm
    assert "scripts/tutorials/00_sim/create_empty.py" in gui.env_cluster
    assert "--headless" not in script
    assert GUI_BOOTSTRAP_NAME in gui.as_dict()


def test_free_display_and_port_selection() -> None:
    assert select_free_display(set()) == DISPLAY_MIN
    assert select_free_display({90, 91, 92}) == 93
    assert select_free_display(set(range(DISPLAY_MIN, DISPLAY_MAX + 1))) is None
    assert display_is_taken(sockets={"X95"}, locks=set(), number=95)
    assert display_is_taken(sockets=set(), locks={".X96-lock"}, number=96)
    assert not display_is_taken(sockets={"X95"}, locks=set(), number=96)

    listeners = {5900, 5901}
    assert select_free_port(listeners, low=VNC_PORT_MIN, high=VNC_PORT_MAX) == 5902
    assert select_free_port(set(range(NOVNC_PORT_MIN, NOVNC_PORT_MAX + 1)), low=NOVNC_PORT_MIN, high=NOVNC_PORT_MAX) is None
    assert select_free_port(set(), low=6080, high=6109) == 6080


def test_gui_cache_layout_and_bootstrap_reuses_it() -> None:
    paths = gui_cache_paths("/scratch/alice/docker-isaac-sim")
    assert paths.x11vnc == "/scratch/alice/docker-isaac-sim/gui/x11vnc"
    assert paths.novnc == "/scratch/alice/docker-isaac-sim/gui/noVNC-1.5.0"
    bootstrap = generate_runtime(_profile(), gui=True).gui_bootstrap or ""
    assert paths.x11vnc in bootstrap
    assert "noVNC-1.5.0" in bootstrap
    assert "GUI_RUNTIME_REUSED" in bootstrap
    assert "GUI_RUNTIME_BOOTSTRAPPED" in bootstrap
    assert "apt-get" not in bootstrap
    assert "sudo " not in bootstrap
    assert "LibVNCServer-0.9.15" in bootstrap
    assert "x11vnc-0.9.17" in bootstrap


def test_cleanup_trap_kills_gui_processes_and_keeps_cache() -> None:
    script = generate_runtime(_profile(), gui=True).run_singularity
    assert "trap cleanup EXIT" in script
    assert "trap 'cleanup; exit 143' TERM" in script
    assert "trap 'cleanup; exit 130' INT" in script
    cleanup = script.split("cleanup() {", 1)[1].split("resolve_tool()", 1)[0]
    for pid in ("ISAAC_PID", "VNC_PID", "WS_PID", "XVFB_PID"):
        assert pid in cleanup
    assert 'kill "$pid"' in cleanup
    assert "gui/x11vnc" not in cleanup
    assert "noVNC" not in cleanup
    assert 'rm -rf "$CLUSTER_ISAAC_SIM_CACHE_DIR"' not in cleanup


def test_parse_gui_result_and_connection_instructions() -> None:
    payload = _ready_payload()
    parsed = parse_gui_result("banner\n" + json.dumps(payload) + "\n")
    assert parsed is not None
    assert parsed.classification == "GUI_READY"
    assert parsed.display == ":97"
    assert parsed.vnc_port == 5903
    assert parsed.novnc_port == 6084
    report = format_gui_ready(parsed, ssh_alias="bonecho")
    assert report.startswith("GUI READY\n")
    assert "Job:       42\n" in report
    assert "Node:      bn070\n" in report
    assert "IP:        172.17.8.10\n" in report
    assert "Display:   :97\n" in report
    assert "VNC:       localhost:5903\n" in report
    assert "noVNC:     172.17.8.10:6084\n" in report
    tunnel = format_tunnel_command(ssh_alias="bonecho", private_ip="172.17.8.10", port=6084)
    assert tunnel in report
    assert "ControlPath" not in tunnel
    assert "ControlMaster" not in tunnel
    assert "mollysun" not in report
    assert format_browser_url(6084) in report
    assert report.strip().endswith("http://127.0.0.1:6084/vnc.html?autoconnect=1&resize=remote")

    failed = parse_gui_result(
        json.dumps(
            {
                "classification": "ISAAC_GUI_FAILED",
                "status": "failed",
                "detail": "Isaac process exited before the GUI was ready",
                "logs": {"isaac": "/scratch/alice/run/logs/gui/isaac.log"},
            }
        )
    )
    assert failed is not None
    text = format_gui_failure(failed)
    assert text.startswith("ISAAC_GUI_FAILED\n")
    assert "Log: /scratch/alice/run/logs/gui/isaac.log" in text
    assert "GUI failed" not in text
    assert parse_gui_result("") is None
    assert parse_gui_result("{not json") is None
    assert classification_from_log("noise\nGUI_CLASSIFICATION=XVFB_FAILED\n") == "XVFB_FAILED"


def test_isaac_graphics_log_requires_gpu_vulkan_rtx_and_rejects_llvmpipe() -> None:
    good = "app ready\nUsing device NVIDIA A40\nvulkan device\nRTX ResourceManager ready\n"
    assert isaac_gui_log_ok(good, gpu_name="NVIDIA A40")
    assert not isaac_gui_log_ok(good + "llvmpipe", gpu_name="NVIDIA A40")
    assert not isaac_gui_log_ok("app ready\nvulkan\nRTX\n", gpu_name="NVIDIA A40")
    assert not isaac_gui_log_ok(good.replace("vulkan", "opengl"), gpu_name="NVIDIA A40")


def test_gui_dry_run_does_not_submit(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = _store(tmp_path, repo)
    fake = FakeExecute()
    runner = CommandRunner(dry_run=True, execute=fake)
    code = GuiCommand(runner, store, start_dir=repo).run(resolution="1920x1080x24")
    assert code == 0
    assert fake.calls == []
    assert not any(call.category == "sbatch" for call in runner.calls)
    assert store.load_jobs() == []
    assert not (repo / ".vector-sim/generated/run_singularity.sh").is_file()


def test_gui_command_prints_tunnel_when_ready(tmp_path: Path, capsys) -> None:
    repo = _isaaclab(tmp_path)
    store = _store(tmp_path, repo)
    fake = FakeExecute()
    fake.add("test -f", stdout="CONTAINER_OK\n")
    fake.add("command -v sbatch", stdout="/usr/bin/sbatch\n")
    fake.add("gui_bootstrap.sh", stdout="GUI_RUNTIME_REUSED\n")
    fake.add("submit_job_slurm.sh", stdout="Submitted batch job 42\n")
    fake.add("result.json", stdout=json.dumps(_ready_payload()) + "\n")
    fake.add("vnc.html", stdout="200\n")
    runner = CommandRunner(execute=fake)
    code = GuiCommand(runner, store, start_dir=repo, sleep=lambda _s: None).run(
        ready_timeout=30,
        poll_interval=0,
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "GUI READY" in out
    assert "ssh \\\n  -o ExitOnForwardFailure=yes \\\n  -N \\\n  -L 6084:172.17.8.10:6084 \\\n  bonecho" in out
    assert "http://127.0.0.1:6084/vnc.html?autoconnect=1&resize=remote" in out
    assert "ControlPath" not in out
    record = store.find_job("42")
    assert record is not None
    assert record.gui is True
    assert record.gui_status == "ready"
    assert record.display == ":97"
    assert record.vnc_port == 5903
    assert record.novnc_port == 6084
    assert record.private_ip == "172.17.8.10"
    assert record.task == "Isaac-Sim-GUI"


def test_gui_command_reports_component_failure(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = _store(tmp_path, repo)
    fake = FakeExecute()
    fake.add("test -f", stdout="CONTAINER_OK\n")
    fake.add("command -v sbatch", stdout="/usr/bin/sbatch\n")
    fake.add("gui_bootstrap.sh", stdout="GUI_RUNTIME_REUSED\n")
    fake.add("submit_job_slurm.sh", stdout="Submitted batch job 77\n")
    fake.add(
        "result.json",
        stdout=json.dumps(
            {
                "classification": "XVFB_FAILED",
                "status": "failed",
                "detail": "no free display passed xdpyinfo",
                "logs": {"xvfb": "/scratch/alice/run/logs/gui/xvfb.log"},
            }
        ),
    )
    runner = CommandRunner(execute=fake)
    with pytest.raises(VectorSimError) as caught:
        GuiCommand(runner, store, start_dir=repo, sleep=lambda _s: None).run(
            ready_timeout=30,
            poll_interval=0,
        )
    assert caught.value.category == "XVFB_FAILED"
    assert "xvfb.log" in (caught.value.suggestion or "")
    assert "GUI failed" not in str(caught.value)
    record = store.find_job("77")
    assert record is not None and record.gui_status == "failed"


def test_gui_command_times_out_with_log_path(tmp_path: Path) -> None:
    repo = _isaaclab(tmp_path)
    store = _store(tmp_path, repo)
    fake = FakeExecute()
    fake.add("test -f", stdout="CONTAINER_OK\n")
    fake.add("command -v sbatch", stdout="/usr/bin/sbatch\n")
    fake.add("gui_bootstrap.sh", stdout="GUI_RUNTIME_REUSED\n")
    fake.add("submit_job_slurm.sh", stdout="Submitted batch job 88\n")
    fake.add("squeue", stdout="88|gui|RUNNING|00:01:00|a40_b1|bn070\n")
    runner = CommandRunner(execute=fake)
    with pytest.raises(VectorSimError) as caught:
        GuiCommand(runner, store, start_dir=repo, sleep=lambda _s: None).run(
            ready_timeout=0,
            poll_interval=0,
        )
    assert caught.value.category == "GUI_TIMEOUT"
    assert "slurm-88.out" in (caught.value.suggestion or "")
    assert store.find_job("88") is not None


def test_gui_job_record_roundtrip(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path)
    record = JobRecord.create(
        job_id="42",
        cluster="bonecho",
        remote_run_dir="/scratch/alice/isaaclab_1",
        slurm_log="/scratch/alice/isaaclab_1/slurm-42.out",
        task="Isaac-Sim-GUI",
        image_profile="base",
        gui=True,
        gui_status="ready",
        hostname="bn070",
        private_ip="172.17.8.10",
        display=":97",
        vnc_port=5903,
        novnc_port=6084,
    )
    store.upsert_job(record)
    loaded = store.find_job("42")
    assert loaded is not None
    assert loaded.gui is True
    assert loaded.novnc_port == 6084
    assert loaded.display == ":97"
