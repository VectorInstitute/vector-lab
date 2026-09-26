"""noVNC GUI session: display/port selection, runtime cache, and readiness.

Headless ``vector-lab run`` does not import this module's script renderers
except through ``generate_runtime(..., gui=True)``.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field

from vector_lab.config.models import JobRecord

# Proven by the Bonecho noVNC spike. Do not compile these on every job.
LIBVNC_VERSION = "0.9.15"
X11VNC_VERSION = "0.9.17"
NOVNC_VERSION = "1.5.0"
GUI_PYTHON = "scripts/tutorials/00_sim/create_empty.py"
DEFAULT_RESOLUTION = "1920x1080x24"

DISPLAY_MIN = 90
DISPLAY_MAX = 119
VNC_PORT_MIN = 5900
VNC_PORT_MAX = 5929
NOVNC_PORT_MIN = 6080
NOVNC_PORT_MAX = 6109

GUI_CLASSIFICATIONS = (
    "GPU_UNHEALTHY",
    "XVFB_FAILED",
    "ISAAC_GUI_FAILED",
    "VNC_FAILED",
    "NOVNC_FAILED",
    "GUI_READY",
)

_RESOLUTION = re.compile(r"^(\d+)x(\d+)x(\d+)$")
_FAILURE_LOG = {
    "GPU_UNHEALTHY": "cuda",
    "XVFB_FAILED": "xvfb",
    "ISAAC_GUI_FAILED": "isaac",
    "VNC_FAILED": "x11vnc",
    "NOVNC_FAILED": "websockify",
}

LIBVNC_URL = (
    "https://github.com/LibVNC/libvncserver/archive/refs/tags/"
    f"LibVNCServer-{LIBVNC_VERSION}.tar.gz"
)
X11VNC_URL = f"https://github.com/LibVNC/x11vnc/archive/refs/tags/{X11VNC_VERSION}.tar.gz"
NOVNC_URL = f"https://github.com/novnc/noVNC/archive/refs/tags/v{NOVNC_VERSION}.tar.gz"


@dataclass(frozen=True)
class GuiCachePaths:
    """Persistent GUI runtime under the cluster cache. Not deleted with a job."""

    root: str
    x11vnc: str
    novnc: str
    version_file: str

    @property
    def version_text(self) -> str:
        return (
            f"libvncserver={LIBVNC_VERSION}\n"
            f"x11vnc={X11VNC_VERSION}\n"
            f"novnc={NOVNC_VERSION}\n"
        )


@dataclass(frozen=True)
class GuiLaunch:
    resolution: str = DEFAULT_RESOLUTION
    display_min: int = DISPLAY_MIN
    display_max: int = DISPLAY_MAX
    vnc_min: int = VNC_PORT_MIN
    vnc_max: int = VNC_PORT_MAX
    novnc_min: int = NOVNC_PORT_MIN
    novnc_max: int = NOVNC_PORT_MAX
    isaac_polls: int = 90
    isaac_poll_seconds: int = 10

    def __post_init__(self) -> None:
        parse_resolution(self.resolution)
        if self.display_min > self.display_max:
            raise ValueError("display range is empty")
        if self.vnc_min > self.vnc_max or self.novnc_min > self.novnc_max:
            raise ValueError("port range is empty")

    @property
    def pixels(self) -> str:
        width, height, _depth = _RESOLUTION.fullmatch(self.resolution).groups()  # type: ignore[union-attr]
        return f"{width}x{height}"


@dataclass
class GuiResult:
    classification: str
    status: str
    job_id: str = ""
    hostname: str = ""
    private_ip: str = ""
    display: str = ""
    vnc_port: int | None = None
    novnc_port: int | None = None
    detail: str = ""
    logs: dict[str, str] = field(default_factory=dict)


def parse_resolution(value: str) -> str:
    if not isinstance(value, str) or not _RESOLUTION.fullmatch(value):
        raise ValueError(f"resolution must look like 1920x1080x24, got {value!r}")
    return value


def gui_cache_paths(cache_dir: str) -> GuiCachePaths:
    root = cache_dir.rstrip("/") + "/gui"
    return GuiCachePaths(
        root=root,
        x11vnc=f"{root}/x11vnc",
        novnc=f"{root}/noVNC-{NOVNC_VERSION}",
        version_file=f"{root}/VERSION",
    )


def gui_result_path(remote_run_dir: str) -> str:
    return remote_run_dir.rstrip("/") + "/logs/gui/result.json"


def select_free_display(
    occupied: set[int],
    *,
    low: int = DISPLAY_MIN,
    high: int = DISPLAY_MAX,
) -> int | None:
    """First X display number in ``low..high`` that is not already taken."""
    for number in range(low, high + 1):
        if number not in occupied:
            return number
    return None


def display_is_taken(*, sockets: set[str], locks: set[str], number: int) -> bool:
    """Collision rule used before starting Xvfb: socket ``X<n>`` or lock file."""
    return f"X{number}" in sockets or f".X{number}-lock" in locks


def select_free_port(listeners: set[int], *, low: int, high: int) -> int | None:
    """First TCP port in ``low..high`` that is not already listening."""
    for port in range(low, high + 1):
        if port not in listeners:
            return port
    return None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)  # type: ignore[arg-type]


def parse_gui_result(text: str) -> GuiResult | None:
    raw = (text or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    classification = str(data.get("classification") or "")
    if classification not in GUI_CLASSIFICATIONS:
        return None
    logs_raw = data.get("logs") or {}
    logs = {str(k): str(v) for k, v in logs_raw.items()} if isinstance(logs_raw, dict) else {}
    return GuiResult(
        classification=classification,
        status=str(data.get("status") or ""),
        job_id=str(data.get("job_id") or ""),
        hostname=str(data.get("hostname") or ""),
        private_ip=str(data.get("private_ip") or ""),
        display=str(data.get("display") or ""),
        vnc_port=_optional_int(data.get("vnc_port")),
        novnc_port=_optional_int(data.get("novnc_port")),
        detail=str(data.get("detail") or ""),
        logs=logs,
    )


def classification_from_log(text: str) -> str | None:
    found: str | None = None
    for line in (text or "").splitlines():
        if line.startswith("GUI_CLASSIFICATION="):
            value = line.split("=", 1)[1].strip()
            if value in GUI_CLASSIFICATIONS:
                found = value
    return found


def isaac_gui_log_ok(text: str, *, gpu_name: str) -> bool:
    """True when Isaac stayed on the allocated GPU with Vulkan/RTX, not llvmpipe."""
    if "app ready" not in text:
        return False
    if gpu_name and gpu_name not in text:
        return False
    lowered = text.lower()
    if "llvmpipe" in lowered:
        return False
    if "vulkan" not in lowered:
        return False
    if "rtx" not in lowered and "resourcemanager" not in lowered:
        return False
    return True


def format_tunnel_command(*, ssh_alias: str, private_ip: str, port: int) -> str:
    """SSH local forward. Authentication stays in the user's SSH config."""
    return (
        "ssh \\\n"
        "  -o ExitOnForwardFailure=yes \\\n"
        "  -N \\\n"
        f"  -L {port}:{private_ip}:{port} \\\n"
        f"  {ssh_alias}"
    )


def format_browser_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/vnc.html?autoconnect=1&resize=remote"


def format_gui_ready(result: GuiResult, *, ssh_alias: str) -> str:
    if result.novnc_port is None or result.vnc_port is None:
        raise ValueError("GUI_READY result is missing ports")
    tunnel = format_tunnel_command(
        ssh_alias=ssh_alias,
        private_ip=result.private_ip,
        port=result.novnc_port,
    )
    url = format_browser_url(result.novnc_port)
    return (
        "GUI READY\n"
        "\n"
        f"Job:       {result.job_id}\n"
        f"Node:      {result.hostname}\n"
        f"IP:        {result.private_ip}\n"
        f"Display:   {result.display}\n"
        f"VNC:       localhost:{result.vnc_port}\n"
        f"noVNC:     {result.private_ip}:{result.novnc_port}\n"
        "\n"
        f"{tunnel}\n"
        "\n"
        f"{url}\n"
    )


def format_gui_failure(result: GuiResult) -> str:
    key = _FAILURE_LOG.get(result.classification, "isaac")
    log = result.logs.get(key, "")
    detail = result.detail or result.classification
    lines = [result.classification, detail]
    if log:
        lines.append(f"Log: {log}")
    return "\n".join(lines)


def format_saved_gui_connection(record: JobRecord | None, *, ssh_alias: str) -> str | None:
    if record is None or not record.gui:
        return None
    if (
        record.gui_status == "ready"
        and record.novnc_port
        and record.vnc_port
        and record.private_ip
    ):
        result = GuiResult(
            classification="GUI_READY",
            status="ready",
            job_id=record.job_id,
            hostname=record.hostname or "",
            private_ip=record.private_ip,
            display=record.display or "",
            vnc_port=record.vnc_port,
            novnc_port=record.novnc_port,
        )
        return format_gui_ready(result, ssh_alias=ssh_alias).rstrip()
    return f"GUI status: {record.gui_status or 'submitted'}"


def render_gui_bootstrap_script(cache_dir: str) -> str:
    """Idempotent user-space install of x11vnc and noVNC into the cache."""
    paths = gui_cache_paths(cache_dir)
    root = shlex.quote(paths.root)
    prefix = shlex.quote(paths.x11vnc)
    novnc = shlex.quote(paths.novnc)
    version_file = shlex.quote(paths.version_file)
    version_body = shlex.quote(paths.version_text)
    return f"""#!/usr/bin/env bash
# Generated by vector-lab. Installs the GUI runtime under the cluster cache.
# Reused by later GUI jobs. Does not use apt, sudo, or the Isaac image.
set -euo pipefail

GUI_ROOT={root}
PREFIX={prefix}
NOVNC={novnc}
VERSION_FILE={version_file}
VERSION_TEXT={version_body}

runtime_ready() {{
    [ -x "$PREFIX/bin/x11vnc" ] && [ -f "$NOVNC/vnc.html" ] && [ -f "$VERSION_FILE" ] || return 1
    [ "$(cat "$VERSION_FILE")" = "$VERSION_TEXT" ]
}}

mkdir -p "$GUI_ROOT"
if runtime_ready; then
    echo GUI_RUNTIME_REUSED
    exit 0
fi

LOCK="$GUI_ROOT/.bootstrap.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "waiting for another GUI runtime bootstrap"
    for _ in $(seq 1 180); do
        if runtime_ready; then
            echo GUI_RUNTIME_REUSED
            exit 0
        fi
        sleep 5
    done
    echo "GUI runtime bootstrap did not finish" >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

echo GUI_RUNTIME_BOOTSTRAP_START
SRC="$GUI_ROOT/src"
rm -rf "$SRC" "$PREFIX" "$NOVNC"
mkdir -p "$SRC"
cd "$SRC"

curl -fsSL -o libvnc.tar.gz {shlex.quote(LIBVNC_URL)}
tar -xzf libvnc.tar.gz
curl -fsSL -o x11vnc.tar.gz {shlex.quote(X11VNC_URL)}
tar -xzf x11vnc.tar.gz
curl -fsSL -o novnc.tar.gz {shlex.quote(NOVNC_URL)}
tar -xzf novnc.tar.gz

cmake -S "libvncserver-LibVNCServer-{LIBVNC_VERSION}" -B "$SRC/libvnc-build" \\
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DBUILD_SHARED_LIBS=ON \\
    -DCMAKE_INSTALL_LIBDIR=lib \\
    -DWITH_GNUTLS=OFF \\
    -DWITH_SYSTEMD=OFF
cmake --build "$SRC/libvnc-build" -j 4
cmake --install "$SRC/libvnc-build"

cd "x11vnc-{X11VNC_VERSION}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${{PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}}"
export LDFLAGS="-Wl,-rpath,$PREFIX/lib"
if [ ! -x ./configure ]; then
    autoreconf -fi
fi
./configure --prefix="$PREFIX" --with-x
make -j 4
make install

rm -rf "$NOVNC"
mv "$SRC/noVNC-{NOVNC_VERSION}" "$NOVNC"
test -x "$PREFIX/bin/x11vnc"
test -f "$NOVNC/vnc.html"
printf '%s' "$VERSION_TEXT" > "$VERSION_FILE"
rm -rf "$SRC"
echo GUI_RUNTIME_BOOTSTRAPPED
"""


def render_gui_runner_script(
    *,
    command: str,
    bind_block: str,
    exec_flags: str,
    launch: GuiLaunch | None = None,
) -> str:
    """Compute-node script: CUDA gate, Xvfb, Isaac GUI, localhost x11vnc, noVNC."""
    launch = launch or GuiLaunch()
    flags = exec_flags.strip()
    sif_line = (
        f"    {flags} $TMPDIR/$CONTAINER.sif \\"
        if flags
        else "    $TMPDIR/$CONTAINER.sif \\"
    )
    template = r"""#!/usr/bin/env bash
# Generated by vector-lab GUI mode.
# Xvfb -> Isaac GUI (isaaclab.python.kit, no livestream) -> localhost x11vnc -> noVNC.
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR/env.cluster"

RUN_DIR="$1"
CONTAINER="$2"
EXTRA_QUOTED=""
if [ "$#" -ge 3 ]; then
    EXTRA_QUOTED=$(printf ' %q' "${@:3}")
fi

GUI_LOG_DIR="$RUN_DIR/logs/gui"
mkdir -p "$GUI_LOG_DIR"
touch "$GUI_LOG_DIR/isaac.log" "$GUI_LOG_DIR/xvfb.log" "$GUI_LOG_DIR/x11vnc.log" \
    "$GUI_LOG_DIR/websockify.log" "$GUI_LOG_DIR/cuda.log"

XVFB_PID=""
ISAAC_PID=""
VNC_PID=""
WS_PID=""
DISPLAY_NUM=""
VNC_PORT=""
NOVNC_PORT=""
PRIVATE_IP=""
GPU_NAME=""
cleaned=0

write_result() {
    local classification="$1"
    local status="$2"
    local detail="$3"
    local vnc_json="null"
    local novnc_json="null"
    if [ -n "$VNC_PORT" ]; then
        vnc_json="$VNC_PORT"
    fi
    if [ -n "$NOVNC_PORT" ]; then
        novnc_json="$NOVNC_PORT"
    fi
    cat > "$GUI_LOG_DIR/result.json" <<EOF
{
  "classification": "$classification",
  "status": "$status",
  "job_id": "${SLURM_JOB_ID:-}",
  "hostname": "$(hostname)",
  "private_ip": "$PRIVATE_IP",
  "display": "${DISPLAY:-}",
  "vnc_port": $vnc_json,
  "novnc_port": $novnc_json,
  "detail": "$detail",
  "logs": {
    "isaac": "$GUI_LOG_DIR/isaac.log",
    "xvfb": "$GUI_LOG_DIR/xvfb.log",
    "x11vnc": "$GUI_LOG_DIR/x11vnc.log",
    "websockify": "$GUI_LOG_DIR/websockify.log",
    "cuda": "$GUI_LOG_DIR/cuda.log"
  }
}
EOF
    echo "GUI_CLASSIFICATION=$classification"
}

fail() {
    write_result "$1" "failed" "$2"
    exit "$3"
}

cleanup() {
    if [ "$cleaned" = 1 ]; then
        return 0
    fi
    cleaned=1
    trap - EXIT INT TERM
    for pid in "$ISAAC_PID" "$VNC_PID" "$WS_PID" "$XVFB_PID"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "$ISAAC_PID" "$VNC_PID" "$WS_PID" "$XVFB_PID"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    if [ -n "$DISPLAY_NUM" ]; then
        rm -f "/tmp/.X${DISPLAY_NUM}-lock"
    fi
    if [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR/docker-isaac-sim" ]; then
        rsync -a "$TMPDIR/docker-isaac-sim" "$CLUSTER_ISAAC_SIM_CACHE_DIR/.." || true
    fi
    if [ "${REMOVE_CODE_COPY_AFTER_JOB:-false}" = "true" ]; then
        rm -rf "$RUN_DIR"
    fi
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

resolve_tool() {
    local name="$1"
    local path
    path="$(command -v "$name" 2>/dev/null || true)"
    if [ -n "$path" ]; then
        printf '%s\n' "$path"
        return 0
    fi
    local candidate="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/$name"
    if [ -x "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    return 1
}

port_busy() {
    local port="$1"
    ss -lnt 2>/dev/null | awk 'NR>1 {print $4}' | grep -Eq ":${port}$"
}

localhost_port() {
    local port="$1"
    ss -lnt 2>/dev/null | awk 'NR>1 {print $4}' | grep -Eq "^(127\\.0\\.0\\.1|\\[::1\\]):${port}$"
}

display_busy() {
    local n="$1"
    [ -S "/tmp/.X11-unix/X${n}" ] || [ -e "/tmp/.X${n}-lock" ]
}

resolve_private_ip() {
    local ip=""
    ip="$(getent hosts "$(hostname)" 2>/dev/null | awk '{print $1}' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)"
    if [ -z "$ip" ]; then
        ip="$(hostname -I 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+\./) {print $i; exit}}' || true)"
    fi
    printf '%s' "$ip"
}

echo "GUI_STAGE=bootstrap"
bash "$SCRIPT_DIR/gui_bootstrap.sh" || fail VNC_FAILED "GUI runtime bootstrap failed" 80

setup_directories() {
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
        fi
    done
}

echo "GUI_STAGE=extract"
setup_directories
cp -r "$CLUSTER_ISAAC_SIM_CACHE_DIR" "$TMPDIR"
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"
cp -r "$RUN_DIR" "$TMPDIR"
dir_name=$(basename "$RUN_DIR")
tar -xf "$CLUSTER_SIF_PATH/$CONTAINER.tar" -C "$TMPDIR" || fail ISAAC_GUI_FAILED "container extract failed" 79

cat > "$GUI_LOG_DIR/cuda_probe.py" <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if not torch.cuda.is_available():
    print("CUDA_MICROTEST FAIL not available")
    raise SystemExit(2)
print("GPU_NAME=" + torch.cuda.get_device_name(0))
x = torch.ones(1, device="cuda")
print("tensor", x)
print("CUDA_MICROTEST PASS")
PY

echo "GUI_STAGE=cuda"
if ! @@COMMAND@@ exec \
@@BIND@@
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw \
    -B "$GUI_LOG_DIR:$GUI_LOG_DIR:rw" \
@@SIF@@
    bash -lc "export LD_LIBRARY_PATH=/.singularity.d/libs:\${LD_LIBRARY_PATH:-} && /isaac-sim/python.sh \"$GUI_LOG_DIR/cuda_probe.py\"" \
    >"$GUI_LOG_DIR/cuda.log" 2>&1; then
    fail GPU_UNHEALTHY "CUDA health gate command failed" 77
fi
if ! grep -q "CUDA_MICROTEST PASS" "$GUI_LOG_DIR/cuda.log"; then
    fail GPU_UNHEALTHY "CUDA_MICROTEST PASS missing" 77
fi
GPU_NAME="$(sed -n 's/^GPU_NAME=//p' "$GUI_LOG_DIR/cuda.log" | head -1 || true)"
if [ -z "$GPU_NAME" ]; then
    fail GPU_UNHEALTHY "allocated GPU name missing from CUDA gate" 77
fi

echo "GUI_STAGE=xvfb"
XVFB_BIN="$(resolve_tool Xvfb || true)"
XDPYINFO_BIN="$(resolve_tool xdpyinfo || true)"
if [ -z "$XVFB_BIN" ] || [ -z "$XDPYINFO_BIN" ]; then
    fail XVFB_FAILED "Xvfb or xdpyinfo is not on PATH" 78
fi
mkdir -p /tmp/.X11-unix
for n in $(seq @@DISPLAY_MIN@@ @@DISPLAY_MAX@@); do
    if display_busy "$n"; then
        continue
    fi
    "$XVFB_BIN" ":$n" -screen 0 @@RESOLUTION@@ +extension GLX +render -noreset -ac \
        >"$GUI_LOG_DIR/xvfb.log" 2>&1 &
    XVFB_PID=$!
    ready=0
    for _ in $(seq 1 15); do
        if ! kill -0 "$XVFB_PID" 2>/dev/null; then
            break
        fi
        if "$XDPYINFO_BIN" -display ":$n" 2>/dev/null | grep -q "@@PIXELS@@"; then
            DISPLAY_NUM=$n
            export DISPLAY=":$n"
            ready=1
            break
        fi
        sleep 1
    done
    if [ "$ready" = 1 ]; then
        break
    fi
    kill "$XVFB_PID" 2>/dev/null || true
    wait "$XVFB_PID" 2>/dev/null || true
    XVFB_PID=""
done
if [ -z "$DISPLAY_NUM" ]; then
    fail XVFB_FAILED "no free display in :@@DISPLAY_MIN@@-:@@DISPLAY_MAX@@ passed xdpyinfo" 78
fi
echo "GUI_STAGE=xvfb_pass DISPLAY=$DISPLAY"

echo "GUI_STAGE=isaac"
@@COMMAND@@ exec \
@@BIND@@
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw \
    -B /tmp/.X11-unix:/tmp/.X11-unix \
@@SIF@@
    bash -lc "export DISPLAY=$DISPLAY && export HEADLESS=0 && unset LIVESTREAM && export ISAACLAB_PATH=/workspace/isaaclab && cd /workspace/isaaclab && /isaac-sim/python.sh $CLUSTER_PYTHON_EXECUTABLE$EXTRA_QUOTED" \
    >"$GUI_LOG_DIR/isaac.log" 2>&1 &
ISAAC_PID=$!

isaac_graphics_ok() {
    grep -q "app ready" "$GUI_LOG_DIR/isaac.log" || return 1
    grep -F -q "$GPU_NAME" "$GUI_LOG_DIR/isaac.log" || return 1
    grep -qi vulkan "$GUI_LOG_DIR/isaac.log" || return 1
    grep -Ei -q 'RTX|ResourceManager' "$GUI_LOG_DIR/isaac.log" || return 1
    if grep -qi llvmpipe "$GUI_LOG_DIR/isaac.log"; then
        return 1
    fi
    return 0
}

graphics=0
for _ in $(seq 1 @@ISAAC_POLLS@@); do
    if ! kill -0 "$ISAAC_PID" 2>/dev/null; then
        fail ISAAC_GUI_FAILED "Isaac process exited before the GUI was ready" 79
    fi
    if isaac_graphics_ok; then
        graphics=1
        break
    fi
    sleep @@ISAAC_POLL_SECONDS@@
done
if [ "$graphics" != 1 ]; then
    fail ISAAC_GUI_FAILED "Isaac did not report app ready on $GPU_NAME with Vulkan/RTX" 79
fi
echo "GUI_STAGE=isaac_pass"

echo "GUI_STAGE=vnc"
X11VNC_BIN="$CLUSTER_ISAAC_SIM_CACHE_DIR/gui/x11vnc/bin/x11vnc"
if [ ! -x "$X11VNC_BIN" ]; then
    fail VNC_FAILED "x11vnc is missing from the GUI runtime cache" 80
fi
for port in $(seq @@VNC_MIN@@ @@VNC_MAX@@); do
    if port_busy "$port"; then
        continue
    fi
    LD_LIBRARY_PATH="$CLUSTER_ISAAC_SIM_CACHE_DIR/gui/x11vnc/lib:$CLUSTER_ISAAC_SIM_CACHE_DIR/gui/x11vnc/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$X11VNC_BIN" -display "$DISPLAY" -forever -shared -rfbport "$port" -localhost -noxdamage -nopw \
        >"$GUI_LOG_DIR/x11vnc.log" 2>&1 &
    VNC_PID=$!
    bound=0
    for _ in $(seq 1 20); do
        if localhost_port "$port"; then
            if ss -lnt 2>/dev/null | awk 'NR>1 {print $4}' | grep -Eq "^(0\\.0\\.0\\.0|\\*|\\[::\\]):${port}$"; then
                kill "$VNC_PID" 2>/dev/null || true
                fail VNC_FAILED "x11vnc bound a non-localhost address" 80
            fi
            VNC_PORT=$port
            bound=1
            break
        fi
        if ! kill -0 "$VNC_PID" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done
    if [ "$bound" = 1 ]; then
        break
    fi
    kill "$VNC_PID" 2>/dev/null || true
    wait "$VNC_PID" 2>/dev/null || true
    VNC_PID=""
done
if [ -z "$VNC_PORT" ]; then
    fail VNC_FAILED "x11vnc did not listen on 127.0.0.1" 80
fi
echo "GUI_STAGE=vnc_pass PORT=$VNC_PORT"

echo "GUI_STAGE=novnc"
WEBSOCKIFY_BIN="$(resolve_tool websockify || true)"
CURL_BIN="$(resolve_tool curl || true)"
NOVNC_ROOT="$CLUSTER_ISAAC_SIM_CACHE_DIR/gui/noVNC-@@NOVNC_VERSION@@"
if [ -z "$WEBSOCKIFY_BIN" ]; then
    fail NOVNC_FAILED "websockify is not on PATH" 81
fi
if [ -z "$CURL_BIN" ]; then
    fail NOVNC_FAILED "curl is not on PATH" 81
fi
if [ ! -f "$NOVNC_ROOT/vnc.html" ]; then
    fail NOVNC_FAILED "noVNC web root is missing from the GUI runtime cache" 81
fi
for port in $(seq @@NOVNC_MIN@@ @@NOVNC_MAX@@); do
    if port_busy "$port"; then
        continue
    fi
    "$WEBSOCKIFY_BIN" --web="$NOVNC_ROOT" "0.0.0.0:$port" "127.0.0.1:$VNC_PORT" \
        >"$GUI_LOG_DIR/websockify.log" 2>&1 &
    WS_PID=$!
    bound=0
    for _ in $(seq 1 20); do
        code="$("$CURL_BIN" -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${port}/vnc.html" 2>/dev/null || true)"
        if [ "$code" = "200" ]; then
            NOVNC_PORT=$port
            bound=1
            break
        fi
        if ! kill -0 "$WS_PID" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done
    if [ "$bound" = 1 ]; then
        break
    fi
    kill "$WS_PID" 2>/dev/null || true
    wait "$WS_PID" 2>/dev/null || true
    WS_PID=""
done
if [ -z "$NOVNC_PORT" ]; then
    fail NOVNC_FAILED "noVNC did not return HTTP 200 for vnc.html" 81
fi

PRIVATE_IP="$(resolve_private_ip)"
if [ -z "$PRIVATE_IP" ]; then
    fail NOVNC_FAILED "could not resolve the compute node private IP" 81
fi
echo "GUI_STAGE=ready"
write_result GUI_READY ready "Isaac GUI is reachable through noVNC"
wait "$ISAAC_PID" || true
echo "GUI_STAGE=isaac_exit"
"""
    return (
        template.replace("@@COMMAND@@", command)
        .replace("@@BIND@@", bind_block)
        .replace("@@SIF@@", sif_line)
        .replace("@@RESOLUTION@@", launch.resolution)
        .replace("@@PIXELS@@", launch.pixels)
        .replace("@@DISPLAY_MIN@@", str(launch.display_min))
        .replace("@@DISPLAY_MAX@@", str(launch.display_max))
        .replace("@@VNC_MIN@@", str(launch.vnc_min))
        .replace("@@VNC_MAX@@", str(launch.vnc_max))
        .replace("@@NOVNC_MIN@@", str(launch.novnc_min))
        .replace("@@NOVNC_MAX@@", str(launch.novnc_max))
        .replace("@@NOVNC_VERSION@@", NOVNC_VERSION)
        .replace("@@ISAAC_POLLS@@", str(launch.isaac_polls))
        .replace("@@ISAAC_POLL_SECONDS@@", str(launch.isaac_poll_seconds))
    )
