# vector-lab

Vector Lab builds and deploys reproducible Isaac Lab workloads to SLURM
clusters over SSH and Apptainer. It supports both **Bonecho** and
**Killarney**. Cluster behavior and paths are discovered during onboarding.

The tool generates its own runtime wrappers; do **not** edit Isaac Lab's
`docker/cluster/` scripts. Ordinary changes under Isaac Lab's `source/` or
`scripts/` directories are synchronized per job and do not rebuild the image.

This guide uses `<cluster>` where the value is either `bonecho` or
`killarney`. Operational examples always pass `--cluster` so it is clear where
the command runs.

## 1. Build the package

Clone the repository:

```bash
git clone git@github.com:VectorInstitute/vector-lab.git
cd vector-lab
```

Vector Lab does not require a virtual environment or `pip install`. Source the
repository setup script once:

```bash
source ./activate.sh
vector-lab --version
```

To make the command available in every Bash session, add the **absolute** path
to `~/.bashrc`:

```bash
echo 'source /absolute/path/to/vector-lab/activate.sh' >> ~/.bashrc
source ~/.bashrc
```

The launcher uses system `python3` and expects Python 3.10+ and PyYAML. On
Ubuntu, install PyYAML only if the launcher reports it missing:

```bash
sudo apt install python3-yaml
```

## 2. Local setup

The laptop needs:

- Python 3.10+, PyYAML, `git`, `ssh`, and `rsync`
- Docker (`/snap/bin/docker` is supported)
- Apptainer for local image conversion
- Approximately 40 GiB of free disk for the first conversion

Inspect local prerequisites. This command reports required actions but never
runs `sudo`:

```bash
vector-lab bootstrap
```

If Docker is installed but access is denied:

```bash
sudo usermod -aG docker "$USER"
# Log out and back in.
```

Keep `docker` as a supplementary group. Do not use `newgrp docker`, run the
whole workflow with `sudo`, or change the Docker socket to mode `666`.

### Configure SSH aliases

Vector Lab does not modify `~/.ssh/config` or bypass MFA. Add entries for each
cluster you use:

```sshconfig
Host bonecho
    HostName bonecho.vectorinstitute.ai
    User <your-user>
    ControlMaster auto
    ControlPersist 10m
    ControlPath ~/.ssh/cm-%C

Host killarney
    HostName killarney.alliancecan.ca
    User <your-user>
    ControlMaster auto
    ControlPersist 10m
    ControlPath ~/.ssh/cm-%C
```

Authenticate to each cluster once to establish its multiplexed SSH session:

```bash
vector-lab auth bonecho
vector-lab auth killarney
```

If a later command reports BatchMode or authentication failure, rerun
`vector-lab auth <cluster>`.

## 3. One-time onboarding and deployment on each cluster

Onboarding discovers the selected cluster's login-shell requirements, scratch
path, SLURM GPU partitions, and Apptainer module. It then writes a profile and
generated wrappers under `.vector-lab/`.

If Isaac Lab is already cloned:

```bash
vector-lab onboard bonecho --isaaclab /absolute/path/to/IsaacLab
vector-lab onboard killarney --isaaclab /absolute/path/to/IsaacLab
```

Without `--isaaclab`, onboarding finds an existing checkout or offers the
pinned clone:

```bash
vector-lab onboard bonecho
vector-lab onboard killarney
```

Verify each profile:

```bash
vector-lab doctor --cluster bonecho
vector-lab doctor --cluster killarney
```

The expected result is `Overall: READY`.

Deploy the container separately to every cluster you will use. `--plan` does
not build, convert, or upload anything:

```bash
# Bonecho
vector-lab deploy --cluster bonecho --plan
vector-lab deploy --cluster bonecho

# Killarney
vector-lab deploy --cluster killarney --plan
vector-lab deploy --cluster killarney
```

The first deployment builds the Docker image, converts it to an Apptainer
artifact, and uploads approximately 20 GiB to the selected cluster. The local
build and conversion are reusable, but the remote artifact must exist on each
cluster. Interrupted uploads can be resumed by rerunning the same deploy
command.

Later deployments should normally report:

```text
Docker build        SKIPPED
Conversion          cached
Upload              SKIPPED
```

Optionally run the acceptance workload on each cluster:

```bash
vector-lab smoke-test --cluster bonecho --video
vector-lab smoke-test --cluster killarney --video
```

## 4. Run jobs

Always pass `--cluster` to avoid submitting to the wrong active profile.

### Submit

```bash
# Bonecho
vector-lab run --cluster bonecho \
  --task Isaac-Cartpole-v0 --video --headless

# Killarney
vector-lab run --cluster killarney \
  --task Isaac-Cartpole-v0 --video --headless
```

Source is synchronized into a new scratch run directory on the selected
cluster. The job uses the container deployed in step 3.

### Interactive GUI

`vector-lab gui` submits an Isaac Sim GUI session and waits until it is
reachable in a browser. On the compute node the path is:

```text
Isaac Sim GUI → Xvfb → x11vnc on 127.0.0.1 → noVNC/websockify
```

Vector Lab then prints an SSH local forward and a localhost URL. It does not
open a browser or keep the tunnel process running. This does not use Isaac Sim
WebRTC or livestream.

```bash
vector-lab gui --cluster bonecho --time 3h
```

Leave the printed `ssh -N -L` command running, then open the printed
`http://127.0.0.1:<port>/vnc.html?autoconnect=1&resize=remote` URL.

The same job id works with `status`, `logs`, and `cancel`. After the session
is ready, `vector-lab status --cluster <cluster> <job-id>` prints the tunnel
again.

The first GUI run builds x11vnc and noVNC 1.5.0 once under the cluster cache
at `<cache>/gui/`. Later runs reuse that directory. The build does not use
apt or sudo and does not rebuild the Isaac image. Job cleanup does not delete
the cache.

Xvfb uses the first free display in `:90`–`:119` at `1920x1080x24`
(`--resolution` overrides the size). Raw VNC stays on localhost. Only the
noVNC port is reachable from the login node, and only through the SSH tunnel.
`--partition`, `--gpu`, `--cpus`, `--memory`, and `--time` match `run`. The
default experience is `scripts/tutorials/00_sim/create_empty.py`.

The compute node needs `Xvfb`, `xdpyinfo`, `curl`, and `websockify` on `PATH`.
The one-time login-node build also needs `cmake`, a C compiler, and X11
headers. This flow was exercised on Bonecho.

Resource settings can be overridden per job:

```bash
vector-lab run --cluster killarney \
  --task Isaac-Cartpole-v0 \
  --partition gpubase_l40s_b1 \
  --gpu l40s --gpus 1 \
  --cpus 8 --memory 32G --time 1h \
  --headless
```

### Monitor and manage

Use the same cluster for all operations on a job:

```bash
vector-lab status --cluster <cluster>
vector-lab status --cluster <cluster> <job-id>
vector-lab logs --cluster <cluster> <job-id>
vector-lab logs --cluster <cluster> <job-id> --follow
vector-lab cancel --cluster <cluster> <job-id>
```

### Retrieve videos

```bash
vector-lab videos --cluster <cluster>
vector-lab pull-video --cluster <cluster> --latest
```

## Caching behavior

| Layer | Cached when | Rebuilds or transfers when |
| --- | --- | --- |
| Docker image | Docker inputs are unchanged | Dockerfile, compose, environment, or dependency inputs change |
| Apptainer artifact | Image digest and converter are unchanged | Docker image changes |
| Remote upload | Selected cluster's checksum matches | Artifact differs or is missing on that cluster |
| Runtime source | Never treated as image input | Synchronized for every job |

## Troubleshooting

| Problem | Next step |
| --- | --- |
| `vector-lab` not found | `source /absolute/path/to/vector-lab/activate.sh` |
| PyYAML missing | `sudo apt install python3-yaml` |
| Docker missing | Install Docker Engine; `/snap/bin/docker` is supported |
| Docker permission denied | Add `$USER` to the `docker` group, then log out and back in |
| SSH alias missing | Add `Host bonecho` or `Host killarney` to `~/.ssh/config` |
| MFA session expired | `vector-lab auth <cluster>` |
| Isaac Lab missing | `vector-lab onboard <cluster> --isaaclab /path/to/IsaacLab` |
| Container missing remotely | `vector-lab deploy --cluster <cluster>` |
| SLURM, scratch, or Apptainer issue | `vector-lab doctor --cluster <cluster>` |
| Upload interrupted | Rerun `vector-lab deploy --cluster <cluster>` |
| GUI startup failed | The error names the component and its log under the run's `logs/gui/` |
| `Xvfb` or `websockify` missing | Those tools must already be on the compute node; GUI mode does not install them with apt |

## Architecture and lower-level commands

Vector Lab separates the user-facing cluster profile name, SSH alias, and
cluster implementation. Discovery allows Bonecho and Killarney to use their
own home paths, scratch paths, modules, and GPU partitions without treating
either cluster's settings as universal defaults.

Most users need only `bootstrap`, `auth`, `onboard`, `doctor`, `deploy`, `run`,
`gui`, and the monitoring commands. Lower-level commands such as `init`,
`setup`, `build`, `push`, and `shell` remain available.

Historical acceptance evidence is in
[docs/KNOWN_GOOD.md](docs/KNOWN_GOOD.md). The clean-machine checklist is in
[docs/FRESH_MACHINE_TEST.md](docs/FRESH_MACHINE_TEST.md).
