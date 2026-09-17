# vector-lab

Vector Lab is the deployment and bootstrap tool for reproducible Isaac Lab
workloads on Vector's cluster environment (SSH + SLURM + Apptainer).

You do **not** edit Isaac Lab’s `docker/cluster/` scripts. This CLI generates the
runtime wrappers, builds and caches the container when needed, syncs source, and
submits jobs. Local tool state lives in `.vector-lab/` (optional user config:
`~/.config/vector-lab/`).

Ordinary Python/source changes do **not** rebuild or re-upload the container.

## Quick start

```bash
git clone <vector-lab-repo>
cd vector-lab

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# configure SSH alias / complete MFA as instructed below

vector-lab onboard bonecho
vector-lab doctor
vector-lab deploy
vector-lab smoke-test --video
```

v1 targets **Ubuntu-like Linux** and the **Bonecho** cluster.

## First-time prerequisites

On the laptop:

- Python 3.10+
- `git`, `ssh`, `rsync`
- Docker on PATH (`/snap/bin/docker` is fine)
- Apptainer (local conversion)
- Tens of GiB free disk for the first container conversion

Check what is missing (this **never** runs `sudo`):

```bash
vector-lab bootstrap
```

If Docker permission is denied:

```bash
sudo usermod -aG docker "$USER"
# log out and back in
```

Keep `docker` as a **supplementary** group. Do not use `newgrp docker`,
`chmod 666` on the Docker socket, or `sudo` for the whole workflow.

## Bonecho SSH setup

Add a `Host bonecho` entry to `~/.ssh/config` **yourself**. vector-lab will
not write that file and will not bypass MFA.

```
Host bonecho
    HostName <login-node>
    User <your-user>
    ControlMaster auto
    ControlPersist 10m
    ControlPath ~/.ssh/cm-%C
```

Then complete MFA once:

```bash
ssh bonecho
# or
vector-lab auth bonecho
```

If later commands fail with BatchMode / permission denied, the multiplexed
session expired — run `vector-lab auth bonecho` again.

## Onboard

```bash
vector-lab onboard bonecho
```

If Isaac Lab is already cloned:

```bash
vector-lab onboard bonecho --isaaclab /path/to/IsaacLab
```

Onboard checks local tools, finds or clones a **pinned** Isaac Lab commit,
authenticates SSH, writes the Bonecho profile, generates wrappers, and runs
doctor. A second run should mostly report READY / SKIPPED.

```bash
vector-lab doctor
```

You want `Overall: READY`. Deployment lines may still say NEEDS BUILD until
`deploy`.

## Deploy

```bash
vector-lab deploy --plan    # no multi-GB work
vector-lab deploy
```

First deploy can take a long time (Docker build, conversion, ~20 GiB upload).
Later deploys should skip cached steps:

```
Docker build        SKIPPED
Conversion          cached
Upload              SKIPPED
```

## Run

```bash
vector-lab run --task Isaac-Cartpole-v0 --video --headless
```

Source is rsynced to a new scratch run directory. The job uses the existing
remote container.

## Monitor

```bash
vector-lab status
vector-lab status <job-id>
vector-lab logs <job-id>
vector-lab logs <job-id> --follow
vector-lab cancel <job-id>
```

## Videos

```bash
vector-lab videos
vector-lab pull-video --latest
```

## Caching behavior

| Layer | Cached when | Rebuilds when |
| --- | --- | --- |
| Docker image | Dockerfile / compose / env / dependency manifests unchanged | Those inputs change, or the image is missing |
| Apptainer artifact | Same image digest + converter | Docker image changed |
| Remote upload | Remote SHA-256 matches local tar | Artifact bytes differ |
| Runtime source | rsynced per job | Everyday `source/` and `scripts/` edits |

Interrupted uploads resume. Runtime Python does **not** trigger Docker rebuild.

## Troubleshooting

| Problem | Next step |
| --- | --- |
| Docker not installed | Install Docker Engine; `/snap/bin/docker` is fine. |
| Docker permission denied | `sudo usermod -aG docker $USER`, then logout/login. Not `newgrp`. |
| Apptainer missing | Install from https://apptainer.org/docs/admin/main/installation.html |
| SSH alias missing | Add `Host bonecho` to `~/.ssh/config` (see above). Tool will not write it. |
| MFA / session expired | `vector-lab auth bonecho` |
| Isaac Lab missing | `vector-lab onboard bonecho` (pinned clone) or `--isaaclab PATH` |
| Container not on cluster | `vector-lab deploy` |
| SLURM / partition / Apptainer | `vector-lab doctor` |
| Upload interrupted | Re-run `vector-lab deploy` |

## Architecture

`vector-lab` is a laptop CLI: **ClusterProfile** (Bonecho adapter) +
**ImageProfile** + **JobProfile**. Bonecho details (login-shell SLURM, Apptainer
module, `--writable-tmpfs`, persistent cache/log binds) are applied
automatically in generated wrappers. You normally do not interact with them.

Lower-level commands (`setup`, `build`, `push`, `shell`) remain available.
Most teammates only need onboard / doctor / deploy / run / smoke-test.

Historical acceptance evidence (not install requirements): see
[docs/KNOWN_GOOD.md](docs/KNOWN_GOOD.md). Fresh-laptop checklist:
[docs/FRESH_MACHINE_TEST.md](docs/FRESH_MACHINE_TEST.md).
