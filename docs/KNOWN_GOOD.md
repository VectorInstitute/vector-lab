# Known-good acceptance evidence

This file records a **historical** Bonecho validation run. It is not a
requirement that future machines reproduce the same Docker digest or artifact
hash.

These hashes must not be used as hard deploy gates.

## Cluster

- Cluster: Bonecho
- GPU: NVIDIA A40
- Partition: `a40_b1`
- GRES: `gpu:a40:1`
- Resources: 8 CPU, 32G RAM, 1 hour

## Job

- SLURM job ID: `390208`
- Task: `Isaac-Cartpole-v0` (RSL-RL)
- Iterations: 150
- Mean episode reward: about 4.94–4.95
- Result: `COMPLETED`, ExitCode `0:0`
- Rendering: Vulkan, headless
- MP4 generation: successful (persistent logs bind; pullable locally)

## Container (at acceptance)

- Docker image digest:
  `sha256:0448e7e5b8b13bb9e3f304e115f77353b0dd19976b51d72eb991cd1fb99ecd4d`
- Deployed artifact SHA-256:
  `98ba9b235957a1d843759c4bb7aa55cf56f747e84240ccfd16dcc3e8ea113e7d`

Future teammates should cache against **their** fingerprints and checksums, not
these values.

## Runtime wrapper behavior (at acceptance)

Generated runner used `--nv --containall --writable-tmpfs` with persistent Isaac
Lab cache/log mounts. Login-shell SLURM and `module load apptainer` were applied
by generated wrappers, not by patching Isaac Lab `docker/cluster/` scripts.

## Isaac Lab pin (tested checkout)

- Remote: `https://github.com/isaac-sim/IsaacLab.git`
- Commit: `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`

Onboard may warn if a local clone is on a different commit. It will not reset
user work.

## Paths mentioned during acceptance (historical)

These are historical layout notes from job 390208. Do not copy them verbatim.
Use `/scratch/<user>/isaaclab-containers` and a unique `isaaclab_<timestamp>`
run directory for new work.

- Remote artifact (then): `/scratch/<user>/isaaclab-containers/isaac-lab-base.tar`
- Run directory (then): `/scratch/<user>/isaaclab_20260917_152801`
- Videos were under that user’s persistent `isaaclab/logs` tree
