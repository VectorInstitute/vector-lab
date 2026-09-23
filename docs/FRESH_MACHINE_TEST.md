# Fresh-machine acceptance test

Assume:

- Ubuntu-like laptop
- no existing `vector-lab` state
- no Isaac Lab clone
- a working Bonecho or Killarney account

Do **not** treat this as a script to run unattended. Several steps are
intentionally manual.

## Procedure

1. Clone vector-lab and `cd` into it.
2. `source ./activate.sh` (no virtual environment or package install).
3. Optionally add `source /absolute/path/to/vector-lab/activate.sh` to `~/.bashrc`.
4. `vector-lab bootstrap` (and `bootstrap --install` only to print commands).
5. Configure the cluster SSH alias in `~/.ssh/config` (tool will not write it).
6. Authenticate / MFA: `vector-lab auth <cluster>`.
7. `vector-lab onboard <cluster>`.
8. `vector-lab doctor --cluster <cluster>` — expect `Overall: READY`.
9. `vector-lab deploy --cluster <cluster>` — first time is expensive.
10. `vector-lab smoke-test --cluster <cluster> --video`.
11. Verify job `COMPLETED` / ExitCode `0:0`
12. Verify an MP4 exists with `vector-lab videos --cluster <cluster>`.
13. Re-run `vector-lab onboard <cluster>` and `vector-lab deploy --cluster <cluster> --plan` and confirm SKIPPED / READY / checksum match.

## Intentionally manual

| Step | Why |
| --- | --- |
| `sudo` package or `usermod -aG docker` | Privilege change; logout/login required |
| Editing `~/.ssh/config` | Tool must not rewrite SSH config |
| MFA / `vector-lab auth` | Keyboard-interactive; no secret storage |
| First `deploy` | Docker build + Apptainer conversion + multi-GB upload |

## Pass criteria

- Onboard is idempotent on a second run
- Deploy cache-hits when inputs are unchanged
- Smoke-test Cartpole completes with `0:0`
- At least one MP4 is listed or pulled
- No edits to Isaac Lab `docker/cluster/` scripts
