# Fresh-machine acceptance test

Assume:

- Ubuntu-like laptop
- no existing `vector-lab` state
- no Isaac Lab clone
- a working Bonecho account

Do **not** treat this as a script to run unattended. Several steps are
intentionally manual.

## Procedure

1. Clone vector-lab and `cd` into it.
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -e .`
4. `vector-lab bootstrap` (and `bootstrap --install` only to print commands)
5. Configure SSH alias in `~/.ssh/config` if needed (tool will not write it)
6. Authenticate / MFA: `ssh bonecho` or `vector-lab auth bonecho`
7. `vector-lab onboard bonecho`
8. `vector-lab doctor` — expect `Overall: READY` (deploy lines may still need build)
9. `vector-lab deploy` — first time is expensive
10. `vector-lab smoke-test --video`
11. Verify job `COMPLETED` / ExitCode `0:0`
12. Verify an MP4 exists (`vector-lab videos` / `pull-video --latest`)
13. Re-run `vector-lab onboard bonecho` and `vector-lab deploy --plan` (or deploy) and confirm SKIPPED / READY / checksum match (idempotency)

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
