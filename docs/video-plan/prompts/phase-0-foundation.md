# Phase 0 — Foundation, audit & GPU setup

Read `CLAUDE.md`, `docs/video-plan/PLAN.md` and `docs/video-plan/PROGRESS.md` first.
You are working on Phase 0 only.

## Goal
Make the repo a safe place to build the term project: CI actually runs, the
GPU is usable, the baseline is measured, build junk is out of git, and the
working conventions from CLAUDE.md are in place.

## Tasks
1. Create and switch to branch `video-analysis` from `GUI`.
2. **CI:** `.github/workflows/test.yml` only triggers on `main`, but this fork
   uses `master`, `GUI`, `development`, `video-analysis`. Make it trigger on
   push/PR to those branches (and `workflow_dispatch`). Update the action
   versions (`actions/checkout`, `actions/setup-python`) to current majors.
   Install CPU-only torch in CI (same version as `requirements.txt`) if the
   default wheel is too large/slow. Cache pip. Keep `pip check`, pytest and
   the wheel build steps. Point the README CI badge at `ply971/deeptune`
   instead of upstream.
3. **Repo hygiene:** build output is committed in four folders:
   `docs/_build/`, `docs/build/`, `docs/source/_build/`, `docs/source/build/`
   (424 files incl. demo `.mp4`s). Remove them from git with
   `git rm -r --cached` (do NOT delete docs sources) and add them to
   `.gitignore`. Already checked: the docs link to the demo videos in
   `docs/source/_static/`, which stays tracked — re-verify before removing, and
   if anything does link into a build folder, stop and ask.
4. **GPU setup (G11):** the machine has an RTX 5060 Laptop GPU (Blackwell,
   8 GB) but `.venv` has `torch 2.14.0+cpu`. Install the CUDA build of the
   **same** torch/torchvision versions (a CUDA ≥ 12.8 wheel index, required for
   Blackwell) into `.venv`. Ask before installing. Verify:
   `torch.cuda.is_available()`, device name, a matmul and a tiny conv
   forward/backward on GPU, and that `pip check` stays clean. Do not change
   `requirements.txt` pins in a way that breaks CPU-only CI; document the
   local GPU install command in `CLAUDE.md` and `README`.
5. **One-command check:** add `scripts/check.ps1` (and `scripts/check.sh` for
   CI/Linux) that runs: pip check → full pytest with a junit xml and
   `--durations=10` → the two video smoke scripts (skippable with `-Quick`) and
   prints a one-line summary including the device (CPU/GPU name). Exit
   non-zero on any failure.
6. **Baseline:** run the full suite and both video smoke checks. Record in
   PROGRESS.md: pass/fail counts, warnings, total runtime, slowest 10 tests,
   Python/torch/torchvision/CUDA versions, GPU name. If tests fail before any
   change, record them as the baseline — do not fix them in this phase.
7. **Audit note:** read the video code paths (`datasets/video_datasets.py`,
   `handlers/video_manifest.py`, `src/video/*`, `trainers/video/train.py`,
   `evaluators/video/evaluate.py`, `embed/video/embed.py`, `inference/video.py`,
   `utils.build_video_model`, video parts of `desktop/`). Confirm or correct
   gaps G1–G13 in PLAN.md section 1 and add anything missed under "Notes for
   later phases". Also check: does the code pick `cuda` automatically when
   available, or is the device hard-coded? Do not fix them now.

## Constraints
- No behaviour changes to the library in this phase (CI/hygiene/scripts/env only).
- Do not touch tests except to add markers if needed for `-Quick`.
- The full suite must still pass on a CPU-only machine (CI).

## Definition of done
- CI workflow file valid (run `act` if available, otherwise validate YAML and
  explain), triggers on the right branches.
- Build output untracked and ignored; nothing user-facing broken.
- GPU visible to torch in `.venv`; verification output pasted.
- `scripts/check.ps1` works and its output is pasted in your summary.
- PROGRESS.md has the baseline and the audit notes. One or more commits.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Before saying done: run `scripts/check.ps1`, paste the real
summary, show `git status` is clean, and list every file you changed. If
anything fails twice, stop and report instead of working around it.
