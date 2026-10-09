# Phase 0 — Foundation & audit

Read `CLAUDE.md`, `docs/video-plan/PLAN.md` and `docs/video-plan/PROGRESS.md` first.
You are working on Phase 0 only.

## Goal
Make the repo a safe place to build the term project: CI actually runs, the
baseline is measured, build junk is out of git, and the working conventions
from CLAUDE.md are in place.

## Tasks
1. Create and switch to branch `video-analysis` from `GUI`.
2. **CI:** `.github/workflows/test.yml` only triggers on `main`, but this fork
   uses `master`, `GUI`, `development`, `video-analysis`. Make it trigger on
   push/PR to those branches (and `workflow_dispatch`). Update the action
   versions (`actions/checkout`, `actions/setup-python`) to current majors.
   Install CPU-only torch in CI if the default wheel is too large/slow.
   Cache pip. Keep `pip check`, pytest and the wheel build steps.
   Point the README CI badge at `ply971/deeptune` instead of upstream.
3. **Repo hygiene:** `docs/_build/`, `docs/build/`, `docs/source/_build/`
   (with demo `.mp4`s) are committed build output. Remove them from git with
   `git rm -r --cached` (do NOT delete docs sources), and add them to
   `.gitignore`. Before removing, check whether any docs page or README links
   to files inside them (e.g. demo videos); if so, list them in PROGRESS.md and
   ask me what to do instead of breaking links.
4. **One-command check:** add `scripts/check.ps1` (and `scripts/check.sh` for
   CI/Linux) that runs: pip check → full pytest with a junit xml →
   the two video smoke scripts (skippable with a `-Quick` flag) and prints a
   one-line summary. Exit non-zero on any failure.
5. **Baseline:** run the full suite and both video smoke checks. Record in
   PROGRESS.md: pass/fail counts, warnings, total runtime, slowest 10 tests
   (`--durations=10`), Python/torch/torchvision versions, GPU or CPU.
6. **Audit note:** read the video code paths (`datasets/video_datasets.py`,
   `handlers/video_manifest.py`, `src/video/*`, `trainers/video/train.py`,
   `evaluators/video/evaluate.py`, `embed/video/embed.py`, `inference/video.py`,
   `utils.build_video_model`, video parts of `desktop/`). Confirm or correct
   the gap list G1–G10 in PLAN.md section 1 and add anything I missed under
   "Notes for later phases". Do not fix them now.

## Constraints
- No behaviour changes to the library in this phase (CI/hygiene/scripts only).
- Do not touch tests except to add markers if needed for `-Quick`.

## Definition of done
- CI workflow file valid (run `act` if available, otherwise validate YAML and
  explain), triggers on the right branches.
- Build output untracked and ignored; nothing user-facing broken.
- `scripts/check.ps1` works and its output is pasted in your summary.
- PROGRESS.md has the baseline and the audit notes. One or more commits.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Before saying done: run `scripts/check.ps1`, paste the real
summary, show `git status` is clean, and list every file you changed. If
anything fails twice, stop and report instead of working around it.
