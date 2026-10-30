# AGENTS.md: instructions for any coding agent (Codex, etc.)

This repo's working rules live in **`CLAUDE.md`**. Read it first and follow it exactly. It applies to you too.

## Read before touching code
1. `CLAUDE.md`: rules, commands, architecture constraints.
2. `docs/video-plan/PLAN.md`, especially §4b: scope, licences, accepted proposals, professional-quality rules.
3. `docs/video-plan/PROGRESS.md`: what is done; the newest entry is the handoff.
4. `.claude/board/QUEUE.md`: **the exact next steps**. Work only on the first open item.
5. `.claude/board/REVIEWS.md` and `docs/video-plan/reports/*.md`: open review items and what each task built.

## Coordination
- While you work, create `.claude/board/LOCK` containing your name and the start time. Claude Code's scheduled
  check-ins see it and only report; they don't edit code. Delete it when you stop.
- Work on the task's existing branch/worktree under `.claude/wt/<task>` (or a new `video/<task>` branch from
  `video-analysis`). Merge into `video-analysis` only after its tests pass, and don't merge without a review entry in
  `.claude/board/REVIEWS.md`.
- When you stop, append what you did, the real pytest summary line, and what is next to `.claude/board/QUEUE.md`
  and `docs/video-plan/PROGRESS.md`.

## Hard rules (summary of CLAUDE.md)
- Tests run on CPU: in PowerShell `$env:CUDA_VISIBLE_DEVICES='-1'` (not `''`, which deletes the variable on PS 5.1).
  Verify with `.venv\Scripts\python.exe -c "import options; print(options.DEVICE, options.NUM_WORKERS)"` → `cpu 0`.
- Never weaken, skip, xfail or delete a test; never hard-code expected outputs.
- No GPU jobs, model or dataset downloads, `pip install`, `git push`, or branch/worktree deletion without the user's OK.
  The laptop freezes under heavy GPU or many-worker load.
- The desktop GUI must never import torch. Video tensors are `[B, T, C, H, W]`.
- Commits: `video: <what>`. No data, weights, `.venv`, `.artifacts/` or secrets.
- Two failed attempts at the same fix → stop and write it up in PROGRESS.md.
