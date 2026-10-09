# How to use this plan with Claude Code (VS Code)

## One-time setup
1. Copy `CLAUDE.md` to the repo root and `docs/video-plan/` into the repo.
   Commit on the `GUI` branch: `docs: add video analysis term plan`.
2. In Claude Code settings, auto-approve safe actions (reading files, running
   pytest/scripts); keep approval ON for deleting files, installing packages,
   and `git push`.

## Every phase
1. Start a **fresh** Claude Code session (`/clear`).
2. Paste the short starter below, replacing N and the file name.
3. Review its plan before letting it code. Push back early if it's off.
4. When it says done: run `scripts\check.ps1` yourself, try the feature, and
   look at `git diff --stat tests/` (no weakened/deleted tests).
5. Paste its final summary + test output into your Claude chat (the planner)
   for review. Adjust the next phase prompt if needed.

## Starter message (paste into Claude Code)

```
Read CLAUDE.md, docs/video-plan/PLAN.md and docs/video-plan/PROGRESS.md.
Then read docs/video-plan/prompts/phase-N-<name>.md and carry out Phase N
only, exactly as specified there.

Start by giving me your plan: files you'll create/change, new CLI flags,
the tests you'll add, and the risks you see. Wait for my OK before writing code.

When you implement: check your work, finish it completely, add all relevant
assurance checks, and test it rigorously. Never weaken, skip or delete an
existing test, and never hard-code results. Run scripts\check.ps1 (or the
full pytest suite before Phase 0 creates it) and paste the real summary line.
Commit in small steps. Update docs/video-plan/PROGRESS.md at the end.
If the same fix fails twice, stop and tell me what you tried.
```

## If something goes wrong
- It loops on a failing fix → stop it, `/clear`, paste the exact error text and
  the relevant PROGRESS.md entry into a fresh session.
- It changed a test's expected value → ask it to justify the change from the
  spec, or revert that file.
- A phase is too big for one session → ask it to finish the current step,
  update PROGRESS.md and commit; continue in a new session with
  "continue Phase N from PROGRESS.md".
