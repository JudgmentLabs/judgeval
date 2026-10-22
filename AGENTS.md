# AGENTS.md

## GitHub Actions

- Coding agents must never modify, delete, rename, or bypass
  `.github/workflows/merge-branch-check.yaml`, including through scripts or
  generated changes. Read-only inspection is allowed. If a task requires a
  change to this workflow, leave it unchanged and ask a human to make the change.
