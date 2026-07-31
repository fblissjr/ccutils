---
name: release
description: Bump version in pyproject.toml + PARSER_VERSION, promote CHANGELOG's Unreleased section, create git tag. Use when cutting a release.
disable-model-invocation: true
---

## Release workflow

### Step 1: Determine version

Ask the user what version to release, or infer from CHANGELOG.md's
`[Unreleased]` content (semver bump matching the size of what's listed).
Semver only, no major bump without explicit permission.

### Step 2: Preflight checks

Run all checks before making changes:

1. **Working tree clean**: `git status` must show no uncommitted changes
2. **Tests pass**: `uv run pytest tests/ --confcutdir=tests -x -q`
3. **CHANGELOG has content to promote**: Verify `CHANGELOG.md`'s `[Unreleased]` section is non-empty
4. **No tag collision**: `git tag -l "v{version}"` must return empty

If any check fails, report and stop.

### Step 3: Bump version -- three places, together

Per `.claude/skills/etl-dev/references/migrations-and-versioning.md`, these
move together or lineage rows from different contracts become
indistinguishable:

1. `pyproject.toml` -- `version = "{new_version}"`
2. `src/ccutils/_version.py` -- `PARSER_VERSION = "{new_version}"` (stamps
   every lineage row; bump even if only populator semantics changed, not
   just the pyproject version)
3. `CHANGELOG.md` -- promote `[Unreleased]` to `## {new_version}` (rename
   the heading; leave a fresh empty `[Unreleased]` above it for subsequent work)

### Step 4: Run doc-drift-checker

Use the doc-drift-checker subagent to verify no hardcoded counts are stale. Fix any drift found before proceeding.

### Step 5: Commit and tag

```bash
git add pyproject.toml src/ccutils/_version.py CHANGELOG.md
git commit -m "release: {version}"
git tag "v{version}" -m "v{version}: {one-line summary from CHANGELOG}"
```

### Step 6: Report

Show:
- Version bumped to
- Tag created
- Reminder: `git push && git push --tags` when ready to publish
