---
name: release
description: Bump version in pyproject.toml, verify CHANGELOG entry exists, create git tag. Use when cutting a release.
disable-model-invocation: true
---

## Release workflow

### Step 1: Determine version

Ask the user what version to release, or infer from CHANGELOG.md (the first `## X.Y.Z` heading that doesn't have a matching git tag).

### Step 2: Preflight checks

Run all checks before making changes:

1. **Working tree clean**: `git status` must show no uncommitted changes
2. **Tests pass**: `uv run pytest tests/ -x -q`
3. **CHANGELOG entry exists**: Verify `CHANGELOG.md` has a `## {version}` heading with content
4. **No tag collision**: `git tag -l "v{version}"` must return empty

If any check fails, report and stop.

### Step 3: Bump version

Update `pyproject.toml` version field to the new version:

```
version = "{new_version}"
```

### Step 4: Run doc-drift-checker

Use the doc-drift-checker subagent to verify no hardcoded counts are stale. Fix any drift found before proceeding.

### Step 5: Commit and tag

```bash
git add pyproject.toml
git commit -m "bump version to {version}"
git tag "v{version}" -m "v{version}: {one-line summary from CHANGELOG}"
```

### Step 6: Report

Show:
- Version bumped to
- Tag created
- Reminder: `git push && git push --tags` when ready to publish
