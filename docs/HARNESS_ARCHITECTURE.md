# Harness architecture: more than one transcript source

Last updated: 2026-09-22

> **Status: PROPOSED.** One phase is built: the harness-generic lake runner
> and the Antigravity lake source, shipped as the EXPERIMENTAL `ccutils lake`
> command, outside semver. Everything after it is a plan, scheduled in
> `docs/ROADMAP.md` under "Harnesses beyond Claude Code", and it rides on the
> Tier 2 rewrite in `docs/ETL_ARCHITECTURE.md` (1.3.0) rather than on today's
> staging.

ccutils was built around one harness, Claude Code, and its Tier 1 and staging
assume Claude Code JSONL in about twenty places: line-delimited text, the
Pydantic envelope, `entry_id = md5(path | line)`, `session_id = file stem`,
`.jsonl` path regexes, a fixed-width `etl.log_entries` filled by a positional
`SELECT *`, one hard-wired `record_source`, and discovery rooted at
`<claude-config>/projects`. A second harness forced into that shape would
repeat the assume-instead-of-measure failure `docs/JSONL_CONTRACT.md` exists
to prevent. Antigravity makes the point: its transcripts are SQLite files of
protobuf blobs, not lines. Other candidates are Cowork's `audit.jsonl`,
Claude.ai exports, and plain Gemini CLI chats.

## Principles

1. **Tier 1 is per-harness and native; nothing is normalized or interpreted
   there.** Each harness mirrors its own source at its own grain,
   byte-faithful, inside a shared envelope (`harness, store, record_source,
   source_relpath, lake_run_id, ingested_at, parser_version,
   lake_format_version, source_present`). For a harness whose own storage is
   not durable (Antigravity has already changed formats once, deletes on
   request and re-uses step indices after a revert), the lake is an archive,
   not a cache: nothing it held leaves the current tree except into
   `_superseded/`. Interpretation baked in there would be irreversible, and
   anything derived from the lake (Phase 1's decoded tables) lives in a
   separate tree, so re-deriving never means deleting inside the archive.
2. **Neutrality starts at Tier 2.** The shaped staging tables planned for
   1.3.0 become the harness-neutral contract; each harness gets one adapter
   that decodes its lake into them, once. This does not come for free: facts
   encode Claude Code vocabulary today (`_AGENT_TOOL_NAMES`,
   `TOOL_CATEGORIES`, the `api_message_id` grain), so each harness also
   declares a vocabulary map. Concepts that exist in one harness only
   (Claude Code's queue operations, Antigravity's checkpoints and battle
   mode) keep harness-specific staging and facts.
3. **Identity comes from the harness's own natural key, never from a path.**
   An Antigravity step is `(conversation id, idx)`, wherever the file sits.
   Warehouse keys get a harness namespace in 1.3.0's key-formula module.
4. **Reads are an allowlist that fails closed.** A harness declares the
   paths it may open; a test proves it opens nothing else, against planted
   decoys. That keeps credential files out; it does not make a lake
   shareable. Shared artifacts stay scoped, never scrubbed.
5. **Every harness has a contract doc** in the `JSONL_CONTRACT.md` format:
   measured claims, each with a canary. `docs/ANTIGRAVITY_CONTRACT.md` is
   the second.
6. **Prefer stated over inferred.** A harness that states a value (step
   names, subagent depth, exit codes) is read, not re-derived.

## Tiers per harness

```
Tier 0  native store             Claude Code JSONL | Antigravity SQLite+protobuf | ...
Tier 1  lake, per harness        <lake_root>/<harness>/...   byte-faithful mirror + envelope
Tier 2  shaped staging           neutral stg_* (with `harness`) + harness-specific stg_*
Tier 3  dims and facts           read staging only (ETL_ARCHITECTURE rule 1)
Tier 4  views
```

The lake runner (`src/ccutils/parsers/lake.py`) is harness-generic: stat
fingerprints (including a SQLite `-wal`), a per-source `lake_format_version`
so a widened writer re-derives an existing lake, atomic unit swaps with
crash recovery, archive semantics (a unit kept on disappearance; rows of a
keyed table and whole tables the source dropped carried forward marked
`source_present = false`; the old unit superseded when the source says the
new one is not a continuation, or when a carry would drop a column),
missing-unit detection scoped to the stores actually scanned, a manifest, a
run log and a lock. A `LakeSource` supplies `discover`, `write_unit` and
`supersedes`; a unit declares its `carry_keys`. The interface is
provisional until Claude Code joins it in Phase 2.

## Neutral staging, sketch

Drafted from the measured shapes of two harnesses; to be finalized against at
least two, ideally three (with Gemini CLI), before 1.3.0 builds it.

| Table | Grain | Claude Code source | Antigravity source |
|---|---|---|---|
| `stg_sessions` | session | file stem, sidecar | summaries row, `trajectory_metadata_blob` |
| `stg_messages` | message | user/assistant entries | `USER_INPUT`, `PLANNER_RESPONSE` (text, thinking) |
| `stg_tool_calls` | call | assistant `tool_use` blocks | planner response tool calls |
| `stg_tool_results` | call | user `tool_result` blocks | tool steps, joined on the call id in Step metadata |
| `stg_usage` | model response | `message.usage` | `gen_metadata` usage |
| `stg_delegations` | spawn | Agent/Task calls + sidecar | `INVOKE_SUBAGENT` + stated parent and depth |
| `stg_artifacts` | artifact version | ExitPlanMode plans | brain artifacts, later `.git` snapshots |

## Phases

The schedule, the concrete steps and the gates live in `docs/ROADMAP.md`
under "Harnesses beyond Claude Code"; this document holds the reasoning
behind them. In short: phase 0 (the raw Antigravity lake) is built and
experimental; phase 1 decodes it through the app's own descriptors and 1b
decrypts the legacy files, neither touching the warehouse; phases 2 to 4
(neutral staging, harness identity, Antigravity populators) ride on the 1.3.0
rewrite, because building them on today's `log_entries` staging would build
the layer that rewrite replaces; phase 5 is render and privacy.

## Open questions

Tracked in `docs/ROADMAP.md` under "Open design questions": what a "project"
is across harnesses (Claude Code's transcript directory vs Antigravity's
workspace URI, git root and `project_id`), whether `fact_messages` splits
into neutral and harness-specific parts, how forks and battle mode map, and
whether the archive should ever honour in-app deletions.
