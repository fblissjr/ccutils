# Harness architecture: more than one transcript source

Last updated: 2026-09-22

> **Status: PROPOSED.** Phase 0 (below) is built: the harness-generic lake
> runner and the Antigravity lake source, shipped as the EXPERIMENTAL
> `ccutils lake` command, outside semver. Everything after it is a plan. It
> rides on the Tier 2 rewrite in `docs/ETL_ARCHITECTURE.md` (1.3.0) rather
> than on today's staging.

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
   lake_format_version`). For a harness whose own storage is not durable
   (Antigravity has already changed formats once and deletes on request), the
   lake is an archive, not a cache. Interpretation baked in there would be
   irreversible.
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
crash recovery, archive semantics (kept on disappearance, superseded instead
of overwritten when rows vanish), a manifest, a run log and a lock. A
`LakeSource` supplies `discover` and `write_unit`. The interface is
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

| Phase | What | Rides on | Warehouse change |
|---|---|---|---|
| 0 Raw lake (built) | Generic runner; Antigravity source; contract doc; `ccutils lake antigravity`, experimental | nothing | no |
| 1 Decode | Extract the protobuf descriptor set from the installed app into `lake/antigravity/_schema/<app>-<version>/` (never committed); add the `protobuf` dependency; decode steps, generations and summaries to named JSON; `antigravity_*` DuckDB views, marked unstable. Gates: an unknown-field walk finds 0, and text matches the brain transcript | nothing | no |
| 1b Legacy decrypt | Opt-in `--decrypt-legacy` (macOS): read the Keychain item at runtime, never store the key, fail loudly without it; verify the plaintext decodes as a `Trajectory` with 0 unknown fields before writing it into the same tables | 1 | no |
| 2 Neutral staging | Write the `stg_*` contract and vocabulary maps into `ETL_ARCHITECTURE.md`; Claude Code goes lake-first (`ccutils lake claude_code`, staging reads the lake) and the `LakeSource` interface is revised against it | 1.3.0 | rebuild |
| 3 Identity | Harness column or dimension; namespaced keys; `record_source` carried from staging instead of a constant; per-harness parser version; a neutral project identity; model provider and family from the stated id | 1.3.0 | rebuild |
| 4 Antigravity populators | Neutral facts through the adapter, plus Antigravity-only facts (checkpoints, agent messages, forks and battle mode, browser subagent); `ccutils audit` per harness | 1.3.0 | rebuild |
| 5 Render and privacy | HTML and markdown through neutral staging; `--no-thinking` wired and asserted on every new surface (Antigravity's thinking text is real, unlike Claude Code's empty blocks) | 4 | no |

Phases 0, 1 and 1b add no DDL and do not touch the 1.0.0 gate. Phases 2 to 4
are deliberately not built on today's `log_entries` staging, which is the
layer 1.3.0 replaces.

## Open questions

- What a "project" is across harnesses: Claude Code's is the transcript
  directory; Antigravity states a workspace URI, a git root and a
  `project_id` named in `config/projects`.
- Whether `fact_messages` splits into neutral and harness-specific parts.
- How forks and battle mode map; Claude Code has no analogue.
- Whether the archive should ever honour in-app deletions (a prune flag).
