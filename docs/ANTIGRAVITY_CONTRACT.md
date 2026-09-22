<!-- path-privacy: skip-file -- references universal Antigravity data and install paths (not personal) -->
# The Antigravity on-disk contract

Last updated: 2026-09-22

What `ccutils lake antigravity` assumes about how Google Antigravity (the Gemini
agentic IDE/hub and its `agy` CLI, from the Windsurf "Cascade" lineage) stores
conversations, and where everything is. Antigravity publishes no format and no
version; it has already changed its tool-step encoding once inside this corpus
and its storage format once before that. So, as in `docs/JSONL_CONTRACT.md`,
every assumption is a claim with the measurement behind it and a canary that
goes red when it stops being true (`tests/test_antigravity_contract.py`).

Corpus at time of writing: Antigravity hub 2.15.1 installed; 414 conversation
databases (309 hub, 105 CLI), 54,001 steps, 2025-11-21 to 2026-09-19. Unless a
claim says "survey", the numbers were measured by a full scan, not a sample,
and the canary re-measures them on every test run.

---

## Store map

The data root is `~/.gemini`. Stores are subdirectories named by the app's
`--app_data_dir`:

| Store | What it is | Ingested |
|---|---|---|
| `antigravity/` | the hub/IDE | yes |
| `antigravity-cli/` | the `agy` CLI | yes |
| `antigravity-ide/` | the IDE app's own data dir; today a byte-identical migration snapshot (19 legacy `.pb`, no `.db`) | yes (claim 10) |
| `antigravity-backup/` | the copy the hub names `IDE_BACKUP_DATA_DIR` in its `paths.js` (survey) | no, reported |
| `tmp/`, `tasks/`, `users/`, `extensions/` | plain Gemini CLI data, a Python sandbox, extensions | no (Gemini CLI is a separate harness) |

Inside a store (counts are hub + CLI):

| Path | Format | Count | Key | Lake |
|---|---|---|---|---|
| `conversations/<uuid>.db` (+ `-wal`) | SQLite, protobuf blobs | 414 | filename = conversation id (claim 2) | every table, byte-faithful |
| `conversations/<uuid>.pb` | encrypted legacy store (claim 7) | 28 | filename | ciphertext bytes |
| `conversation_summaries.db` | SQLite index of every conversation | 425 rows | `conversation_id` | mirrored |
| `brain/<uuid>/.system_generated/logs/transcript_full.jsonl` | JSONL projection of steps, step names stated (claim 6) | 1 per conversation | `step_index` | bytes |
| `brain/<uuid>/.system_generated/logs/transcript.jsonl` | same, fields truncated; the only readable form of 7 CLI `.pb` conversations (survey) | | | bytes |
| `brain/<uuid>/.system_generated/logs/chunks/**` | exact split of the two files above (claim 8) | | | excluded |
| `brain/<uuid>/.system_generated/messages/**` | agent-to-agent messages (JSON; `undelivered/` has no extension) | | | bytes |
| `brain/<uuid>/.system_generated/steps/<N>/*` | full tool output for step N (`output.txt`, `content.md`) | | | bytes |
| `brain/<uuid>/.system_generated/tasks/*.log` | background task logs | | | bytes |
| `brain/<uuid>/*.md`, `artifacts/*.md`, `*.metadata.json`, `*.resolved[.N]` | artifacts (task, implementation plan, walkthrough) and versions | | | bytes |
| `brain/<uuid>/.git/` | per-conversation snapshot history, "Snapshot" commits (survey: 184 repos, up to 769 commits, 63,039 loose objects, 0 packs, never gc'd; 1.2 GB on disk) | | | bytes, as its own `brain_git` unit; interpreted in Phase 4 |
| `brain/<uuid>/.user_uploaded/`, `.tempmediaStorage/`, images, video | user uploads and media | | | inventory only |
| `browser_recordings/<uuid>/` | `metadata.json` plus JPEG frames | | | metadata bytes, frames inventory |
| `annotations/<uuid>.pbtxt` | protobuf text: title, pinned, archived, last viewed | | | bytes |
| `history.jsonl` (CLI) | prompt history: `display`, `timestamp` (ms), `workspace`, `conversationId` | | | bytes |
| `agyhub_summaries_proto.pb`, `jetbox_summaries_proto.pb` | plaintext protobuf mirror of the summaries | | | bytes |
| `implicit/<uuid>.pb` | encrypted "implicit" trajectories, not conversations | 46 files across 3 stores | | ciphertext bytes |
| `code_tracker/`, `mcp/`, `builtin/`, `plugin_data/`, `scratch/`, `bin/`, `*.pbtxt` state, `user_settings.pb` | caches, config, bundled skills, agent scratch | | | not read |

Outside any store: `config/projects/*.json` names the `project_id` a conversation
carries (ingested). The data root also holds `oauth_creds.json`,
`jetski-standalone-oauth-token`, `google_accounts.json`, a browser profile and
`config/mcp_config.json`; none is ever opened (claim 12).

Inside a conversation database:

| Table | Rows | Holds |
|---|---|---|
| `trajectory_meta` | 1 | `trajectory_id`, `cascade_id` (= filename), `trajectory_type`, `source` |
| `steps` | 54,001 | `idx` (0..n-1), `step_type`, `status`, and blobs; `step_payload` is a complete serialized `gemini_coder.Step` (claim 3) |
| `gen_metadata` | 25,956 | one model generation each: model, token usage, latency, and on one row per conversation a full prompt snapshot (survey: up to 97 MB) |
| `executor_metadata` | 1,607 | one per execution: termination reason, config snapshot |
| `parent_references` | 4 | forks |
| `trajectory_metadata_blob` | 1 | workspace, git root and branch, parent/root conversation, nesting depth, project id |
| `battle_mode_infos` | 0 | declared, never populated here |

The installed binaries embed the full protobuf schema (survey: 504 descriptor
files in `language_server` 2.15.1; decoding with it found 0 unknown fields in
2.6M). ccutils does not commit it; Phase 1 extracts it locally
(`docs/HARNESS_ARCHITECTURE.md`). The lake decodes nothing.

---

## 1. The conversation schema is uniform; the summaries columns are

- Measured: all 414 conversation databases have byte-identical DDL, equal to
  `REAL_CONVERSATION_DDL` in `tests/helpers_antigravity.py`. The two summaries
  tables have the same 21 columns in the same order, but not the same DDL
  text: the CLI's gained `raw_summary` through a migration and declares it
  unquoted (`raw_summary BLOB`). `PRAGMA user_version` is 1 on conversations
  and 3 on summaries.
- Consequence: fixtures are built from the real DDL. The writer checks
  columns, not DDL text: an unknown column or table is mirrored and recorded
  as drift in the manifest; a missing one fails that unit only.
- Canary: `TestSchemaIsUniform`.
- If this changes: fixtures stop modelling the real files, and
  `schema.CONVERSATION_TABLES` needs the new column.

## 2. The filename is the conversation id

- Measured: in all 412 non-empty databases, `trajectory_meta` has exactly
  one row and its `cascade_id` equals the filename stem. Both empty databases
  are exempt: one carries a `cascade_id` found nowhere on disk. The same id
  names the brain directory, the summaries row and the annotation file.
  `trajectory_id` is a different UUID.
- Consequence: the lake keys a conversation on `(store, filename stem)` and
  stamps it on every row as `conversation_id`.
- Canary: `TestFilenameIsTheConversationId`.

## 3. The step blob columns are copies of fields of `step_payload`

- Measured: on all 54,001 steps, `step_type` is Step field 1, `status` is
  field 4, `metadata` is byte-identical to field 5, `error_details` to 31,
  `permissions` to 133, `task_details` to 148, and `has_subtrajectory` is
  true exactly when field 6 is present. `render_info` lives deeper, at
  `generic.result.step_render_info` (survey). `step_format` is 0 everywhere
  and has no enum in any descriptor (survey).
- Consequence: `step_payload` alone carries the step; the lake still mirrors
  every column, because a mirror is not the place to decide which copy wins.
- Canary: `TestBlobColumnsAreCopies`.

## 4. Stored types match declared types

- Measured: `typeof()` over every value of every column of every file: each
  column holds exactly one storage class (plus NULL). `datetime` columns hold
  text (`YYYY-MM-DD HH:MM:SS.ffffff+00:00`); `numeric` booleans hold integers.
- Consequence: SQLite does not enforce declared types, so the writer checks
  every value and fails the unit on a mismatch rather than coercing it.
- Canary: `TestStorageTypes`.

## 5. The files are in WAL mode

- Measured: header bytes 18 and 19 are 2 on all 414 conversation files and
  both summaries files. No `-wal` existed during the survey, but an open
  conversation writes there first. `PRAGMA journal_mode` under `immutable=1`
  misreports this as `delete`.
- Consequence: `immutable=1` ignores the WAL and would miss recent steps, and
  opening the original even `mode=ro` may create `-wal`/`-shm` beside the
  app's files. The writer copies `.db` and `-wal` (never `-shm`) into its own
  scratch dir only when both files' stat is unchanged across the copy, and
  opens the copy. Change detection stats the `-wal` too, since a WAL write
  leaves the main file untouched.
- Canary: `TestWalMode`; behaviour in `test_antigravity_lake.py::TestWal`,
  whose fixture keeps rows only in the WAL and checks that first.

## 6. Step-type names are stated in the brain transcript

- Measured: 49,777 transcript lines join to a step by `step_index`. Each
  `step_type` has one dominant name (`15` `PLANNER_RESPONSE`, `132`
  `GENERIC`, `8` `VIEW_FILE`, `14` `USER_INPUT`, ...); fewer than 0.01% of
  lines disagree. For every index written once, the line's `created_at`
  equals Step metadata field 1 to the second; all 32 mismatches are on
  indices the transcript writes twice. The transcript covers about 94% of
  steps and omits token counts, model ids and call ids (survey), so it is a
  label source, not a replacement for the database.
- Consequence: Phase 1 can name step types from data the app wrote rather
  than from a hard-coded table, and the descriptors can be cross-checked.
- Canary: `TestStepNamesAreStated` (rate threshold 0.1%).

Observed step types, both stores: 15 `PLANNER_RESPONSE` 26,293 · 132 `GENERIC`
14,200 · 8 `VIEW_FILE` 3,432 · 5 `CODE_ACTION` 2,483 · 21 `RUN_COMMAND` 2,083 ·
101 `SYSTEM_MESSAGE` 1,291 · 14 `USER_INPUT` 967 · 7 `GREP_SEARCH` 928 · 9
`LIST_DIRECTORY` 595 · 23 `CHECKPOINT` 385 · 17 `ERROR_MESSAGE` 357 · 25 `FIND`
337 · 38 `MCP_TOOL` 318 · 127 `INVOKE_SUBAGENT` 103 · 98 `CONVERSATION_HISTORY`
102 · 33 `SEARCH_WEB` 33 · 91 `GENERATE_IMAGE` 32 · 28 `COMMAND_STATUS` 20 · 31
`READ_URL_CONTENT` 17 · 90 `EPHEMERAL_MESSAGE` 13 · 138 `ASK_QUESTION` 9 · 85
`BROWSER_SUBAGENT` 3.

## 7. Legacy `conversations/*.pb` are encrypted

- Measured: all 47 (20 hub, 8 CLI, 19 in `antigravity-ide`) have 7.99 to 8.00
  bits of entropy per byte and do not parse as protobuf. The pre-migration
  backup copies are encrypted too, so this is the old storage format, not a
  migration artifact. The app still reads them. Community tools report
  AES-128-CTR with the key in the macOS Keychain item `Antigravity Safe
  Storage`; unverified here, because it needs a credential this work was not
  given.
- Consequence: the lake archives the ciphertext bytes. Opt-in decryption is
  a later phase.
- Canary: `TestLegacyPbIsEncrypted`.

## 8. `logs/chunks/` duplicates the whole transcript files

- Measured: for all 368 chunk directories (both stores, both transcript
  kinds), the chunks concatenated in name order equal the whole file byte for
  byte.
- Consequence: the lake excludes `chunks/`.
- Canary: `TestChunksDuplicateTheWholeFile`.

## 9. Subagent parent and depth are stated

- Measured: 259 of 336 hub summary rows have `nesting_depth` 1 to 4, and
  exactly those rows name a `parent_conversation_id`. The parent's
  `INVOKE_SUBAGENT` step names the child ids (survey: 254 of 254 where the
  parent file exists).
- Consequence: delegation depth is read, never walked (the lesson of
  Claude Code's `spawnDepth`).
- Canary: `TestSubagentLinksAreStated`.

## 10. Stores are disjoint, and the backup adds nothing

- Measured: no conversation id has a `.db` in more than one store.
  `antigravity-backup` holds no `.db`, and its `.pb` files are byte-identical
  to `antigravity-ide`'s.
- Consequence: `(store, conversation id)` is a sufficient unit key; skipping
  the backup loses nothing. `antigravity-ide` is ingested because it is the
  IDE's live data dir name; today its 19 conversations duplicate the hub's.
- Canary: `TestStoresAreDisjoint`.

## 11. Tool calls moved to GENERIC steps, at different times per store

- Measured, per-tool step types (5, 7, 8, 9, 21, 25) against step 132 by the
  month of Step metadata field 1: the hub wrote 6,825 per-tool and 453 GENERIC
  in July 2026 and only GENERIC after; the CLI wrote 212 per-tool next to
  1,292 GENERIC in August and none since. GENERIC wraps the old step as
  `google.protobuf.Any` (`type.googleapis.com/gemini_coder.Step`, survey).
- Consequence: any decoder reads both encodings.
- Canary: `TestGenericToolSteps` fires if a per-tool step appears after
  2026-09-01.

## 12. Reads are an allowlist, and credentials are never opened

- Measured: a full real run under a Python audit hook opened 13,327 paths
  under the data root, none outside the allowlist, no subprocess, no
  credential file. At lake format 2, which also reads `brain/<id>/.git`:
  77,102 paths, none outside the allowlist.
- Consequence: the allowlist in `parsers/antigravity/stores.py` is anchored
  regexes over store-relative paths; symlinks are recorded, never followed,
  and files are opened `O_NOFOLLOW` so one planted between the `lstat` and
  the open is not followed either; excluded directories are never entered. It keeps credential FILES out. It
  does not make the lake shareable: transcript content can hold secrets that
  agents printed.
- Canary: `tests/test_antigravity_lake.py::TestReadAllowlist`, with `chmod
  000` decoys, a symlink to the credentials file, and a positive control.

## 13. There is no retention setting

- Measured: no retention, auto-delete or cleanup setting in the hub's
  `app.asar`, the `language_server` binary, the IDE extension's settings or
  either settings file; deletion is a user action (`DeleteCascadeTrajectory`).
  `pruneTrajectory` touches only implicit trajectories.
- Consequence: conversations persist until deleted in the app; the risks to
  the Tier 0 copy are in-app deletes, reverts (claim 14), format changes and
  data-dir migrations. The lake is an archive: nothing it held leaves the
  current tree except into `_superseded/`. A unit whose source disappears is
  kept; a summaries row, a brain file or a whole table the source dropped is
  carried forward with `source_present = false`.
- Canary: none; re-check on a major app update.

## 14. Every step states its creation time, and a revert re-uses indices

- Measured: all 54,001 steps carry Step metadata field 1 (the creation time
  of claim 6). In the brain transcripts, 3 conversations write 81 step
  indices twice with different times; in 5 of them the step type changes too
  (e.g. idx 142 `GENERIC` at 18:46, then `USER_INPUT` at 19:16). The database
  holds only the later step for every one checked: a revert truncates the
  conversation and the new branch re-uses the indices. The earlier branch
  survives on disk only in the transcript, which appends.
- Consequence: an idx-set check cannot see a revert once the new branch is as
  long as the old one. The lake compares field 1's raw bytes per idx and
  supersedes the old snapshot when any differ (the bytes are compared, never
  decoded or stored). A spurious difference costs one extra snapshot; a
  missed one would lose the pre-revert branch, so the check errs toward
  superseding. Whether field 1 stays fixed across in-place status updates is
  unmeasured (it needs two snapshots of a running step); a violation would
  show as superseded units on runs where nothing was reverted.
- Canary: `TestStepCreationTimeIsStated` (every step has field 1). The reuse
  itself is a survey number, not a canary: the design is the same whether it
  is rare or common.

---

## Adding to this document

Same rule as the JSONL contract: a claim earns a place when the lake (or a
later decoder) would be wrong if it stopped being true. Write the
measurement, the consequence, and the canary; "no canary yet" is an
acceptable entry.
