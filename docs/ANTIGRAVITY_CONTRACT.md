<!-- path-privacy: skip-file -- references universal Antigravity data and install paths (not personal) -->
# The Antigravity on-disk contract

Last updated: 2026-11-20

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

**A second corpus, 2026-11-20: a machine with only the CLI store (`agy`
1.5.13).** It contradicts or narrows claims 6, 9, 11, 13, 14 and 15 and the
store map. Claims 6, 13, 14 and 16 below carry the correction; the rest are
listed in `docs/ROADMAP.md` under "Antigravity lake: known gaps" and are not
rewritten here yet. Where a claim below says "hub corpus" or "CLI corpus", it
holds for that one only.

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
- Corrected 2026-11-20 on the owner's Mac (120,475 steps in the two
  stores the canary reads). A column is its field's bytes OR NULL: on 40
  steps the `permissions` column is NULL while the payload states field
  133 once. All 40 are `GENERIC` steps of the desktop app's store
  (`antigravity`), 39 with status `DONE` and one `ERROR`. The count was 39
  on a run earlier the same afternoon, so the app (2.33.1 installed) was
  still writing them that day. No column states anything its field lacks,
  and no other column was NULL beside a stated field. The Linux corpus has
  none.
- Consequence, restated: the payload is the authority. Read a copied value
  from `step_payload`, never from its column, which can be empty. Nothing
  derived reads `permissions` yet.
- Canary: `TestBlobColumnsAreCopies`. A column that contradicts its field
  fails outright; `permissions` left NULL is held as a rate
  (`MAX_UNFILLED`), and any other column left NULL fails. `-s` prints the
  counts and shapes.

## 4. Stored types match declared types

- Measured: `typeof()` over every value of every column of every file: each
  column holds exactly one storage class (plus NULL). `datetime` columns hold
  text (`YYYY-MM-DD HH:MM:SS.ffffff+00:00`); `numeric` booleans hold integers.
- Consequence: SQLite does not enforce declared types, so the writer checks
  every value and fails the unit on a mismatch rather than coercing it.
- Scope, corrected 2026-10-09: this holds for the tables the contract
  declares (`schema.CONVERSATION_TABLES`, `schema.SUMMARIES_TABLES`). It does
  not hold for the full-text index the hub added later (claim 15), and the
  check above applies only to declared columns.
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
- Corrected 2026-11-20 on the CLI corpus. A line belongs to a step when the
  index AND the creation second agree: a revert re-uses indices (claim 14)
  and the transcript appends, so an index also carries the abandoned
  branch's lines. Joined on the index alone, a store where reverts are
  common disagrees far above the threshold; joined on both, no line
  disagrees, and on every index written more than once the LAST line is the
  database's step. The CLI transcript also states more than this claim says:
  tool calls with their arguments, thinking text, and since autumn 2026
  token counts on model responses. It still carries no call id, and a model
  id on few lines.
- Canary: `TestStepNamesAreStated` (rate thresholds in the test; `-s` prints
  the counts).

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
- Enforced at runtime since 2026-11-13. Before that `stores.is_allowed` had
  no caller in `src` and the measurement above held only because discovery
  happened to hand over allowed paths. Both open points (`writer.file_row`,
  `writer.snapshot_sqlite`) now call `writer.require_allowed` first, with the
  data root a required argument, and refuse anything else. A full real run
  after the change (app 2.31.2) wrote every unit with no refusal.
- Not independent of discovery: `is_allowed` and store discovery share
  `KNOWN_STORES`, so widening that constant widens both. The decoys are what
  catch it.
- Canary: `tests/test_antigravity_lake.py::TestReadAllowlist`, with `chmod
  000` decoys, a symlink to the credentials file, and a positive control;
  `test_a_widened_discovery_is_refused_before_the_open` widens discovery to
  `config/mcp_config.json` and asserts from the audit trace that it was
  never opened.

## 13. There is no retention setting, and the CLI prunes anyway

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
- Corrected 2026-11-20 on the CLI corpus: the CLI deletes old conversations
  itself. The count of conversation databases on disk is capped (a round
  number plus those created since the CLI last started); the summaries index
  lists conversations with no database and no brain directory, and every one
  of them was last modified before every conversation still on disk; the
  lake's earlier snapshot holds conversations that have since left the disk.
  Subagent conversations take slots like roots, and the brain directory goes
  with the database. The binary holds `trajectorystore.(*Manager).PruneAll`
  and `summarystore.(*StoreClient).pruneAfterReconcile`; no settings key,
  flag, environment variable or experiment flag names a limit, and the
  vendor's settings reference and changelog state none. The cap's value is
  inferred from counts, not read from the binary. Only `historySize` exists,
  and it caps the prompt history file. The hub was not re-measured.
- Consequence, added: the lake must run more often than the window turns
  over, and the summaries row, which outlives the conversation, is then the
  only statement of its parent and depth.
- Canary: none; re-check on a major app update.

## 14. Every step states its creation time, and a revert re-uses indices

- Measured: all 54,001 steps carry Step metadata field 1 (the creation time
  of claim 6). In the brain transcripts, 3 conversations write 81 step
  indices twice with different times; in 5 of them the step type changes too
  (a `GENERIC` step, then a `USER_INPUT` step written about half an hour later under the same idx). The database
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
- Corrected 2026-11-20 on the CLI corpus: reverts are common there, not
  rare, and the unmeasured question has its evidence. The creation time does
  stay fixed across an in-place update, including the one that empties the
  step (claim 16), so this check cannot see a body that was lost.

## 16. The CLI empties old steps in place

- Measured 2026-11-20 on the CLI corpus: about a third of all steps have
  status 5 (`CORTEX_STEP_STATUS_CLEARED` in the binary). Such a step keeps
  its idx, its creation time and its metadata, usage included, and has lost
  its body: text, thinking, tool output, the typed tool payload and with it
  the exit code, and the tool arguments on about half. They sit in the long
  conversations as a prefix that ends before the last checkpoint. The brain
  transcript, joined as in claim 6, still has content for most of them; a
  few thousand have no transcript line and are gone from disk. Comparing a
  lake snapshot with the same conversations five days later, no step of an
  idle conversation had shrunk, so clearing happens while a conversation is
  live. What triggers it is undetermined.
- Consequence: the database is not a stable record of a step's content, and
  no single source is complete. The lake keeps the fuller row
  (`schema.VERSION_KEYS`): a `steps` row whose payload got shorter under the
  same idx stays beside the new one, marked `source_present = false`. A
  shaped layer reads a cleared step's content from the transcript and says
  that it did.
- Canary: none on the real store. The behaviour is held by
  `tests/test_antigravity_lake.py::TestClearedSteps`.

## 15. The hub's summaries database carries a full-text index, and its columns are untyped

- Measured 2026-10-09 on app 2.21.1: `conversation_summaries.db` in the
  `antigravity` store holds an FTS5 virtual table `conversations_fts`, the
  five tables FTS5 keeps its index in (`_config`, `_content`, `_data`,
  `_docsize`, `_idx`) and `search_schema_version`. The CLI store's database
  has none of them. Row counts and per-column storage classes are in
  `CHANGELOG.md` under "The Antigravity lake survives the hub's full-text
  index".
- Every column of the index is declared with no type. Most hold one storage
  class. `user_step_byte_offsets` and `agent_step_byte_offsets` hold a blob
  in some rows and text in others. `_config` and `_idx` are `WITHOUT ROWID`.
- The indexed columns `title`, `workspace`, `project`, `user_body` and
  `agent_body` are plain text: a conversation's user and agent text in
  chunks, with the step range each chunk covers. It is the first place a
  hub conversation's text can be read without decoding a protobuf blob, and
  it bears on the parked phase 1 decision in `docs/ROADMAP.md`. How complete
  the bodies are against the steps is unmeasured.
- Consequence: the writer takes the kind of a column the contract does not
  declare from the data (`typeof`), not from its declaration. A column
  holding several storage classes is written as bytes plus a
  `<name>__storage` column naming the class of each row
  (`schema.encode_mixed` / `decode_mixed`). A table with no rowid is read in
  primary-key order. Lake format 3.
- This claim exists because the first real run after the app updated lost
  the hub's whole summaries unit to it: an untyped column was assumed to be a
  blob, the first text value failed the type check, and the unit holds every
  conversation's only stated parent and nesting depth (claim 9).
- Re-measured 2026-11-13 on app 2.31.2: the same seven tables, with
  `conversations_fts` at fifteen columns (`conversation_id`, `chunk_index`,
  `chunk_first_step`, `chunk_last_step`, `last_modified_time`, `step_count`,
  `last_step_created_at`, `archived`, the two offsets columns, then the five
  indexed text columns). `conversations_fts_content` stores them as `c0` to
  `c14`, so its mixed `c8` and `c9` are the two offsets columns.
- Declared since 2026-11-13 in `schema.SUMMARIES_OPTIONAL_TABLES`: known
  tables a summaries database may lack (the CLI store's has none), each with
  its column names and its known-mixed columns. Before that every table and
  both offsets columns were noted as drift on every summaries rewrite, which
  would have buried a real change.
- Canary: none on the real store. The fixtures
  (`tests/test_antigravity_lake.py::TestSchemaDrift`) use the 2.31.2 shape:
  it mirrors with no notes, and an added index column is still noted.
- If this changes: a new table, a new column, a listed column gone, or a
  column newly holding several storage classes is mirrored and noted as
  drift in the manifest, and nothing fails.

## 17. What a step's fields hold, by number

Antigravity publishes no schema, so `parsers/antigravity/shaped.py` reads a
`step_payload` by field number. Notation: `5.4.1` is field 1 of the message
in field 4 of the message in field 5.

- Measured 2026-11-20 on the CLI corpus, every step of every database
  against the brain transcript line written for it (claim 6's join). No
  payload failed to parse.

| Field | Reading | Against the transcript |
|---|---|---|
| `1` | step type: 14 `USER_INPUT`, 15 `PLANNER_RESPONSE`, 17 `ERROR_MESSAGE`, 23 `CHECKPOINT`, 101 `SYSTEM_MESSAGE`, 132 `GENERIC` in what the CLI writes today; older conversations hold the per-tool types of claim 6's table, which the name map carries too | `type`, equal on every line |
| `4` | status: 2 `RUNNING`, 3 `DONE`, 4 `INVALID`, 5 `CLEARED`, 6 `CANCELED`, 7 `ERROR` | `status`, equal on nearly all; a line can precede the step's last status |
| `5.1` | creation time {1 seconds, 2 nanos} | `created_at`, the join itself |
| `5.3` | source: 2 `MODEL`, 4 `USER_EXPLICIT`, 5 `SYSTEM`, 6 `SYSTEM_SDK` | `source`, equal on every line |
| `5.9` | usage: `.2` input (uncached), `.3` output, `.5` cache-read, `.9` thinking tokens | the three token counts, equal wherever both state them |
| `5.11` | model, as an enum number (the id string is in `gen_metadata`) | not stated there |
| `5.12` | execution id (a turn) | not stated there |
| `5.20` | lineage: `.3` generation index, `.4` the conversation that owns the step | not stated there |
| `19.2` | a user input's text; a cleared one keeps it only as the repeated parts `19.3[].1` | inside `content`, which wraps it |
| `20.1`, `20.3` | a model response's text, and its thinking text as a separate field | `content` and `thinking`, equal on every line |
| `114.1`, `30.5` | a system message's text; a checkpoint's summary | inside `content` |
| `24.3.2` | an error's message | `error`, equal |
| `140.2.1` | a tool step's output text | inside `content` on most; the transcript truncates some |

- A step whose `5.20.4` names another conversation was replayed into this
  one: some conversations open with a prefix of another's steps
  (`antigravity_steps.is_inherited` counts them). That is this harness's
  fork cut, and it is stated.
- Not read yet, though located by the audit: a command's exit code (inside
  the typed payload at `140.2.6`), the model id string and per-generation
  timings in `gen_metadata`, and the conversation's stated root. Field
  numbers for those are in `internal/antigravity_audit_2026-11-20/FINDINGS.md`
  and become claims here when something reads them.
- Consequence: the shaped layer reads these fields and no others, and reads
  an absent field as NULL, never as 0 or an empty string.
- Canary: `TestStepFieldReadings`, on a sample of the databases with a floor
  per reading; `-s` prints the counts. The full scan agreed with the sample.

## 18. A tool call and its result are one step

- Measured 2026-11-20 on the CLI corpus: every `GENERIC` step carries the
  call in its metadata (`5.4`: `.1` id, `.2` tool name, `.3` arguments as
  JSON) and the result in its body. A model response lists the calls it
  made, and each has a `GENERIC` step with the same id later in the
  conversation. A call id is not unique: the same id occurs more than once
  inside a conversation, and a fork carries its source's ids.
- Consequence: a tool call is keyed on its step, (conversation, idx), and the
  call id is a column. Nothing joins on the id alone. The per-tool step
  types of claim 11 no longer occur at the top level in this store.
- Measured 2026-11-20 on the owner's Mac, through the decoded views: the
  per-tool step types of claim 11 state the call in the same place. Every
  `VIEW_FILE` (3,432), `RUN_COMMAND` (2,083), `GREP_SEARCH` (928),
  `LIST_DIRECTORY` (595), `FIND` (337), `MCP_TOOL` (318) and
  `INVOKE_SUBAGENT` (103) step has an id and a name in `5.4`, and 2,373 of
  2,483 `CODE_ACTION` steps do. `CONVERSATION_HISTORY` and
  `EPHEMERAL_MESSAGE` state none and are not tool calls. So a tool call is
  a step that states one, whatever its type; `antigravity_steps.is_tool_call`
  is that rule. The bodies of the per-tool types are not read; their output
  comes from the transcript line, a reading that is measured for `GENERIC`
  (claim 17) and assumed for them.
- An `ERROR_MESSAGE` step can be a tool call's only record: 132 of them on
  that Mac state a call in `5.4`, and for none does another step of the
  conversation state the same call id. They are tool calls that failed.
- A checkpoint states usage too: 468 `CHECKPOINT` steps on the Linux
  corpus carry token counts and no model, 466 of them under the generation
  index of the response that follows, with different counts. That is read
  as the checkpoint's own call, and it is not a second copy of the
  response's usage.
- Canary: `TestStepFieldReadings` asserts every tool step in its sample
  states an id and a name. The repeated ids, the per-tool counts and the
  checkpoint usage are measurements with no canary.

## 19. A background task is named after the step that started it

- Measured 2026-11-20 on the Linux CLI corpus, through the lake and the
  decoded views. A conversation's brain directory holds
  `.system_generated/tasks/task-<N>.log` (4,021 files in 276
  conversations). `N` is a step index: the step at that index is a
  `run_command` tool step for 3,811 of them and a `schedule` tool step for
  204. The other 6 land on a different step, which is what a revert
  re-using the index would leave (claim 14); not checked one by one.
- When the task ends, the app adds a `SYSTEM_MESSAGE` step whose text names
  it: `Task id "<conversation>/task-<N>" finished with result:` followed by
  the exit code, the output and the log's path. All 3,873 such messages
  name a `run_command` step, and all but 2 come after it. 271 of them name
  a task whose log file the lake does not hold.
- System messages come in kinds a reader can tell apart by their opening:
  a finished task (3,873), a `[Message] timestamp=... sender=...` from an
  agent or task (2,197), a `Reminder:` (53), and others (445).
- A task that is stopped is reported the same way with `was canceled with
  result:` (214 messages), and states no exit code. A fork replays its
  source's reports: 271 of the finished messages sit in a conversation
  other than the one they name, all of them inherited steps, and those are
  the ones whose log the conversation does not hold. No task has more than
  one report of its own. The phrase `exited with code` occurs once in all
  but 2 finished messages; the first occurrence is the task's, since it
  comes before the output.
- Consequence: a background task ties three things with no id of its own:
  the tool step that started it, its log file, and the system message that
  reports it, all through (conversation, step index). `antigravity_tasks`
  is that tie: 4,034 tasks on this corpus, 3,615 finished with an exit
  code, 195 canceled, 224 with no report, 6 whose index no longer holds
  the step that started them.
- Canary: `TestTaskIsNamedAfterItsStep` holds the launch tie as a rate. The
  message wording has none: if it changes, tasks lose their outcome and
  keep their row.

---

## Adding to this document

Same rule as the JSONL contract: a claim earns a place when the lake (or a
later decoder) would be wrong if it stopped being true. Write the
measurement, the consequence, and the canary; "no canary yet" is an
acceptable entry.
