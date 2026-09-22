# Roadmap and open items

Last updated: 2026-09-22

This is the one git-tracked status board. It consolidates what used to live
only in gitignored `internal/` notes: the release sequence, the next
concrete steps, and every known defect or open question that has been found
but not fixed. When a session ends, update this file, not a scratch note.

Detail that is deliberately not tracked (corpus measurements, per-session
logs, the external audit transcripts) stays under `internal/`. Where an item
below has a longer write-up there, the file is named so it can be found on a
machine that has it; nothing here depends on it.

## Where things stand

- **Released:** 0.20.1, tagged 2026-08-28.
- **In progress toward 1.0.0 (paused 2026-09-10, resume here).** All
  eight numbered steps landed, plus the first two of the four merges
  decided the same day: `fact_tool_calls` (uses + results + chain steps +
  errors in one table) and the summary and session-file bridge as views.
  `ccutils audit` exits clean on the full corpus and on a one-project
  subset build apart from columns that project never fills. Version is
  still 0.20.1; the tag waits on the list under "Next session".
- **A second harness, Tier 1 only (2026-09-22).** `ccutils lake
  antigravity` mirrors Google Antigravity's native store (SQLite files of
  protobuf blobs, brain files, encrypted legacy files) byte-for-byte into a
  Parquet archive under the home-anchored lake root (`default_lake_root()`),
  through a harness-generic lake runner. EXPERIMENTAL and outside semver:
  the command and the lake layout may change before they are declared
  stable. No warehouse change. Where this goes next: "Harnesses beyond
  Claude Code" below. Design: `docs/HARNESS_ARCHITECTURE.md`; store map and
  measured claims: `docs/ANTIGRAVITY_CONTRACT.md`.
- **Done and closed:** the JSONL contract doc (`docs/JSONL_CONTRACT.md`), the
  test audit, the CLI defect walk, the CLI restructure, every stated sidecar
  field, delegation outcomes, and the history-scoping leak. Their write-ups
  are in `CHANGELOG.md` under 0.19.1 through 0.20.1.

## Next session

Resume in this order. Each item was designed and its reads done on
2026-09-10; nothing below is open-ended.

1. **Start from a green suite.** The full suite passed on the last
   commit of 2026-09-10 (1,308 passed, 1 skipped). Rerun
   `uv run pytest tests/ --confcutdir=tests` once to confirm the tree is
   as left, then begin step 2.
2. **Delegations become a view; `--embed` is retired.** Delete
   `fact_agent_delegations`, its populator, `populate_delegation_completion`
   and `run_post_session_reconciliation` (and the `reconciliation` run
   kind). `semantic_agent_delegations` is a view: the spawn row from
   `fact_tool_calls` where the tool is Agent or Task (task description,
   prompt, subagent type, `agent_is_async`, `agent_id`, the stated rollup
   columns, `is_error` as spawn failure) joined to the child `dim_session`
   on `md5('agent-' || agent_id)` and to `semantic_session_summary` for the
   derived duration, tokens and tool count. `completion_state`: spawn
   failed when the spawn row is an error; `completed` when the child's last
   assistant message carries a stop reason (compare the two maxima, never
   `arg_max`, which skips NULLs); `no_completion_recorded` otherwise; NULL
   when the child transcript is absent. Derived columns keep the
   `derived_` prefix; stated and derived never share a column. Add
   delegation features to `semantic_session_summary` (delegation count,
   spawn failures, max child depth, delegated output tokens). `--embed`
   goes with it: `embed_sessions` and `cluster_sessions` were dead and
   `match_delegations` wrote into the table being removed; delete
   `schemas/star/embeddings.py`, `fact_session_embeddings`, the flag on
   both commands, the `colbert` extra, and the README lines. Tests to port
   from `test_fact_agent_delegations_v15.py`: one row per spawn, task input
   captured, refused spawn is `spawn_failed`, async spawn re-derived from
   the child transcript, sync spawn keeps stated values, no completion
   leaves derived NULL, stated and derived never share a column. Update
   the audit's ground-truth check to read the view, `test_etl_metadata`
   and `test_cli_surface` for the removed run kind and flag, and the
   downstream consumer's queries (it reads `semantic_agent_delegations`).
3. **Collapse the five thin entry facts** (`fact_attachments`,
   `fact_meta_events`, `fact_queue_operations`, `fact_pr_links`,
   `fact_file_history_snapshots`) into
   `fact_entry_events(entry_id, entry_type, subtype, value_text,
   payload_json)`; `fact_system_events` and `fact_progress_events` stay.
   `semantic_session_summary`, `semantic_decisions` and `fact_diagnostics`
   read the thin tables; `dim_session` meta columns read staging and are
   unaffected.
4. **Delete the five plain-join views** marked `delete` in
   `TABLE_COVERAGE` and update the query-warehouse skill's routing table
   and recipes, which still name them.
5. **Gate and tag.** One full-corpus build (about 15 minutes), `ccutils
   audit` clean, then `docs/STAR_SCHEMA.md` table list, `README.md`,
   `CHANGELOG.md` promoted from Unreleased, `pyproject.toml` and
   `PARSER_VERSION` to 1.0.0, tag `v1.0.0`. Bump approved by the owner
   on 2026-09-10.

Working notes: verify with a one-project subset build
(`ccutils --source --format duckdb -o <dir> -p ccutils`, about 70
seconds) and `ccutils audit -o <dir>`; the full corpus is for the final
gate only. Two scratch warehouses from this work sit under the ccutils
archive root (`audit-calibration`, `audit-subset`); delete them after the
tag. If the schema changes again, subset builds must be rebuilt, not
reopened.

## Release map

Sequence settled in a design interview on 2026-08-28. The destination is a
warehouse the owner and agents can query and trust, whose end artifact is
the DuckDB file plus a generated reader's guide, scoped per project and never
git-tracked.

| Release | Contents | Gate |
|---|---|---|
| **1.0.0** | The warehouse break (below) | **Next.** `ccutils audit` exits clean on a fresh full-corpus build |
| 1.1.0 | Decomposition + incremental CDC; Bash-derived file operations, skill and slash-command invocations, hook and permission denials; `tool-results/` index (path, size, type, never bodies) | Derived rows carry `derived_from`; a second run performs zero upsert steps |
| 1.2.0 | `fact_commits` + attribution view; `ccutils report`; reader's guide extended; behavior analytics folded into the report | The report answers every question it declares |
| pause | Use the tool for a while before rewriting under it | |
| 1.3.0 | The ETL layer rewrite (`docs/ETL_ARCHITECTURE.md`) | Report-output diff: every changed answer explained |

Behind the rewrite, unscheduled: `fact_file_backups`, memory history
backfill, the Cowork `audit.jsonl` ingester, facet Tier 2 to Tier 3
clustering, comprehensive `--private` hardening.

Harnesses other than Claude Code run on their own track, below
("Harnesses beyond Claude Code"): phase 0 is built, phases 1 and 1b add no
DDL and can land any time, phases 2 to 4 ride on 1.3.0.

Semver binds from 1.0.0 onward. No major bump without the owner's permission.

## 1.0.0: the warehouse break

One interlocking change, so one release. Warehouses written before it cannot
be read after it; that is the whole point of the major bump.

Do these in order. Steps 1 through 8 are DONE (2026-09-10); their text is
kept so the ordering argument survives.

1. **Delete `src/ccutils/schemas/migrations/` and `tests/test_migrations.py`.**
   The runner has zero call sites. It is also why `etl.schema_version` has
   never had a row.
2. **Delete `_COLUMN_MIGRATIONS`, `_apply_column_migrations` and its backfills**
   in `schemas/star/schema.py`. Every column on that list is already in its
   table's CREATE (`TestCreateTableIsSelfSufficient` guards this). The sync
   test goes with the list.
3. **Delete `_repair_duplicate_natural_keys` and
   `tests/test_natural_key_repair_v15.py` in the same change as step 4.** The
   repair is the current safety net for warehouses built before the
   natural-key raise. Never delete it before the replacement lands.
4. **`create_star_schema` refuses a schema it did not write** and tells the
   user to rebuild. No compatibility views, no upgrade path.
5. **Move metadata and staging to an `etl` schema**: `etl.runs`,
   `etl.batch_runs`, `etl.steps`, `etl.versions`, `etl.schema_version`,
   `etl.table_coverage`, `etl.audit_exceptions`, `etl.log_entries`. The main
   schema then holds only dims, facts and semantic views. The md5
   `version_key` becomes a plain version string (`0.20.1/1`: ccutils version
   and business-rules version).
6. **Coverage layer.** `etl.table_coverage` records per table whether it is
   populated, a declared stub, or conditionally populated, and why. This
   replaces the stub list in
   `.claude/skills/query-warehouse/references/gotchas.md`, which two external
   auditors failed to find. Record here that reasoning text is not on disk
   (`has_thinking` counts blocks whose `thinking` field is empty at the
   source), so nobody builds a populator for it.
7. **`ccutils audit`.** Three check families, walking `NATURAL_KEYS` rather
   than a hand-kept list:
   - Structural invariants: no join key on a populated fact is 100% NULL; no
     column is uniformly single-valued unless declared; every FK resolves;
     only declared stubs are empty; views return rows when their sources do;
     every declared natural key is unique; `fact_token_usage` has one row per
     `api_message_id`.
   - Ground-truth scoring: where some rows state a value that others derive
     (sync delegations state the token rollup that async ones derive), score
     the derivation on the stated rows.
   - Coverage reconciliation against `etl.table_coverage`, which is also the
     allowlist, so an accepted exception is reviewable data rather than a
     deleted check.
   Exits nonzero on any unallowlisted finding. Calibrate against a
   full-corpus build, then delete that build.
8. **Reader's guide v1**, generated from the coverage tables.

Then: bump `pyproject.toml`, `CHANGELOG.md` and `PARSER_VERSION` together;
tag `v1.0.0`.

## Harnesses beyond Claude Code

Design and principles: `docs/HARNESS_ARCHITECTURE.md` (PROPOSED). Antigravity's
store map and measured claims: `docs/ANTIGRAVITY_CONTRACT.md`. This section is
the schedule and the concrete next steps.

The destination is one warehouse that answers the same questions about every
agent harness on the machine, without pretending they are the same thing. The
ordering argument: Tier 1 is per-harness and native, so a new harness can be
captured today without touching the warehouse; harness-neutrality belongs to
the shaped staging that 1.3.0 builds, so everything that reshapes facts waits
for it. Building phases 2 to 4 on today's `log_entries` staging would be
building the layer 1.3.0 replaces.

### Phase 0 -- the raw Antigravity lake. DONE 2026-09-22, EXPERIMENTAL

`ccutils lake antigravity` mirrors the native store into
`<lake_root>/antigravity/` byte for byte: every SQLite table of every
conversation, the summaries index, brain text files, media as inventory,
legacy `.pb` and `implicit/*.pb` as ciphertext. Nothing is decoded.
`parsers/lake.py` is the harness-generic runner; `parsers/antigravity/` is the
first `LakeSource`.

Measured on the first real run: 469 units, 0 errors, 8.4s; second run 1.5s,
everything unchanged, 0 files rewritten; 752 MB; 54,001 step payloads
sha256-identical to the source; a traced run opened 13,327 paths under the
data root and none outside the allowlist.

The command and the lake layout are outside semver until declared stable.
Declaring them stable is a decision for after phase 1, when a consumer exists.

### Phase 1 -- decode Antigravity. Unscheduled, no DDL, can land any time

The lake is opaque without this: everything interesting (text, thinking,
tool calls, tokens, models, timestamps) is inside protobuf blobs. The schema
is stated by the app's own binaries, so the decoder reads it rather than
hard-coding field numbers.

1. **Capture the schema into the lake.** Extract the `FileDescriptorSet` from
   the installed Go binary (rodata scan for `FileDescriptorProto` bytes, walk
   to each descriptor's end, keep the longest duplicate per name, check the
   dependency closure of `trajectory.proto`). Write it to
   `<lake_root>/antigravity/_schema/<app>-<version>/` with the manifest row
   that says which run captured it. Never committed: it is extracted from a
   proprietary binary, and the owner accepted that exposure for local use
   only (2026-09-22). Capturing it per app version is what keeps old blobs
   decodable after the app updates.
2. **Add the `protobuf` dependency** and build a descriptor pool from the
   captured set; no generated code in the repo.
3. **Decode into a derived layer**, `decoded/*.parquet` beside each unit's raw
   tables, with its own format version: `steps.step_payload`
   (`gemini_coder.Step`), `gen_metadata.data`, `raw_summary`,
   `trajectory_metadata_blob`, `executor_metadata`, `parent_references`, and
   the annotation `.pbtxt`. Derived from raw plus the captured schema, so it
   is re-derivable and never replaces the mirror.
4. **Read both tool encodings** (contract claim 11): per-tool step types, and
   step 132 `GENERIC`, which wraps the old step as an `Any`.
5. **Keep bytes as bytes.** Inline images and the one ~97 MB prompt-snapshot
   row per conversation must not become base64 in JSON.
6. **`antigravity_*` DuckDB views over the decoded layer**, marked unstable:
   steps with names and times, tool calls joined to their results on the
   stated call id, token usage, models, subagents.

Gates: an unknown-field walk over the whole corpus finds 0 (the survey's
number on 2.6M fields); decoded planner text equals the brain transcript's
content for joined lines within the measured rate; a decode failure is
recorded in the manifest, never a silent NULL.

### Phase 1b -- the 27 encrypted legacy conversations. Opt-in, after phase 1

20 hub and 7 CLI conversations (Nov 2025 to May 2026) are in the old encrypted
`.pb` format; the lake archives the ciphertext. Community tools report
AES-128-CTR with the first 16 bytes as the nonce and the key in the macOS
Keychain item `Antigravity Safe Storage`; unverified here.

- Verify the scheme against the local files first: does the plaintext decode
  as a `gemini_coder.Trajectory` with 0 unknown fields?
- `--decrypt-legacy`, macOS only. The key is read at runtime (the OS prompts),
  never stored, never logged, and its absence fails loudly rather than
  skipping. The ciphertext stays in the lake.
- Output goes into the same per-conversation tables with `record_source`
  `antigravity_legacy_pb`, so a decrypted conversation is queried like any
  other.
- Gate: each decrypted conversation's step count equals its summaries
  `step_count`, or the file is recorded as undecryptable with a reason.
- Fallback, only if decryption does not work: export through the running
  language server's local RPC. It needs the app running and a CSRF token from
  process arguments, so it is the second choice, not the first.

### Phase 2 -- harness-neutral staging. Rides 1.3.0

- Write the `stg_*` contract into `docs/ETL_ARCHITECTURE.md` as Tier 2's
  target: sessions, messages, tool calls, tool results, usage, delegations,
  artifacts, each carrying `harness`. Draft it from the measured shapes of at
  least two harnesses, ideally three with plain Gemini CLI, so it is not a
  generalization of one.
- Per-harness vocabulary maps, because facts encode Claude Code's today:
  `_AGENT_TOOL_NAMES`, `TOOL_CATEGORIES`, the `api_message_id` usage grain.
- **Claude Code becomes a `LakeSource`** (`ccutils lake claude_code`), its
  lake moves under `<lake_root>/claude_code/`, and staging reads the lake.
  That also makes true what `README.md` has claimed for a year: the warehouse
  can be rebuilt from Parquet without re-parsing JSONL. No code path does that
  today.
- Revise the `LakeSource` interface against that second real harness; it was
  designed against one and a test-only toy.
- Gate: a Claude Code warehouse rebuilt from the lake alone matches a build
  from JSONL, and both harnesses produce the same `stg_*` shapes.

### Phase 3 -- harness identity in the warehouse. Rides 1.3.0

- A `harness` column or dimension, and harness-namespaced keys in the
  key-formula module (a Claude Code `session_key` changes; that is a rebuild,
  not a migration).
- `record_source` carried from staging instead of `lineage_upsert`'s constant
  default. Of five allow-listed values, only two ever reach a row today.
- A per-harness parser version: one global `PARSER_VERSION` cannot say which
  contract wrote a row when two harnesses are in the file.
- A neutral project identity. Claude Code's project is the transcript
  directory; Antigravity states a workspace URI, a git root and a `project_id`
  named in `config/projects/*.json`.
- `dim_model` gains provider, and family comes from a structural parse of the
  stated id: the corpus holds `gemini-3.8-flash-tiered` and, through
  Antigravity, `claude-opus-4-6-thinking`.
- Gate: `ccutils audit` runs per harness and is clean; no key collides; no
  `record_source` is a constant.

### Phase 4 -- Antigravity populators. Rides 1.3.0

Neutral facts through the adapter: `PLANNER_RESPONSE` to messages and tool
calls; tool steps to results joined on the stated call id; `gen_metadata` to
usage; `INVOKE_SUBAGENT` plus the stated parent and depth to delegations;
brain artifacts, and later their `.git` snapshots, to plan revisions. Stated
values win: Antigravity states a command's exit code, where Claude Code's is
derived from output text.

Antigravity-only facts, which do not get forced into neutral columns:
checkpoints, agent-to-agent messages, forks and battle mode, browser
subagent.

Gate: audit clean per harness; delegation depth equals the stated
`nesting_depth` with no walking.

### Phase 5 -- render and privacy. After phase 4

HTML and markdown from neutral staging. `--no-thinking` must be wired and its
effect asserted on every new surface: Antigravity's thinking text is real,
unlike Claude Code's empty blocks, so the flag stops being a no-op that looks
fine. Anything built for a third party stays `include_thinking=False`.

### Further harnesses, unscheduled

Cowork's `audit.jsonl` (already on the list behind the rewrite), Claude.ai
exports (HTML-only today), plain Gemini CLI chats (`tmp/*/chats/*.json`,
found while surveying Antigravity). Each is a `LakeSource` plus a contract
doc; the third one is what proves the interface was not built around two.

## Known defects and open items

Found, verified against the code on 2026-09-10, not fixed. Grouped by where
they land.

### Fix at 1.0.0 or before

- **`docs/JSONL_CONTRACT.md` claim 5 has gone red.** On 2026-09-22
  `tests/test_jsonl_contract.py::TestThinkingTextIsAbsent::test_corpus_holds`
  failed on a clean `main`: 12 thinking blocks in its seeded sample now carry
  text. Either Claude Code has started persisting reasoning text or the rare
  case (54 of 50,268 corpus-wide when the claim was written) moved into the
  sample. Measure the rate corpus-wide, restate the claim as a rate, and
  revisit the "reasoning text is not on disk" rule in `CLAUDE.md` and the
  coverage layer's entry.

- **Continued sessions replay their history.** 706 `api_message_id`
  values recur across sessions in the same chain (1,403 rows), and the
  same is true of every replayed message and tool call. Nothing marks a
  replayed row, so per-project token sums double-count continuations.
  Belongs with 1.1.0 decomposition: a derived `is_replay` on the entry
  grain, computed from the chain, so aggregates can exclude it.
  Replay also happens inside a single file (measured 2026-09-18 on a
  320-session build of every non-temp project): 37 transcript uuids occur
  twice in the same session file, 1,036 to 4,017 lines apart, with identical
  timestamp and text (29 tool-result rows, 7 assistant, 1 meta; none typed
  by the user). So `(session_id, message_id)` is not unique either, and
  `is_replay` must be computed within a file as well as across a chain. The
  same build had 1,889 uuids spanning more than one session file (11
  sessions, 64 of them on typed messages).
- **F14 `human_message_count` counts tool-result entries.** The facet
  (`etl/fact_session_facets.py`, the F14 `_insert_facet`) counts every
  non-meta `message_type = 'user'` row, and tool results are user rows. In
  one session it reported 87 where 7 were typed; the other 80 matched that
  session's 80 tool calls exactly. Filter `has_tool_result` (and
  `is_compact_summary`). A value fix, not a schema change. Found 2026-09-18.

- **`tests/test_fact_token_usage_v15.py` needs a grain oracle** asserting
  `count(*) = count(DISTINCT api_message_id)`. `lineage_upsert` cannot catch
  a grain regression there because the declared natural key is `entry_id`.
- **`TaskCreate` sits in the agent-rollup CASE** in `etl/fact_tool_calls.py`
  (nine branches). Its results carry no agent payload; it is the task-list
  tool. Harmless but misleading, and nothing asserts the extraction list and
  `_AGENT_TOOL_NAMES` agree.
- **`--embed` ships partly dead output.** In `schemas/star/embeddings.py`,
  `embed_sessions` and `fact_session_embeddings` have no consumer and store a
  mean-pooled ColBERT vector, which discards the late interaction that is the
  reason to use ColBERT. `cluster_sessions` writes `cluster_N` into
  `dim_session.domain`, injecting a second vocabulary into a column of real
  labels. `match_delegations` in the same module is live and correct. Retire
  the first two, delete the third, keep the last.
- **`semantic_cost_analysis.cache_hit_rate_pct` discriminates nothing.** It
  reads near 100% for every project because `input_tokens` is
  post-breakpoint by construction. Redefine or drop.
- **`fact_agent_delegations.agent_total_tool_use_count` is NULL, not 0, when
  an agent used no tools** (LEFT JOIN over `fact_tool_uses`), conflating
  "used none" with "unknown".

### 1.1.0

- **Bash contributes nothing to file tracking.** `_FILE_TOOL_OPS` in
  `etl/fact_file_operations.py` covers the file tools only. In a repo whose
  house style reads and writes through the shell, that is most of the file
  activity. Inferred rows need a `derived_from` marker so they never mix
  with tool-level truth.
- **`fact_tool_input_params` is empty while `input_json` is fully
  populated.** Populate from staging or cull with the other stubs.
- **`tool-results/` sidecar files are unknown to the pipeline.** When a tool
  output is too large to inline, Claude Code writes the full result to
  `<project>/<session-uuid>/tool-results/<slug>.txt` and the transcript keeps
  a preview plus a dangling path. Index them (path, size, type, join on
  `tool_use_id` where the file is named by it); never ingest the bodies.
  Decide privacy, grain and retention before building.
- **CDC watermarks must consider the `subagents/` directory.** A parent
  transcript can be unchanged while its agent files grow, so a parent-only
  mtime/size check misses completions indefinitely.
- **`find_all_sessions` makes two full passes per file** and reads to EOF
  when `cwd` is absent. Correct, slow at corpus scale. Fold into one pass
  when discovery time matters.

### Behind the rewrite or conditional

- **DDL stubs.** `fact_content_blocks`, `fact_code_blocks`,
  `fact_entity_mentions`, `fact_tool_input_params`, `fact_facet_embeddings`
  have no populator; `fact_turn_durations` and `fact_stop_events` are
  subsumed by `fact_system_events`; `fact_tool_calls` by `fact_tool_uses` +
  `fact_tool_results`. Populate or cull each; the coverage layer records the
  decision.
- **Agent identity residue.** A handful of agent sessions carry a sidecar
  `agentType` yet land `agent_type IS NULL` (arithmetic on the corpus left
  roughly 42 unexplained). `fork` delegations are not distinguished from
  named subagents, and some delegations carry NULL `subagent_type`.
  Pre-2026 seven-hex agent ids can collide across parents; composite
  identity only if those sessions matter analytically.
- **Every `bridgeSessionId` target is absent from `dim_session` and the
  lake.** They point outside the ingested corpus entirely. Unexplained.
- **Temp-dir exclusion gap.** Sessions whose only entry type is `ai-title`
  never carry `cwd`, so `is_temp_dir_cwd` cannot exclude them. Root cause:
  `extract_session_metadata` prefilters lines on `"sessionId"`, so a line
  carrying `cwd` without it is never examined.
- **A third agent layout may exist upstream.** An external audit described
  sidechain transcripts at `tasks/*.output`. This pipeline knows
  `<uuid>/subagents/agent-*.jsonl` and `subagents/workflows/<wf-id>/`.
  Establish whether that layout is real before building against it.
- **Facet Tier 2 has never run at scale.** F20 has zero rows corpus-wide, so
  there is no quality evidence for the description text that Tier 3
  clustering would embed. Run F20 across the corpus and read the output
  before expanding `FACET_SPECS`.
- **Tier 2 inputs are weaker than they look.** Measured 2026-09-18:
  `SessionInputs.last_assistant_message` (`etl/facets/populator.py`,
  `_build_session_inputs`) takes the last assistant entry whether or not it
  carries text, so it is empty when a session ends on a tool call (11 of 50
  sampled sessions). Take the last assistant entry that has text, and check
  whether `dim_session.last_assistant_message` shares the bug. Separately,
  the "first user message" is a slash-command wrapper in 146 of 320
  sessions and a paste longer than the 800-character cut in 73, so F20's
  main input is often harness text rather than the user's request. And the
  `first_user` query skips `is_meta` rows only: compaction summaries
  (Claude-written, able to quote tool output) are user rows too, and none of
  19 measured carries `isMeta`. 1 of 320 sessions opens with one, and 5 open
  with a tool result, giving an empty first message. Filter
  `is_compact_summary`, and skip tool-result rows when picking the first
  message.
- **`--private` is best-effort on render formats only.** Known channels:
  message text, thinking, raw non-message entries, tool_use keys beyond the
  allowlist, list-form tool results, the batch search index, index and
  project-name surfaces, pasted foreign paths. Superseded in part by the
  scoping rule (a shared artifact is scoped by project, not scrubbed).
- **Behavior classifiers are untrustworthy.** Keyword `outcome` labels do not
  discriminate (sessions labeled `failure` committed cleanly about as often
  as `success`). Demote `outcome` to `keyword_outcome`, emit measured
  booleans (`had_clean_commit`, `ended_on_tool_error`) as Tier 1 facets
  F32/F33, and decide whether `fact_agentic_runs` is materialized before any
  Tier 2 run-grain facet.

### 1.2.0: how agents consume the warehouse

Added 2026-09-10 at the owner's request. The primary consumer of ccutils
output is a Claude Code agent (or another agent), not a person at a SQL
prompt: the warehouse is a context database spanning projects, machines
and sessions. Explore, then decide, the best surfaces for that:

- **The reader's guide** (1.0.0 step 8) is the first piece: a generated
  markdown file beside the archive saying what this warehouse holds, what
  each table means, the join paths, and the questions it can answer, so an
  agent opening it cold does not have to reverse-engineer the schema.
- **Candidates to evaluate against real agent use:** a `ccutils report`
  that emits markdown answers to the declared questions (1.2.0); per-project
  markdown digests an agent can be pointed at; a skill or MCP-shaped query
  surface with the routing table and gotchas built in; templates a
  downstream repo can fill from `semantic_*` views. Measure by whether an
  agent given the surface answers a cross-session question correctly
  without hand-written SQL.
- **Constraints already decided:** output is scoped per project, never
  git-tracked, lives under the home-anchored archive directory; nothing is
  scrubbed after the fact.

### 1.2.0: commit attribution, as specified by a downstream consumer

Filed 2026-09-10 by a sibling research project that uses the warehouse as
evidence for "which session made this commit". Git cannot answer it: every
commit carries the same author, and commit trailers naming sessions are
out for public repos. The transcript is the only record, so the join lives
here and stays local.

- **`fact_commits` from transcripts alone.** One row per git commit
  invocation: session key (subagents included), tool use id, timestamp,
  subject, success, is_amend. Parse the command, do not grep: sessions
  commit as `git -c commit.gpgsign=false commit -F - <<'EOF'` with the
  subject on the heredoc's first line, so a naive match found 13 of about
  200 calls. Hash from the result when present (`-q` suppresses it). Flag
  pushes and force-pushes (`--force-with-lease` appears).
- **Optional `--git-repo <path>`.** Read that clone's reflog and log and
  bind each call to the hash it produced: subject match plus a small time
  window (author time at or just after the call; warehouse times are UTC,
  git's are local with offset). Reflog also yields amends, commits an amend
  replaced, and pushes no loaded transcript contains. An amended commit
  keeps its original author timestamp, so anchor on the last successful
  call that produced the landed hash, not the first.
- **Acceptance test with a known answer.** One day, one branch, 78 commits
  from 7 parallel sessions plus 19 subagent transcripts: per-session counts
  24/15/15/9/7/6/2, zero unattributed, zero double-claimed; one commit
  pushed by no loaded transcript, amended twice, then replaced on origin by
  a force-push. The consumer holds the fixture.
- **Depends on** the derived failure kind on tool calls (a hook-blocked
  commit must read as blocked, not as NULL).

### Open design questions

- **More than one harness.** What a project is across harnesses (Claude
  Code's transcript directory vs Antigravity's workspace URI, git root and
  `project_id`); whether `fact_messages` splits into neutral and
  harness-specific parts; how Antigravity forks and battle mode map; whether
  the lake archive should ever honour in-app deletions. Context in
  `docs/HARNESS_ARCHITECTURE.md`.

- **Agent rollup provenance: DECIDED 2026-09-10.** `fact_tool_results` is
  the only home for the stated rollup (`status`, `totalDurationMs`,
  `totalTokens`, `totalToolUseCount`, `wasInterrupted`, `resolvedModel`).
  `fact_agent_delegations` stops copying those columns and keeps only the
  spawn-side identity (tool use id, parent and agent keys, timestamps, task
  prompt, subagent type, `agent_is_async`). Every outcome value is derived
  from the agent's own transcript under a `derived_` name, in a view rather
  than the post-loop reconciliation pass, so it cannot go stale or be skipped
  by one entry point. Consumers wanting the API's stated number join back to
  `fact_tool_results`. Today `agent_total_duration_ms` and
  `agent_total_tool_use_count` hold a stated value on sync rows and a derived
  one on async rows under one name; the split ends that. Known cost to
  measure first: sync delegations whose agent transcript was pruned lose
  their outcome numbers, which `ccutils audit` should report as coverage.
  Lands with the 1.3.0 rewrite; `semantic_agent_delegations` becomes the
  derived view instead of being deleted.
- **TypeSafe Jev for Tier 2 enum facets: EXPLORING, decide before the
  1.0.0 tag.** Jev (a hosted classifier: choice / score / yes-no questions
  over one state, returning per-option probabilities, no text generation)
  fits the designed-but-unbuilt enum facets F22-F25, F28, F29; not the text
  facets (F20, F21, F26, F27) and not Tier 3. For a batch ETL its speed and
  price decide nothing; what it adds over the Haiku path is probabilities.
  Owner decisions so far: opt-in flag like `--llm-facets`; if a test run
  earns it, `confidence` and `probabilities_json` become real columns on
  `fact_session_facets` before the tag (the metadata JSON is excluded from
  `hash_cols`, so a confidence change there would never update a row; after
  the tag a new column forces a rebuild); call the HTTP API directly, not
  the young SDK; send only scrubbed excerpts from sessions and fields the
  owner selects. Scrubbing by listing bad shapes failed an adversarial
  check (about 20 surviving shape classes, plus a username scrub that no-opped
  when `$USER` was unset). The harness now gates with a token allowlist: only
  plain words, short numbers and punctuation pass, and everything else
  becomes a category placeholder. It also has an independent residue check
  that blocks the session, and credential and identity checks that fail
  closed. Against an independently written probe file
  (`internal/egress_probes/`, 254 probes) it passes 252. The 2 misses are
  deliberate over-scrubs. Those probes then informed rule changes, so a fresh
  author must write new ones before that file counts as independent again. A
  Jev extractor must build its inputs with
  `include_thinking=False` whatever `--no-thinking` says:
  `_build_session_inputs` passes the flag to `extract_text_from_content_json`,
  whose default concatenates thinking blocks into the message text. The test
  harness is in untracked `internal/jev_spike/`.
  It has not run: no key yet. Claude-written labels are evaluation data
  only, never training data for another vendor's model.
- **`recursive=` on `find_agent_sessions` is a documented no-op.** If a real
  depth selector is ever wanted, build it from the sidecar's stated
  `spawnDepth`.
- **Corrections owed to `docs/ETL_ARCHITECTURE.md`**, to land with the items
  that touch them: the reconciliation pass addresses a temporal dependency,
  not an extraction one, so it does not dissolve when extraction moves to
  staging; and the verification section names baseline assets that no
  longer exist and, by decision, will not be rebuilt.

### Antigravity lake: known gaps

Found while building phase 0 (2026-09-22), not fixed. None blocks the lake.

- **The lake has no self-check.** The manifest records what was written, and
  the contract canaries check the SOURCE, but nothing verifies the lake
  against the source after the fact the way `ccutils audit` checks the
  warehouse. A `lake --check` re-reading a sample of blobs and comparing
  sha256 would close it.
- **`brain/<id>/.git` is excluded**, so the per-conversation snapshot history
  (up to 769 commits, which is artifact revision history and earlier versions
  of the transcripts) is not archived. Phase 4 extracts it. Until then a
  conversation deleted in the app loses that history even though its current
  files are kept.
- **`implicit/*.pb` is archived without knowing what it is.** 46 encrypted
  files whose uuids match no conversation; the language server prunes them.
  Phase 1b may reveal them, or they stay a labelled unknown.
- **`antigravity-ide` duplicates 19 of the hub's legacy conversations.** It is
  ingested because it is the IDE app's live data-dir name, so it will hold
  real conversations if that app is used again; today it costs a second copy
  of 19 ciphertext files and their brain files.
- **Plain Gemini CLI chats are not ingested** (`tmp/*/chats/*.json`). A
  separate harness, found while surveying; listed under further harnesses.
- **One conversation's `cascade_id` is not its filename.** An empty database
  in the hub store carries an id found nowhere else on disk; the contract
  claim exempts empty files rather than explaining them.
- **The `.pb` encryption scheme is unverified here** -- community-reported
  only, and verifying it needs a Keychain credential. Phase 1b verifies
  before it decrypts.

### Environment and tooling

- `gh` fails with a TLS certificate error in the sandbox, so PR text on the
  remote is unverified from a session.
- The live-API smoke test needs a valid key in the `ccutils-anthropic`
  keychain entry; a 401 there is not a regression.
- `uv run` needs the sandbox disabled for its cache; calling the project
  venv directly (`.venv/bin/python -m pytest tests/ --confcutdir=tests`)
  runs inside it.
- Commits are signed through a 1Password agent whose socket is outside the
  sandbox, so `git commit` needs the sandbox off for that one command.

## What the test suite cannot see

Stated once so green is read correctly:

- **Corpus scale.** Fixtures are kilobytes; the corpus is gigabytes. Every
  structural bug this project has shipped was found by querying a real
  full-corpus build and asking whether the numbers were plausible. Re-run
  that after any ETL change (`uv run ccutils --source --format duckdb -o
  <scratch-dir>`), and delete the build afterwards.
- **Machine dependence.** Several tests skip when the local Claude data
  directory is absent, so a fresh machine reports green with fewer tests.
- **Concurrency.** Nothing tests two writers against one warehouse, and
  `ccutils --source -j N` writes in parallel.
- **One backend, one platform.** DuckDB, macOS, one Python.

Convention going forward: every new test carries one line naming what breaks
if it is deleted.

## Decisions that should not be relitigated

- **No migrations from the past unless the owner asks for one.** Rebuild
  instead. 1.0.0 deletes the machinery.
- **No baseline warehouse is built or pinned.** Current output is not
  trustworthy enough to be a reference; the rewrite is gated on a report
  diff. A calibration build for `ccutils audit` is made and deleted.
- **A shared artifact is scoped, not scrubbed.** Output is generated for
  named projects only. `--private` is a secondary pass inside that scope,
  never the boundary.
- **Hard CLI break, no aliases.** Removed names exit 2 with a pointer.
- **A `SubagentStop` hook writing its own telemetry is rejected.** Every
  delegation value is recoverable from files at rest.
- **The same value is stored once.** Stated values live where they are
  stated (`fact_tool_results`); derived values carry a `derived_` name and
  live in views. A fact never copies another fact's columns.
- **Tier 1 is per-harness and native; neutrality starts at Tier 2.** A
  second harness is mirrored at its own grain, byte-faithful, and is not
  reshaped into another harness's envelope. Decided 2026-09-22 while
  building the Antigravity lake, against the alternative of extending
  `etl.log_entries`.
- **A harness lake whose Tier 0 is not durable is an archive, not a cache.**
  It keeps units whose source disappeared, supersedes rather than overwrites,
  and lives apart from warehouse output dirs so that rebuilding a warehouse
  cannot delete it.
- **`TaskCreate` is not an agent spawn.** `Agent` is the only tool name in
  the corpus carrying an agent rollup; `Task` is kept for older transcripts.

## Local-only references

Present only on the machine that wrote them; nothing above requires them.

- `internal/plans/etl_layer_rewrite.md`: per-item specs, estimates and gates
  for the 1.3.0 rewrite (populator disposition table, key-formula module,
  shaped staging, `--no-thinking` at staging load, step-grain CDC, view
  demotions).
- `internal/plans/post_018_roadmap.md`: the evidence and corpus numbers
  behind each item above.
- `internal/plans/test_audit_2026-08-28.md`: per-test verdicts and the
  mutation results.
- `internal/plans/behavior_analytics.md`, `private_hardening.md`,
  `2026-08-01_agent_delegation_capture_gap.md`: the longer plans for the
  items of the same names.
- `internal/log/`: per-session logs through 2026-08-05.
