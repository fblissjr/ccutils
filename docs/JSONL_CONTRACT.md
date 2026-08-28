# The Claude Code JSONL contract

Last updated: 2026-08-28

What this pipeline assumes about the transcript format it reads. Claude Code
ships continuously and this format is not versioned or documented upstream, so
every assumption here is a bet. The bets are written down so that when one
stops being true, the failure is a named canary going red rather than a number
looking wrong to somebody months later.

**Every claim carries the measurement that supports it.** A claim without one is
marked as such and should be treated as a guess. Re-run the measurements rather
than trusting the numbers; they were taken on one machine's corpus at one time.

Corpus at time of writing: 4,783 `.jsonl` files, 1.87 GB, under the projects
directory. Samples below are random draws from it, seeded so they reproduce.

This document exists because the alternative failed repeatedly. Four structural
bugs shipped from unverified format assumptions, and two external audits of the
resulting warehouse each drew a wrong conclusion in a different place — one
reported that prose was being dropped from 80% of assistant messages, which is
claim 2 below misread. An auditor with no contract to check against reasons from
whatever seems plausible.

---

## 1. Identity comes from the file, never from `sessionId`

A session's identity is its file stem. Agent transcripts carry the **parent's**
`sessionId` on every line, so that field cannot distinguish one agent from
another, or an agent from its parent.

- Measured: 300 random agent transcripts, every one carries a `sessionId` equal
  to the parent directory's uuid. Zero exceptions, zero absent.
- Consequence: an agent's `session_id` is `'agent-' || agent_id`, stamped at
  Tier 1 in `parquet_writer` and re-enforced at staging load.
- Canary: `tests/test_typed_parser.py` and the `entry_session_id` parameter in
  `tests/helpers_ccutils.py::write_minimal_session`, which exists so synthetic
  fixtures model this rather than the convenient fiction.
- If this changes: everything keyed on agent identity silently merges or splits.

## 2. An assistant entry carries one content block, with a measured long tail

Text, thinking, and tool calls arrive as **separate entries** in the
overwhelming majority of cases. Parallel tool calls are separate entries.
A small number of entries do carry several blocks, including mixed kinds.

- Measured **corpus-wide** (not sampled): **17 violations in 198,230**
  assistant entries with list content -- 0.0086%. Shapes seen include
  `[text, thinking, tool_use]`, `[text, thinking]`, `[thinking, tool_use]`
  and `[tool_use, tool_use]`.
- **This claim was originally written as an absolute**, from a 120-file
  sample in which zero violations appear. Its own canary caught it hours
  later on a different draw. The correction is recorded rather than quietly
  applied, because the lesson is about method: a rare-event claim cannot be
  established by a sample that is smaller than the event's inverse rate. A
  0.009% event is invisible in 120 files roughly always.
- Consequence, unchanged in substance: `fact_messages.content_text` being
  NULL on a `has_tool_use` row is the format, not a loss -- those entries
  genuinely have no text block. The external audit that read this as an 80%
  data-loss bug is still wrong about the magnitude. But the margin is
  0.009%, not zero.
- Consequence, verified for the tail: the SQL projection handles multi-block
  entries correctly. An entry containing `[text, thinking, tool_use, text]`
  yields both prose blocks in `content_text` (space-joined), with
  `content_block_count` = 4 and both flags set. Nothing is dropped.
- Canary: `tests/test_jsonl_contract.py::TestAssistantBlockShape`, which
  asserts the violation RATE stays under 0.1% (~10x headroom) scanning the
  whole corpus, plus a deterministic test of the requirement that actually
  matters -- every text block is captured however many there are.
- If the rate climbs: `fact_messages`' one-row-per-entry grain needs
  revisiting, and so does the reading above.

## 3. One API response spans several entries sharing `message.id`

Each entry of a response repeats the response's `usage` block.

- Measured: same sample, 1,810 distinct `message.id` values, 1,476 of them
  spanning more than one entry uuid; the widest response spans 12 entries.
- Consequence: `fact_token_usage` is one row per API **response**, deduplicated
  on `api_message_id`, not one row per entry. Summing usage per entry
  multiplies every response's cost by its entry count.
- Canary: the R23 grain assertions in the token-usage tests.
- If this changes: token and cost figures inflate silently.

## 4. `is_error` lives on the tool_result block, and absent means success

Tool results come back as **user** entries whose content contains a
`tool_result` block.

- Measured: same sample, 1,307 `is_error` occurrences, all at
  `message.content[].tool_result.is_error`; values `false` 1,239, `true` 68.
  None on `toolUseResult`.
- Consequence: NULL means not-an-error. A predicate must not treat absence as
  unknown, or accepted plans get classified `unknown`.
- Canary: `tests/test_typed_parser.py::TestIsErrorAbsenceContract`.
- If this changes: error rates and outcome classification invert quietly.

## 5. Thinking text is not persisted

Thinking blocks carry an empty `thinking` string plus a signature.

- Measured corpus-wide: 50,214 blocks with `"thinking": ""` against 54 with
  content (0.1%, presumed an older format).
- Consequence: `has_thinking` is a flag over content the source never wrote.
  Reasoning cannot be recovered from these transcripts by any means, so a
  populator for it would be empty by construction. This is recorded in the
  warehouse's coverage layer so a reader does not conclude the pipeline
  discarded it.
- Canary: `tests/test_jsonl_contract.py::TestThinkingTextIsAbsent`. This is the
  claim most likely to go red for a good reason -- if Claude Code starts
  persisting reasoning, nothing else in the pipeline would notice and the
  coverage layer would keep telling readers the data does not exist.

## 6. Subagent transcripts live below their parent, in two layouts

```
<project>/<parent-uuid>.jsonl
<project>/<parent-uuid>/subagents/agent-<id>.jsonl
<project>/<parent-uuid>/subagents/workflows/<wf-id>/agent-<id>.jsonl
```

- Measured: 2,499 agent transcripts, **zero** outside a `subagents/` directory;
  2,465 at the first depth and 34 at the workflow depth. A `journal.jsonl` sits
  beside the workflow agents and is not a transcript.
- Consequence: never assume `subagents` is the last directory. Search below it,
  and cut the project boundary at the `subagents` segment rather than at the
  file's parent. The boundary rule lives in exactly two mirrored places,
  `etl/utils.py::project_dir_sql` and the walk-up in
  `parsers/discovery.py::find_all_sessions`.
- Canary: `tests/test_all.py::TestProjectRuleEquivalence`, which now carries an
  expected-value oracle alongside the equivalence check — the two
  implementations agreed with each other while both were wrong about the
  workflow layout, and agreement alone could never have caught it.
- If this changes: subagents vanish from exports, or a synthetic project appears
  named after a grouping directory.

## 7. Nesting is stated in the sidecar, never derivable from the files

Nested agents are **flat siblings** in the same `subagents/` directory, all
carrying the root's `sessionId`. Depth and parentage exist only in the optional
`agent-<id>.meta.json`.

- Measured: 1,540 sidecars. Keys present — `agentType` 1,540, `description`
  1,267, `spawnDepth` 1,213, `toolUseId` 1,146, `model` 313, `parentAgentId`
  235, `isFork` 71, `name` 70, plus `worktreePath`, `worktreeBranch`,
  `stoppedByUser`. Of 241 agents with `spawnDepth > 1`, 235 carry
  `parentAgentId`; all 235 resolve to a sidecar in the same directory, all to a
  parent whose `spawnDepth` is exactly one less, and no depth-1 agent carries
  one.
- Consequence: deriving depth by walking `sessionId` structurally cannot produce
  a value above 1, because every agent inherits the root's. Read the sidecar.
- Canary: **none yet.** Lands with the sidecar work.

## 8. An agent's task prompt is an `isMeta` user entry

The delegated task arrives flagged `isMeta`, which on a parent session marks
harness injections worth skipping.

- Measured: 400-file random sample of agent transcripts; 23 summarised to
  `(no summary)` under the old isMeta skip, 17 of which recover when agent files
  accept it. The remaining 6 open with an assistant entry or a
  `fork-context-ref`.
- Consequence: skipping isMeta on an agent file discards its only descriptive
  line, and render-format curation then drops the transcript entirely.
- Canary: `tests/test_agent_discovery.py::TestAgentTranscriptSummary`.

## 9. Entry types the pipeline encounters

From the same 120-file sample, by frequency: `assistant`, `user`, `attachment`,
`progress`, `last-prompt`, `mode`, `permission-mode`, `ai-title`,
`bridge-session`, `queue-operation`, `system`, `file-history-snapshot`. Others
appear rarely, including `summary`, `fork-context-ref` and `journal`.

`advisor_tool_result` appears as a content block kind and has no handler
anywhere in `src/`. That is a known gap, not a decision.

- Canary: `tests/test_typed_parser.py::TestArchiveCoveragePostA2`.
- If a new type appears: it lands in staging and is ignored by every populator,
  silently. The coverage layer is where that becomes visible.

## 10. Surrogate key formulas

- `session_key = md5(session_id)`
- an agent's `session_id = 'agent-' || agent_id`, so
  `agent_session_key = md5('agent-' || agent_id)` — derived, never joined,
  because a join depends on ETL ordering and the parent is normally processed
  first.
- `project_key = md5(project_dir)`, where `project_dir` is the boundary rule in
  claim 6.

`NATURAL_KEYS` in `schemas/star/schema.py` is the single source of truth for
table-to-key, guarded by a drift test.

---

## Adding to this document

A claim earns a place here when a populator would be wrong if it stopped being
true. State the claim, the measurement that supports it with enough detail to
re-run, the canary that guards it, and what breaks if it changes. "No canary
yet" is an acceptable and useful entry — the point of writing the claims down is
to make the missing canaries obvious.
