# Unified analysis: the plan

Status: **PLANNED, not started.** Written 2026-11-20 on the owner's direction.
Nothing in this document is built. It says what the phase is, what it must
answer, what it produces, and what waits for it.

## Why this comes before any more building

Three statements from the owner, 2026-11-20:

1. Do not build the harnesses in isolation.
2. A unified analysis phase comes first: both harnesses, both machines,
   measured together, before a model is drawn.
3. The schema that comes out can be net new. It does not have to fit the
   star schema, and nothing of the current one is carried over unless the
   analysis shows it is right.

The same day supplied the evidence. Each model drawn from one corpus was
wrong on the next one:

- The decoder's step-type names were taken from the Linux CLI store and left
  a second machine's older steps unnamed (`CHANGELOG.md`, "names the older
  per-tool step types").
- `antigravity_tool_calls` read the one tool-call encoding that store held
  and missed every call in the other (`docs/ANTIGRAVITY_CONTRACT.md` claim
  18).
- A project key built on the workspace path matched two projects across the
  two machines; the repository's name matches five, and the owner's own
  example was not among the two (claim 20).

Each was caught only because a second corpus arrived. A bridge between
today's Claude Code warehouse and today's Antigravity views would be the
same mistake one level up: two models, each drawn alone, joined afterwards.

## What the phase is

Measurement and design. It reads what each harness states, side by side, on
the same work, and produces a correspondence, a schema proposal and the
decisions the owner has to make. It ships no table, view or command.

It is not a bridge view, not an extension of the star schema, and not a
decision about releases. Those follow from its output.

## What is fixed and what is open

Fixed, because they are facts about the sources or rules that exist for a
reason:

- The two contracts, `docs/JSONL_CONTRACT.md` and
  `docs/ANTIGRAVITY_CONTRACT.md`. They describe upstream, not our schema,
  and every claim in them cost a defect to learn. The new schema is checked
  against them, and the phase adds to them.
- The Antigravity lake as an archive, its read allowlist, and one lake base
  with a `machine=<name>` directory per machine (`docs/ROADMAP.md`,
  "Decisions that should not be relitigated").
- The privacy rules in `CLAUDE.md`: a shared artifact is scoped, anything
  that leaves the machine passes an allowlist, `--no-thinking` is enforced
  on every surface.
- The vision's eight requirements (`docs/ROADMAP.md`, "Vision"). They are
  what the proposal is judged by.

Open, and decided by this phase's output and then the owner:

- Every table and view of the current star schema, and the Antigravity
  views built on 2026-11-20. Either can be kept, changed or dropped.
- Whether Claude Code is archived into the same machine-partitioned lake
  (phase 2 of the harness track planned `ccutils lake claude_code`).
- The form of the unified layer: tables an ETL run builds, as the Claude
  Code warehouse is today, or derived tables with views over the archive,
  as the Antigravity side is today, or a mix. Both exist in this repository
  and neither was chosen with the other in view.
- Where harness-neutrality starts. The standing decision is Tier 1 native
  and neutral from Tier 2. The phase either confirms it with evidence from
  two harnesses or says what replaces it.
- The release map. Phases 2 to 4 of the harness track were scheduled to
  ride the 1.3.0 rewrite of the current schema. If the schema is net new,
  that schedule is one of the things the output restates.

## Inputs

- **Antigravity, both machines.** Already in one lake base on the Linux
  machine, decoded, with `machine` on every row.
- **Claude Code, both machines.** The Linux machine's transcripts are local.
  The Mac's are not yet anywhere the analysis can read; getting them there
  (measured in place, or archived and copied like the Antigravity lake) is
  the first step and the owner's call, since raw transcripts are private.
- **Claude Code is read from the transcripts, not from the star schema.**
  The point is to see what the source states. The current warehouse is
  useful as a checklist of what was already learned (replayed calls, fork
  prefixes, agent identity, absent payloads), and every item on that
  checklist is in `docs/JSONL_CONTRACT.md` or `CLAUDE.md`.
- **A test bed.** One repository the owner worked on in both harnesses on
  the Linux machine and in Antigravity on both machines (the example of
  contract claim 20). Scoping to it keeps every query small and gives a
  ground truth outside both harnesses: the repository's own git history.
  A second, smaller repository checks that nothing was fitted to the first.

## The correspondence to fill

One row per concept. For each harness: where it is stated, at what grain,
what identifies it, what is derived and not stated, what is lost and when,
the measured rate, and the canary that would notice a change. The known
asymmetries are listed so the phase starts from them.

| Concept | Claude Code | Antigravity | What has to be settled |
|---|---|---|---|
| Machine | not stated in a transcript | not stated; a directory of the lake | one rule for both |
| Project | a transcript directory named after the launch directory; `cwd` and a branch per entry; no remote. One repository's work is spread over several directories (the repository, a subdirectory, worktrees, scratch directories under the temp dir) | workspace URI, git root, remote, `owner/repo` and branch per conversation (claim 20); `project` is the repository's name | whether the name can be derived for Claude Code from the directory alone, how the several directories roll up, and whether reading a remote from the live repository is acceptable |
| Unit of work | a session file; resumed and backgrounded sessions replay history; forks state a cut | a conversation; forks replay a prefix and name its owner; reverts re-use indices | whether "session" and "conversation" are one entity, and what a chain of resumed sessions is |
| Subagent | an agent transcript under the parent, identity from the file, a sidecar for depth and parent | a child conversation with a stated parent and depth, messages between agents, subagent records | one delegation entity: the call that started it, the child, how it ended |
| Turn | user entries, some of them tool results and some meta | `USER_INPUT` steps, some of them artifact approvals; system messages in between | a definition of a turn that holds in both, measured |
| Event | an entry holding content blocks: text, thinking, several tool uses; results arrive in a later user entry | a step: a response, or one tool call with its result | the common grain. An entry and a step are not the same size |
| Tool call | a `tool_use` block and its `tool_result`; ids replay across sessions | a step stating the call; ids repeat; two encodings | identity, and a shared vocabulary: the tools have different names and argument shapes for the same act |
| Files touched | `fact_file_operations` from tool inputs | paths inside tool arguments, undecoded bodies for older calls | one file-touch shape, and how much of each harness it covers |
| Commits | git commands run through the shell tool; nothing stated | the same; brain `.git` snapshots are something else | whether commits can be tied at all, measured against the repository's own log for the same days |
| Background work | background shell and its output | tasks by step index (claim 19) | one task entity or two |
| Usage and model | per assistant message, model id as a string | per response step plus checkpoints; the model as a number, its name in `gen_metadata` undecoded | grain and naming |
| Thinking | on disk for a minority of blocks (JSONL claim 5) | text absent for a large share of responses that report thinking tokens | what is promised to a reader |
| Plans and artifacts | plan mode, memory files | artifacts with versions and approvals | whether these are one thing |
| Time | per entry, some entries without | per step, plus index times that can be unset | one ordering across harnesses for "what happened on this project that day" |
| What disappears | to be measured: what Claude Code removes and when | pruning by count, emptying in place (claims 13 and 16) | what the archive has to capture and how often |

## Rules of the phase

- Whole corpus, both machines, never a sample for a rare-event claim. A
  claim is a rate with its date and the command that produced it.
- Nothing is built to ship. Measurement scripts live under the gitignored
  `internal/`; what they find goes into the two contract documents and the
  phase's own findings document, which are tracked.
- Every correspondence either gets a canary or says it has none.
- Reports describe shapes and counts. Transcript text, repository names,
  hostnames and addresses stay out of anything tracked.
- A finding that contradicts a contract claim corrects the claim in the same
  change.

## What it produces

1. **The filled correspondence**, as a tracked document, with measurements.
2. **A schema proposal**, net new: the entities, their keys (each on its
   whole identity), the levels and the key from each level into the next,
   what is stored and what is a view, and for every object the thing a
   reader would get wrong without it. It states which existing objects
   survive and why.
3. **The form of the build**: what an "ETL build" is in the new design,
   what it reads, and what makes it repeatable on two machines.
4. **The decisions for the owner**, each with the evidence and a
   recommendation.
5. **A restated roadmap**: what happens to the 1.0.0 gate list, the harness
   phases and the release map.
6. **An acceptance test for the vision.** A fixed set of questions about
   the test-bed repository, from "what happened on this day" down to "which
   command produced this file and what did it exit with", each with an
   answer established from the git history and the raw transcripts. The
   schema is done when an agent given only the built data and its generated
   guide answers them. This is the one part that becomes a permanent test.

## Order of work

1. Get the Mac's Claude Code transcripts where the analysis can read them.
2. Read the test-bed repository's Claude Code transcripts as they are, and
   list what each entry type states, against `docs/JSONL_CONTRACT.md`.
3. Fill the correspondence, concept by concept, test bed first, then the
   whole corpus for every rate.
4. Write the acceptance questions and establish their answers.
5. Draft the schema proposal and check it against the eight requirements,
   the two contracts and the acceptance questions.
6. Decisions to the owner; restate the roadmap.

## What waits, and what does not

Waits for this phase:

- Any object that joins or unions the two harnesses.
- New levels on the Antigravity side that both harnesses have to share:
  turns, a file level, a commit tie, a delegation view, a generated guide.
- Identity changes to the Claude Code warehouse.
- Further 1.0.0 gate work that only makes sense if the current schema
  survives. Which items those are is an early question for the phase.

Does not wait:

- The archive. Running the lake, its correctness, and anything that stops
  data being lost.
- Decoding more of what Antigravity states into the derived tables (the
  older tool-call bodies, model names, error details). That is raw material
  for the analysis and commits to no model.
- Defects in what exists.

## Where the analysis starts from

Already measured, and where each is recorded:

- The Antigravity field map, tool calls, tasks and workspaces:
  `docs/ANTIGRAVITY_CONTRACT.md` claims 17 to 20.
- What the Antigravity views leave NULL and why: `docs/ROADMAP.md`,
  "Antigravity lake: known gaps".
- The eight requirements and where each stands: `docs/ROADMAP.md`, "Vision".
- What Claude Code's format guarantees: `docs/JSONL_CONTRACT.md`.
- The earlier sketch of neutral staging, drawn before either harness was
  measured against the other: `docs/HARNESS_ARCHITECTURE.md`, "Neutral
  staging, sketch". It is an input, not a commitment.
