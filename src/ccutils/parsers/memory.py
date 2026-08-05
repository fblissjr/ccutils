"""Parser for Claude Code auto memory -- markdown notes Claude writes itself.

Two documented locations, both plain markdown with optional YAML frontmatter:

- **Project scope**: ``<HOME>/.claude/projects/<project>/memory/``  # path-privacy: ignore
  One directory per repository (keyed by git repo root, so worktrees share
  one). ``MEMORY.md`` is the index loaded every session; sibling ``*.md``
  files are topic notes read on demand.
- **Subagent scope**: ``<HOME>/.claude/agent-memory/<agent>/`` (user),  # path-privacy: ignore
  ``<repo>/.claude/agent-memory/<agent>/`` (committed) and
  ``<repo>/.claude/agent-memory-local/<agent>/`` (gitignored). Subagents that
  declare a ``memory:`` frontmatter field get their own directory here; it is
  a separate store from the main session's auto memory.

Frontmatter is parsed by hand rather than with PyYAML. Every frontmatter line
in the observed corpus is a plain ``key: value`` with at most one level of
nesting under ``metadata:``, no folded or multi-line scalars, so a dependency
would buy nothing. Unparseable frontmatter degrades to "no frontmatter"
rather than raising -- a malformed memory file must not fail an archive build.

Two frontmatter shapes coexist. Newer files nest under ``metadata:``
(``type``, ``node_type``, ``originSessionId``, ``modified``); older ones carry
a top-level ``type:``. Both are read; ``metadata:`` wins where they overlap.

``content_hash`` deliberately excludes the ``modified:`` stamp. Claude Code
rewrites that field on every write, so hashing the raw file would open a new
SCD version each time a memory was touched without its meaning changing.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

__all__ = [
    "MemoryFile",
    "parse_memory_file",
    "iter_project_memories",
    "iter_agent_memories",
    "scanned_owners",
]

#: The index file loaded into every session (first 200 lines / 25KB).
INDEX_FILE_NAME = "MEMORY.md"

#: ``<repo>/.claude/<dir>/<agent>/`` -> the scope that directory represents.
_REPO_AGENT_DIRS = {
    "agent-memory": "project",
    "agent-memory-local": "local",
}

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
_KEY_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<key>[\w-]+):[ \t]*(?P<value>.*?)[ \t]*$")
_LINK_RE = re.compile(r"\[\[([^\]\[\n]+)\]\]")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})", re.MULTILINE)


@dataclass
class MemoryFile:
    """One memory markdown file, parsed.

    ``owner_key`` is the project directory name for project scope and the
    subagent name for agent scope -- the thing the memory belongs to, which
    is what the warehouse resolves to a ``project_key`` or agent identity.

    ``owner_root`` disambiguates owners whose key is not globally unique.
    Project directory names already are, so it is None there. Subagent names
    are NOT: the same agent can declare ``memory: user`` in one place and
    ``memory: project`` in another, and a committed ``reviewer`` agent can
    exist in any number of repositories. Without the root in the identity
    those all collapse into one memory that appears to flip-flop between
    bodies on every import.
    """

    source_path: Path
    file_name: str
    scope: str
    owner_key: str
    owner_root: str | None
    memory_name: str
    body_text: str
    content_hash: str
    raw_text: str
    has_frontmatter: bool
    is_index: bool
    description: str | None = None
    memory_type: str | None = None
    node_type: str | None = None
    origin_session_id: str | None = None
    modified: datetime | None = None
    agent_scope: str | None = None
    links: list[str] = field(default_factory=list)

    @property
    def body_chars(self) -> int:
        return len(self.body_text)

    @property
    def body_lines(self) -> int:
        return self.body_text.count("\n") + 1 if self.body_text else 0


def _unquote(value: str) -> str:
    """Strip one layer of matching surrounding quotes.

    Descriptions are written both bare and double-quoted, and quoted ones may
    contain a colon -- so the value is taken as everything after the first
    colon, then unquoted, never re-split.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_frontmatter(text: str) -> tuple[dict, dict, str]:
    """Split ``text`` into (top-level keys, metadata keys, body).

    Returns empty mappings and the untouched text when there is no
    frontmatter block, which is the normal case for ``MEMORY.md``.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, {}, text

    top: dict[str, str] = {}
    meta: dict[str, str] = {}
    in_metadata = False

    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key_match = _KEY_RE.match(line)
        if not key_match:
            # Not a plain key: value line (a list item, a folded scalar).
            # Nothing in the observed corpus looks like this; skip rather
            # than guess at YAML semantics.
            continue
        indent = key_match.group("indent")
        key = key_match.group("key")
        value = _unquote(key_match.group("value"))

        if indent:
            if in_metadata:
                meta[key] = value
            continue

        in_metadata = key == "metadata"
        if not in_metadata:
            top[key] = value

    return top, meta, text[match.end():]


def _parse_modified(value: str | None) -> datetime | None:
    """Parse the ISO 8601 ``modified`` stamp Claude Code writes.

    Written with a trailing ``Z``, which ``fromisoformat`` only accepts from
    Python 3.11; normalise it so 3.10 (the floor in pyproject) also parses.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _strip_fenced_code(text: str) -> str:
    """Blank out fenced code blocks so links inside them are not graph edges.

    A fence in a memory file is usually documentation showing the ``[[...]]``
    syntax itself; treating it as an edge would invent relationships.
    """
    out = []
    fence: str | None = None
    for line in text.splitlines():
        match = _FENCE_RE.match(line)
        marker = match.group(1)[0] if match else None
        if fence is None:
            if marker:
                fence = marker
                continue
            out.append(line)
        elif marker == fence:
            fence = None
    return "\n".join(out)


def _extract_links(body: str) -> list[str]:
    """Every ``[[name]]`` occurrence outside fenced code, in document order.

    Occurrences are kept rather than deduped: the bridge table records one row
    per link, and repeat references are real signal about which memories lean
    on which.
    """
    return [m.group(1).strip() for m in _LINK_RE.finditer(_strip_fenced_code(body))]


def parse_memory_file(
    path: str | Path,
    *,
    scope: str,
    owner_key: str,
    agent_scope: str | None = None,
    owner_root: str | None = None,
) -> MemoryFile:
    """Parse one memory markdown file.

    Args:
        path: The ``.md`` file.
        scope: ``"project"`` or ``"agent"``.
        owner_key: Project directory name, or subagent name for agent scope.
        agent_scope: ``"user"`` / ``"project"`` / ``"local"`` for agent scope.
        owner_root: Directory the owner lives under. Required for agent
            scope, where the owner key alone is not unique.
    """
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    top, meta, body = _parse_frontmatter(raw_text)
    has_frontmatter = bool(top or meta)

    # metadata: wins over the older top-level type: where both are present.
    memory_type = meta.get("type") or top.get("type") or None
    description = top.get("description") or None
    memory_name = top.get("name") or path.stem

    # Hash material is everything that carries meaning, and nothing that is a
    # write stamp. modified: is excluded on purpose -- see the module docstring.
    # \x00 separator: no memory field can contain it, so distinct field
    # splits can never collide into the same hash material.
    hash_material = "\x00".join(
        [
            memory_name,
            description or "",
            memory_type or "",
            meta.get("node_type") or "",
            meta.get("originSessionId") or "",
            body,
        ]
    )

    return MemoryFile(
        source_path=path,
        file_name=path.name,
        scope=scope,
        owner_key=owner_key,
        owner_root=owner_root,
        agent_scope=agent_scope,
        memory_name=memory_name,
        description=description,
        memory_type=memory_type,
        node_type=meta.get("node_type") or None,
        origin_session_id=meta.get("originSessionId") or None,
        modified=_parse_modified(meta.get("modified")),
        is_index=path.name == INDEX_FILE_NAME,
        has_frontmatter=has_frontmatter,
        body_text=body,
        raw_text=raw_text,
        content_hash=hashlib.md5(hash_material.encode("utf-8")).hexdigest(),
        links=_extract_links(body),
    )


def _iter_markdown(directory: Path) -> Iterator[Path]:
    """Direct ``.md`` children of ``directory``, sorted, if it exists.

    Non-markdown siblings are skipped -- a memory directory is documented to
    hold markdown, and reading anything else would put arbitrary files in the
    warehouse.
    """
    if not directory.is_dir():
        return
    yield from sorted(p for p in directory.glob("*.md") if p.is_file())


def iter_project_memories(
    projects_root: str | Path,
    only: Iterable[str] | None = None,
) -> Iterator[MemoryFile]:
    """Yield every project-scope memory file under a ``projects`` root.

    Args:
        projects_root: The ``<HOME>/.claude/projects`` directory.  # path-privacy: ignore
        only: Restrict to these project directory names. Passing the projects
            already present in ``dim_project`` keeps a filtered archive run
            (``-p mitate``) from ingesting the whole machine's memory corpus.

    A missing root yields nothing rather than raising: memory is optional and
    may be disabled entirely (``autoMemoryEnabled: false``).
    """
    projects_root = Path(projects_root)
    if not projects_root.is_dir():
        return

    wanted = set(only) if only is not None else None
    for project_dir in sorted(projects_root.iterdir()):
        if not project_dir.is_dir():
            continue
        if wanted is not None and project_dir.name not in wanted:
            continue
        for md in _iter_markdown(project_dir / "memory"):
            yield parse_memory_file(md, scope="project", owner_key=project_dir.name)


def scanned_owners(
    projects_root: str | Path | None = None,
    only: Iterable[str] | None = None,
    agent_user_root: str | Path | None = None,
    agent_repo_paths: Iterable[str | Path] = (),
) -> set[tuple[str, str, str | None]]:
    """The ``(scope, owner_key, owner_root)`` triples a scan would LOOK AT.

    Deliberately independent of whether those owners currently hold any
    memory files. A consumer that decides what to retire from what a scan
    *returned* cannot tell "this project's memory was deleted" from "this
    project was never scanned" -- so wiping a whole memory directory would
    leave its rows marked current forever. Ownership of that distinction
    belongs here, next to the directory rules it depends on.
    """
    owners: set[tuple[str, str, str | None]] = set()

    if projects_root is not None:
        root = Path(projects_root)
        wanted = set(only) if only is not None else None
        if root.is_dir():
            for project_dir in root.iterdir():
                if not project_dir.is_dir():
                    continue
                if wanted is not None and project_dir.name not in wanted:
                    continue
                owners.add(("project", project_dir.name, None))

    for root_path in (
        [agent_user_root] if agent_user_root is not None else []
    ) + [
        Path(repo) / ".claude" / sub
        for repo in agent_repo_paths
        for sub in _REPO_AGENT_DIRS
    ]:
        root = Path(root_path)
        if not root.is_dir():
            continue
        for agent_dir in root.iterdir():
            if agent_dir.is_dir():
                owners.add(("agent", agent_dir.name, str(root)))

    return owners


def iter_agent_memories(
    user_root: str | Path | None = None,
    repo_paths: Iterable[str | Path] = (),
) -> Iterator[MemoryFile]:
    """Yield every subagent-scope memory file.

    Args:
        user_root: ``<HOME>/.claude/agent-memory`` -- cross-project subagent  # path-privacy: ignore
            memory (``memory: user``).
        repo_paths: Repository roots to scan for ``.claude/agent-memory/``
            (``memory: project``, committed) and ``.claude/agent-memory-local/``
            (``memory: local``, gitignored). These live in the repo, not under
            the Claude home, so callers supply them from ``dim_session.cwd``.

    Each ``<root>/<agent>/`` subdirectory is one subagent's store; the
    subagent name is the directory name.
    """

    def _walk(root: Path, agent_scope: str) -> Iterator[MemoryFile]:
        if not root.is_dir():
            return
        for agent_dir in sorted(root.iterdir()):
            if not agent_dir.is_dir():
                continue
            for md in _iter_markdown(agent_dir):
                yield parse_memory_file(
                    md,
                    scope="agent",
                    owner_key=agent_dir.name,
                    agent_scope=agent_scope,
                    owner_root=str(root),
                )

    if user_root is not None:
        yield from _walk(Path(user_root), "user")

    for repo in repo_paths:
        claude_dir = Path(repo) / ".claude"
        for sub, agent_scope in _REPO_AGENT_DIRS.items():
            yield from _walk(claude_dir / sub, agent_scope)
