# Memory MCP

Persistent memory and session search for AI coding assistants, exposed as an MCP server.

## What this is

A Python MCP server that:
1. Indexes session history from Claude Code (`~/.claude/projects/`) and OMP (`~/.omp/agent/sessions/`)
2. Provides full-text search across all historical sessions via SQLite FTS5
3. Offers explicit `save_memory`/`search_memory` tools for cross-session knowledge persistence

No Flask, no Postgres, no file watchers. SQLite handles everything. The MCP SDK handles transport.

## Architecture

```
memory_mcp/
  server.py              # FastMCP entry point, lifespan yields fast then runs scan in background
  readiness.py           # Lazy embedder + scan/backfill coordination (v0.3.0)
  config.py              # Auto-detects session dirs, DB path (~/.memory_mcp/memory.db)
  db.py                  # SQLite + FTS5 + sqlite-vec schema, all query functions
  embeddings.py          # Lazy fastembed wrapper (BAAI/bge-small-en-v1.5)
  scanner.py             # Walks session dirs, dispatches to parsers, indexes into DB
  parsers/
    base.py              # ParsedSession / ParsedMessage dataclasses, SessionParser protocol
    claude_code.py       # Claude Code JSONL parser (merges streamed assistant blocks)
    omp.py               # OMP JSONL parser
  tools/
    memory.py            # save_memory, search_memory, list_memories, delete_memory
    sessions.py          # list_sessions, get_session, search_sessions, refresh_sessions
```

## Startup contract (v0.3.0)

This is a hard contract. Breaking it causes MCP clients (Claude Code, Codex, VS Code) to silently miss the server's tools on startup, the bug that motivated v0.3.0.

**Rule:** `lifespan` MUST yield in <500ms on every cold boot. Anything that blocks longer than that goes in a background task started after `yield`.

What this means in practice:
- `server.py:lifespan` does only `init_db()` + `ReadinessState.new()` + `init_readiness(state)` + `asyncio.create_task(_background_startup(...))` before yielding. Do not add anything else pre-yield.
- The embedding model is **never** loaded pre-yield. The background startup task warm-starts it (`readiness.warm_start_semantic`) right after the initial scan, so it's usually ready before the user's first prompt. If a semantic call races the warm-start, `ensure_semantic_ready()` loads the model on demand — but it must **never** wait for scan or backfill completion; semantic queries search whatever vectors exist and backfill catches up in the background. Blocking on backfill is what caused the 30s MCP client timeouts fixed in v0.5.0.
- `save_memory` uses `readiness.get_embedder_if_ready()` — opportunistic embedding only. It never triggers a cold load. Backfill embeds any rows saved before the model was up.
- The initial session scan runs in a background task, not in lifespan. Tools work as soon as MCP is ready.
- All scan/backfill work runs in worker threads via `asyncio.to_thread`. SQLite connections are thread-bound, so worker threads always open their own connection via `init_db()`. Never pass the lifespan connection into `to_thread`.
- Scan + backfill are serialised via `state.maintenance_lock` to prevent concurrent writers and to keep semantic results consistent.
- `embeddings.py` does **not** import `fastembed` at module top. The import lives inside `Embedder.__init__`. Probe availability with `importlib.util.find_spec("fastembed")` instead.

**Test:** `tests/test_startup.py` measures wall time from `subprocess.Popen` to the `tools/list` response. Threshold is 1.5s on this machine; observed values are ~200–300ms cold, ~150–250ms warm. If this test starts failing, an eager import or pre-yield blocking call has been added — find it before merging.

## Shutdown contract

**Rule:** the server MUST NOT outlive the client that spawned it.

`main()` calls `start_parent_watchdog()` (`memory_mcp/parent_watchdog.py`) before
`mcp.run()`. The watchdog thread waits on the parent process handle (Windows) or
polls `getppid()`/`kill(pid, 0)` (POSIX) and calls `os._exit(0)` when the parent
is gone.

Do not delete this on the theory that stdin EOF already covers it. EOF is not
reliable: if any other process inherited the write end of our stdin pipe — a
shell shim, a broker, a sibling spawned by the same client — the handle stays
open after the client dies and the read never returns.

The failure this prevents, observed 2026-07-24: an orphaned `pythonw.exe -m
memory_mcp` survived its client by a day holding an open connection to
`~/.memory_mcp/memory.db`. Every later client then died in `init_db()` at
`_refresh_fts_triggers` with `sqlite3.OperationalError: database is locked`, so
each new session came up with no memory tools at all and silently stopped
indexing new sessions. Under `pythonw.exe` the orphan has no console or window,
so nothing on screen explained it.

**Test:** `tests/test_orphan_exit.py` builds the real topology — a shim spawns
the server on a hand-made pipe plus a keeper process that inherits the
inheritable write end — then kills the shim. The server must exit within 20s
while the keeper still holds stdin open. Verified discriminating: with the
watchdog the server exits in ~0.1s, without it the test fails at the deadline.

## Running

```bash
# Install
pip install -e .

# Run directly (stdio transport for MCP)
python -m memory_mcp

# Or via entry point
memory-mcp
```

## MCP Registration

Add to your Claude Code `~/.claude/mcp.json` or project-level `.mcp.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "memory_mcp"],
      "env": {}
    }
  }
}
```

Or with uv (no install needed):

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "--directory", "C:/Users/Austin/source/repos/memory_mcp", "python", "-m", "memory_mcp"]
    }
  }
}
```

## Tools (8 total)

**Memory (explicit knowledge store):**
- `save_memory(content, tags?, context?)` - persist a note across sessions
- `search_memory(query, tags?, limit?)` - FTS search through saved memories
- `list_memories(tag?, limit?)` - browse recent memories
- `delete_memory(memory_id)` - remove a memory

**Sessions (historical session search):**
- `list_sessions(source?, project?, limit?)` - browse past sessions
- `get_session(session_id)` - retrieve a specific conversation
- `search_sessions(query, limit?)` - FTS search across all session messages
- `refresh_sessions()` - re-scan for new/changed session files

## Data storage

- Database: `~/.memory_mcp/memory.db` (SQLite, override with `MEMORY_MCP_DB` env var)
- Session sources auto-detected; extend with `MEMORY_MCP_SOURCES` env var (format: `type:path;type:path`)
- FTS5 indexes on both memories and messages tables with sync triggers
- Incremental indexing: files are skipped if mtime hasn't changed since last scan
- Re-indexing a changed session file diffs per-message on the stable message `id` — unchanged messages keep their rowid so their FTS entries and vectors survive. Do NOT go back to delete-all/re-insert in `upsert_session`: that orphans every vector on each re-scan of an active session (the bloat that made semantic search 30x slower pre-v0.5.0). `prune_orphan_vectors` (called from every backfill) cleans any orphans other paths leave behind.
- Semantic results are filtered at `db.DEFAULT_MAX_DISTANCE` (L2 over unit vectors); KNN otherwise returns the nearest rows no matter how irrelevant.

## Session format notes

**OMP**: One JSONL line per event. Session header provides title/cwd. Messages have `id`/`parentId` chains. Cost data available per-message in `usage.cost.total`.

**Claude Code**: One JSONL line per content block. Assistant responses are streamed as multiple lines sharing the same `message.id` - the parser groups and merges these. User messages have string content (not arrays). Summary-only files (type "summary") are skipped.

## Adding a new session source

1. Create `memory_mcp/parsers/new_source.py` implementing `SessionParser` protocol
2. Class needs `source_type: str` attribute and `parse_file(path) -> ParsedSession | None` method
3. Register in `memory_mcp/parsers/__init__.py` PARSERS dict
4. Add directory detection in `config.py` `get_session_sources()`

## Dependencies

- Python >= 3.11
- `mcp` >= 1.20.0 (MCP SDK with FastMCP)
- SQLite with FTS5 (included in Python's bundled sqlite3)

No other dependencies. No database servers. No background processes.

## Testing

```bash
# End-to-end test: starts MCP subprocess, exercises all 9 tools over stdio protocol
python tests/test_e2e.py

# Lifecycle: server answers tools/list fast, and dies with its client
python tests/test_startup.py
python tests/test_orphan_exit.py
```

The e2e test uses a throwaway DB (`MEMORY_MCP_DB` pointed at a temp dir), runs
the full MCP handshake via `mcp.client.stdio`, and asserts every tool's output.
It scans real session data from your machine, so search results depend on what
sessions exist locally.
