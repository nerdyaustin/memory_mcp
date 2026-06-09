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
- The embedding model is **never** loaded at startup. It loads lazily on the first `search_memory(semantic=True)` or `search_sessions(semantic=True)` call, via `readiness.ensure_semantic_ready()`.
- `save_memory` uses `readiness.get_embedder_if_ready()` — opportunistic embedding only. It never triggers a cold load. Periodic backfill embeds any rows it leaves behind once semantic search is actually used.
- The initial session scan runs in a background task, not in lifespan. Tools work as soon as MCP is ready; semantic search awaits scan completion via `state.scan_done` before backfilling.
- All scan/backfill work runs in worker threads via `asyncio.to_thread`. SQLite connections are thread-bound, so worker threads always open their own connection via `init_db()`. Never pass the lifespan connection into `to_thread`.
- Scan + backfill are serialised via `state.maintenance_lock` to prevent concurrent writers and to keep semantic results consistent.
- `embeddings.py` does **not** import `fastembed` at module top. The import lives inside `Embedder.__init__`. Probe availability with `importlib.util.find_spec("fastembed")` instead.

**Test:** `tests/test_startup.py` measures wall time from `subprocess.Popen` to the `tools/list` response. Threshold is 1.5s on this machine; observed values are ~200–300ms cold, ~150–250ms warm. If this test starts failing, an eager import or pre-yield blocking call has been added — find it before merging.

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
```

The e2e test uses a throwaway DB (`MEMORY_MCP_DB` pointed at a temp dir), runs
the full MCP handshake via `mcp.client.stdio`, and asserts every tool's output.
It scans real session data from your machine, so search results depend on what
sessions exist locally.
