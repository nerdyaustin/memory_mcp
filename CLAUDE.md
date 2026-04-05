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
  server.py              # FastMCP entry point, lifespan wires DB + initial scan
  config.py              # Auto-detects session dirs, DB path (~/.memory_mcp/memory.db)
  db.py                  # SQLite + FTS5 schema, all query functions
  scanner.py             # Walks session dirs, dispatches to parsers, indexes into DB
  parsers/
    base.py              # ParsedSession / ParsedMessage dataclasses, SessionParser protocol
    claude_code.py       # Claude Code JSONL parser (merges streamed assistant blocks)
    omp.py               # OMP JSONL parser
  tools/
    memory.py            # save_memory, search_memory, list_memories, delete_memory
    sessions.py          # list_sessions, get_session, search_sessions, refresh_sessions
```

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
# End-to-end test: starts MCP subprocess, exercises all 8 tools over stdio protocol
python tests/test_e2e.py
```

The e2e test uses a throwaway DB (`MEMORY_MCP_DB` pointed at a temp dir), runs
the full MCP handshake via `mcp.client.stdio`, and asserts every tool's output.
It scans real session data from your machine, so search results depend on what
sessions exist locally.
