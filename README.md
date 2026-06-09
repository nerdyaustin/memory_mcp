# Memory MCP

Persistent memory and full-text session search for AI coding assistants, exposed as an [MCP](https://modelcontextprotocol.io/) server.

## The problem

AI coding assistants forget everything between sessions. Architecture decisions, user preferences, project context, what you debugged last Tuesday -- gone. You re-explain the same things constantly.

Memory MCP fixes this with two capabilities:

1. **Explicit memory** -- save notes, decisions, patterns, and preferences that persist across sessions. Your assistant remembers what you told it.
2. **Session search** -- full-text search across your entire conversation history. Find that thing you discussed three weeks ago without scrolling through logs.

No database servers. No background processes. No cloud. One SQLite file on your machine.

## Supported session sources

| Source | Location | Format |
|--------|----------|--------|
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | `~/.claude/projects/` | JSONL (streamed content blocks) |
| Claude Code history | `~/.claude/history.jsonl` | JSONL (survives session file pruning) |
| [OpenCode](https://opencode.ai) | `~/.local/share/opencode/opencode.db` | SQLite (sessions, messages, parts tables) |
| [Oh My Pi](https://github.com/can1357/oh-my-pi) | `~/.omp/agent/sessions/` | JSONL (event-per-line) |
| [Codex CLI](https://github.com/openai/codex) | `~/.codex/sessions/` | JSONL (rollout events) |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | `~/.gemini/tmp/` | JSON (chat sessions) |
| [LM Studio](https://lmstudio.ai) | `~/.lmstudio/conversations/` | JSON (conversations) |
| LM Studio API logs | `~/.lmstudio/api-logs/` | JSONL (via `lms-log-capture`) |

Adding a new source requires one parser file and a registry entry. See [Adding a new source](#adding-a-new-session-source).

## Installation

Requires Python 3.11+ with SQLite FTS5 support (included in standard Python builds).

```bash
pip install -e .
```

Or run directly with [uv](https://docs.astral.sh/uv/) (no install needed):

```bash
uv run --directory /path/to/memory_mcp python -m memory_mcp
```

## MCP configuration

Add to your MCP client config (e.g., `~/.claude/mcp.json` or project-level `.mcp.json`):

**With pip install:**

```json
{
  "mcpServers": {
    "memory": {
      "command": "memory-mcp"
    }
  }
}
```

**With uv (no install):**

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/memory_mcp", "python", "-m", "memory_mcp"]
    }
  }
}
```

## Tools

### Memory (explicit knowledge store)

| Tool | Description |
|------|-------------|
| `save_memory` | Persist a note with optional tags and context. Survives across all future sessions. |
| `search_memory` | Full-text search across saved memories. Keyword-based, ranked by relevance. |
| `list_memories` | Browse recent memories, optionally filtered by tag. |
| `delete_memory` | Remove a memory by ID. |

### Sessions (historical conversation search)

| Tool | Description |
|------|-------------|
| `list_sessions` | Browse past sessions. Filter by source (`claude_code`, `omp`) or project path. |
| `get_session` | Retrieve the full conversation from a specific session. |
| `get_tool_calls` | Get tool calls and results for a session, optionally filtered by tool name. |
| `search_sessions` | Full-text search across all session messages, thinking blocks, and tool usage. |
| `refresh_sessions` | Re-scan session directories and index new or changed files. |

## How it works

On startup, Memory MCP yields its tool list to the MCP client immediately (<500 ms cold) and runs the initial session scan in a background task. The embedding model loads lazily on the first semantic search call — keyword search and saved memories work without it. Subsequent startups skip files whose mtime hasn't changed.

- **Database location:** `~/.memory_mcp/memory.db` (override with `MEMORY_MCP_DB` env var)
- **Session sources:** auto-detected from standard locations (extend with `MEMORY_MCP_SOURCES` env var, format: `type:path;type:path`)
- **Indexing:** incremental by file mtime, parallelized across 8 threads
- **Search:** FTS5 with BM25 ranking, prefix matching, phrase support; optional vector search via sqlite-vec + fastembed (BAAI/bge-small-en-v1.5) when `semantic=true` is passed
- **Startup:** non-blocking — heavy work (scan, model load, vector backfill) runs after the server is already responding to tool calls

## Adding a new session source

1. Create `memory_mcp/parsers/your_source.py` implementing the `SessionParser` protocol:
   - `source_type: str` attribute
   - `parse_file(path: str) -> ParsedSession | None` method
2. Register it in `memory_mcp/parsers/__init__.py`
3. Add directory detection in `memory_mcp/config.py`

See `parsers/claude_code.py` or `parsers/omp.py` for examples.

## Testing

```bash
python tests/test_e2e.py        # end-to-end: spawns server, exercises all 9 tools
python tests/test_startup.py    # startup contract: cold Popen -> tools/list under 1.5s
```

`test_e2e.py` starts the MCP server as a subprocess, exercises all 9 tools over the stdio protocol, and asserts tool responses. `test_startup.py` enforces the v0.3.0 startup contract — if an eager import or pre-yield blocking call regresses startup speed, it fails immediately. Both use throwaway databases so your real data is untouched.

## Architecture

```
memory_mcp/
  server.py        # FastMCP entry point, lifespan yields fast then runs scan in background
  readiness.py     # Lazy embedder + scan/backfill coordination
  config.py        # Auto-detects session dirs, DB path
  db.py            # SQLite + FTS5 + sqlite-vec schema, all queries, sync triggers
  embeddings.py    # Lazy fastembed wrapper (BAAI/bge-small-en-v1.5)
  scanner.py       # Walks session dirs, dispatches to parsers, parallel indexing
  parsers/
    base.py        # ParsedSession / ParsedMessage dataclasses, SessionParser protocol
    claude_code.py # Claude Code JSONL parser (merges streamed assistant blocks)
    claude_history.py # Claude Code history.jsonl parser (one file, many sessions)
    omp.py         # OMP JSONL parser
    opencode.py    # OpenCode SQLite parser (reads DB directly, read-only)
  tools/
    memory.py      # save_memory, search_memory, list_memories, delete_memory
    sessions.py    # list_sessions, get_session, search_sessions, refresh_sessions
```

## License

MIT
