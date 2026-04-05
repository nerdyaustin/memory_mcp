"""Configuration for memory MCP server.

Session sources are auto-detected from known locations. Override or extend
via the MEMORY_MCP_SOURCES environment variable (semicolon-separated
entries of ``type:path``, e.g. ``omp:/other/omp;claude_code:/alt/claude``).
"""

from __future__ import annotations

import os
from pathlib import Path


def get_session_sources() -> list[dict[str, str]]:
    """Discover session data directories on this machine."""
    home = Path.home()
    sources: list[dict[str, str]] = []

    # Claude Code stores sessions under ~/.claude/projects/
    claude_projects = home / ".claude" / "projects"
    if claude_projects.is_dir():
        sources.append({"type": "claude_code", "path": str(claude_projects)})

    # Claude Code keeps a running log of every user prompt, even after
    # full session files are pruned (~30 days).  Invaluable for search.
    claude_history = home / ".claude" / "history.jsonl"
    if claude_history.is_file():
        sources.append({"type": "claude_history", "path": str(claude_history)})

    # OMP stores sessions under ~/.omp/agent/sessions/
    omp_sessions = home / ".omp" / "agent" / "sessions"
    if omp_sessions.is_dir():
        sources.append({"type": "omp", "path": str(omp_sessions)})

    # Additional sources from env: "type:path;type:path"
    extra = os.environ.get("MEMORY_MCP_SOURCES", "")
    for entry in extra.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            sources.append({"type": parts[0], "path": parts[1]})

    return sources


def get_db_path() -> Path:
    """Return the path to the SQLite database file."""
    custom = os.environ.get("MEMORY_MCP_DB")
    if custom:
        return Path(custom)
    db_dir = Path.home() / ".memory_mcp"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "memory.db"
