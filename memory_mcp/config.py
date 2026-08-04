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

    # OpenCode stores sessions in a SQLite database under XDG data dir.
    xdg_data = os.environ.get("XDG_DATA_HOME", "")
    if not xdg_data:
        xdg_data = str(home / ".local" / "share")
    opencode_db = Path(xdg_data) / "opencode" / "opencode.db"
    if opencode_db.is_file():
        sources.append({"type": "opencode", "path": str(opencode_db)})

    # LM Studio stores conversations as individual JSON files.
    lmstudio_convos = home / ".lmstudio" / "conversations"
    if lmstudio_convos.is_dir():
        sources.append({"type": "lmstudio", "path": str(lmstudio_convos)})

    # OpenAI Codex CLI stores sessions under ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
    codex_sessions = home / ".codex" / "sessions"
    if codex_sessions.is_dir():
        sources.append({"type": "codex", "path": str(codex_sessions)})

    # Google Gemini CLI stores sessions under ~/.gemini/tmp/<project>/chats/session-*.json
    gemini_sessions = home / ".gemini" / "tmp"
    if gemini_sessions.is_dir():
        sources.append({"type": "gemini", "path": str(gemini_sessions)})

    # LM Studio API call logs captured by lms-log-capture.
    lmstudio_api_logs = home / ".lmstudio" / "api-logs"
    if lmstudio_api_logs.is_dir():
        sources.append({"type": "lmstudio_api", "path": str(lmstudio_api_logs)})

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

def get_sync_api_url() -> str | None:
    """Return the sync server URL if configured, or None (local-only mode)."""
    return os.environ.get("MEMORY_MCP_SYNC_URL") or None


def get_sync_api_key() -> str | None:
    """Return the sync server API key if configured."""
    return os.environ.get("MEMORY_MCP_SYNC_KEY") or None


def is_sync_enabled() -> bool:
    """True when both sync URL and key are configured."""
    return bool(get_sync_api_url() and get_sync_api_key())


def get_sync_config() -> dict:
    """Return the full sync configuration as a dict.

    Returns an empty dict when sync is not configured.
    """
    url = get_sync_api_url()
    key = get_sync_api_key()
    if not (url and key):
        return {}
    return {"api_url": url.rstrip("/"), "api_key": key}
