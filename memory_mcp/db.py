"""SQLite database with FTS5 for memory and session storage.

Every public function takes an explicit ``sqlite3.Connection`` so callers
control the lifetime (the MCP server passes it via lifespan context).
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .config import get_db_path

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
-- Explicit memories (the primary feature)
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    tags        TEXT,           -- JSON array of strings
    context     TEXT,           -- what prompted this memory
    source_session_id TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags, context,
    content='memories',
    content_rowid='id'
);

-- Keep FTS in sync via triggers (required for content-external tables).
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags, context)
    VALUES (new.id, new.content, new.tags, new.context);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, context)
    VALUES ('delete', old.id, old.content, old.tags, old.context);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, context)
    VALUES ('delete', old.id, old.content, old.tags, old.context);
    INSERT INTO memories_fts(rowid, content, tags, context)
    VALUES (new.id, new.content, new.tags, new.context);
END;

-- Parsed session headers
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,   -- 'claude_code' | 'omp'
    title           TEXT,
    cwd             TEXT,
    model           TEXT,
    started_at      TEXT,
    message_count   INTEGER DEFAULT 0,
    total_cost_usd  REAL    DEFAULT 0.0,
    file_path       TEXT    NOT NULL,
    file_mtime      REAL    NOT NULL
);

-- Parsed messages (one row per conversational turn)
CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    parent_id   TEXT,
    role        TEXT NOT NULL,      -- 'user' | 'assistant' | 'tool_use' | 'tool_result'
    content     TEXT,               -- human-readable text
    thinking    TEXT,               -- thinking blocks
    tool_name   TEXT,
    tool_input  TEXT,               -- JSON string
    tool_output TEXT,
    timestamp   TEXT,
    model       TEXT,
    cost_usd    REAL
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, thinking, tool_name, tool_input, tool_output,
    content='messages',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, thinking, tool_name, tool_input, tool_output)
    VALUES (new.rowid, new.content, new.thinking, new.tool_name, new.tool_input, new.tool_output);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, thinking, tool_name, tool_input, tool_output)
    VALUES ('delete', old.rowid, old.content, old.thinking, old.tool_name, old.tool_input, old.tool_output);
END;

CREATE INDEX IF NOT EXISTS idx_messages_session  ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_source   ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_started  ON sessions(started_at DESC);
"""

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_db(path: Path | None = None) -> sqlite3.Connection:
    """Create tables / indexes and return a connection."""
    db_path = path or get_db_path()
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(_SCHEMA)
    return db


# ---------------------------------------------------------------------------
# FTS helper
# ---------------------------------------------------------------------------

_FTS_SPECIAL = re.compile(r"[^\w\s*\"-]", re.UNICODE)


def _safe_fts_query(raw: str) -> str:
    """Sanitise a user query for FTS5 MATCH.
    
    Joins tokens with OR so multi-word queries (typical of LLM callers)
    match any term rather than requiring all.  BM25 naturally ranks rows
    with more matching terms higher.  Adds prefix wildcards to bare tokens
    so partial words still hit.  Preserves explicit double-quoted phrases.
    """
    tokens = re.findall(r'"[^"]*"|\S+', raw)
    parts: list[str] = []
    for tok in tokens:
        if tok.startswith('"') and tok.endswith('"'):
            parts.append(tok)  # user-supplied phrase, keep exact
        else:
            clean = _FTS_SPECIAL.sub("", tok)
            if clean:
                parts.append(f'"{clean}"*')  # prefix wildcard
    return " OR ".join(parts) if parts else '""'


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------


def save_memory(
    db: sqlite3.Connection,
    content: str,
    tags: list[str] | None = None,
    context: str | None = None,
    source_session_id: str | None = None,
) -> int:
    """Store a memory note.  Returns the new row id."""
    tags_json = json.dumps(tags) if tags else None
    cur = db.execute(
        "INSERT INTO memories (content, tags, context, source_session_id) VALUES (?, ?, ?, ?)",
        (content, tags_json, context, source_session_id),
    )
    db.commit()
    return cur.lastrowid  # type: ignore[return-value]


def search_memories(
    db: sqlite3.Connection,
    query: str,
    tags: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Full-text search across memories."""
    safe_q = _safe_fts_query(query)
    base = (
        "SELECT m.id, m.content, m.tags, m.context, m.created_at, fts.rank "
        "FROM memories_fts fts "
        "JOIN memories m ON m.id = fts.rowid "
        "WHERE memories_fts MATCH ?"
    )
    params: list = [safe_q]

    if tags:
        clauses = " AND ".join("m.tags LIKE ?" for _ in tags)
        base += f" AND {clauses}"
        params.extend(f'%"{t}"%' for t in tags)

    base += " ORDER BY fts.rank LIMIT ?"
    params.append(limit)

    try:
        rows = db.execute(base, params).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in rows]


def list_memories(
    db: sqlite3.Connection,
    tag: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List recent memories, optionally filtered by tag."""
    if tag:
        rows = db.execute(
            "SELECT id, content, tags, context, created_at "
            "FROM memories WHERE tags LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (f'%"{tag}"%', limit),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, content, tags, context, created_at "
            "FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_memory(db: sqlite3.Connection, memory_id: int) -> bool:
    """Delete a memory by id.  Returns True if a row was removed."""
    cur = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


def upsert_session(db: sqlite3.Connection, session: dict) -> None:
    """Insert (or replace) a parsed session and all its messages.

    Deletes existing messages first so the FTS delete-triggers fire and the
    index stays consistent.
    """
    sid = session["id"]
    # Remove stale data (triggers clean FTS).
    db.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
    db.execute("DELETE FROM sessions WHERE id = ?", (sid,))

    db.execute(
        "INSERT INTO sessions "
        "(id, source, title, cwd, model, started_at, message_count, total_cost_usd, file_path, file_mtime) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            sid,
            session["source"],
            session.get("title"),
            session.get("cwd"),
            session.get("model"),
            session.get("started_at"),
            session.get("message_count") or len(session.get("messages", [])),
            session.get("total_cost_usd", 0.0),
            session["file_path"],
            session["file_mtime"],
        ),
    )

    for msg in session.get("messages", []):
        db.execute(
            "INSERT OR IGNORE INTO messages "
            "(id, session_id, parent_id, role, content, thinking, "
            "tool_name, tool_input, tool_output, timestamp, model, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                msg["id"],
                sid,
                msg.get("parent_id"),
                msg["role"],
                msg.get("content"),
                msg.get("thinking"),
                msg.get("tool_name"),
                msg.get("tool_input"),
                msg.get("tool_output"),
                msg.get("timestamp"),
                msg.get("model"),
                msg.get("cost_usd"),
            ),
        )
    db.commit()


def get_session_mtime(db: sqlite3.Connection, file_path: str) -> float | None:
    """Return the stored mtime for a session file, or None if unseen."""
    row = db.execute(
        "SELECT file_mtime FROM sessions WHERE file_path = ?", (file_path,)
    ).fetchone()
    return float(row["file_mtime"]) if row else None


def count_sessions(
    db: sqlite3.Connection,
    source: str | None = None,
    project: str | None = None,
) -> int:
    """Return total session count matching the given filters."""
    clauses, params = _session_filters(source, project)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    row = db.execute(f"SELECT count(*) FROM sessions {where}", params).fetchone()
    return row[0]


def _session_filters(
    source: str | None, project: str | None,
) -> tuple[list[str], list]:
    clauses: list[str] = []
    params: list = []
    if source:
        clauses.append("source = ?")
        params.append(source)
    if project:
        clauses.append("cwd LIKE ?")
        params.append(f"%{project}%")
    return clauses, params


def list_sessions(
    db: sqlite3.Connection,
    source: str | None = None,
    project: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """List sessions, newest first.  Optional filters by source or project path."""
    clauses, params = _session_filters(source, project)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    rows = db.execute(
        f"SELECT id, source, title, cwd, model, started_at, "
        f"message_count, total_cost_usd "
        f"FROM sessions {where} "
        f"ORDER BY started_at DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    return [dict(r) for r in rows]


def get_session_messages(
    db: sqlite3.Connection,
    session_id: str,
    limit: int = 50,
    offset: int = 0,
) -> tuple[dict | None, list[dict], int]:
    """Return (session_header, messages_page, total_messages) for a session."""
    header = db.execute(
        "SELECT id, source, title, cwd, model, started_at, message_count, total_cost_usd "
        "FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if not header:
        return None, [], 0
    total = db.execute(
        "SELECT count(*) FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    rows = db.execute(
        "SELECT id, role, content, thinking, tool_name, tool_input, "
        "tool_output, timestamp, model, cost_usd "
        "FROM messages WHERE session_id = ? ORDER BY rowid LIMIT ? OFFSET ?",
        (session_id, limit, offset),
    ).fetchall()
    return dict(header), [dict(r) for r in rows], total


def search_messages(
    db: sqlite3.Connection,
    query: str,
    limit: int = 10,
    offset: int = 0,
    role: str | None = None,
) -> list[dict]:
    """Full-text search across session messages with session context.

    Uses bm25() column weights to prioritize conversational content
    (content, thinking) over tool noise (tool_input, tool_output).
    FTS5 columns: content, thinking, tool_name, tool_input, tool_output
    bm25 weights: higher magnitude = more important.
    """
    safe_q = _safe_fts_query(query)
    # bm25() args map to FTS columns in declaration order:
    #   content=10, thinking=5, tool_name=8, tool_input=1, tool_output=1
    sql = (
        "SELECT m.id, m.session_id, m.role, m.content, m.thinking, "
        "m.tool_name, m.tool_output, m.timestamp, "
        "s.title AS session_title, s.cwd AS session_cwd, "
        "s.started_at AS session_date, s.source AS session_source "
        "FROM messages_fts fts "
        "JOIN messages m ON m.rowid = fts.rowid "
        "JOIN sessions s ON s.id = m.session_id "
        "WHERE messages_fts MATCH ?"
    )
    params: list = [safe_q]

    if role:
        sql += " AND m.role = ?"
        params.append(role)

    sql += " ORDER BY bm25(messages_fts, 10.0, 5.0, 8.0, 1.0, 1.0) LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        rows = db.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in rows]