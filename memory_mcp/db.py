"""SQLite database with FTS5 and sqlite-vec for memory and session storage.

Every public function takes an explicit ``sqlite3.Connection`` so callers
control the lifetime (the MCP server passes it via lifespan context).

Vector search (semantic) is optional: if sqlite-vec or fastembed is not
installed, the server falls back to FTS5-only mode with no loss of
existing functionality.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

from .config import get_db_path

log = logging.getLogger(__name__)

# Set True once sqlite-vec loads successfully on any connection.
# Checked by vec-dependent functions so they can no-op gracefully.
VEC_AVAILABLE = False

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

# vec0 tables are created separately because they require the sqlite-vec
# extension, which may not be available.
_VEC_SCHEMA = """\
CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    embedding float[384]
);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(
    embedding float[384]
);
"""

# ---------------------------------------------------------------------------
# sqlite-vec extension loading
# ---------------------------------------------------------------------------


def _load_vec(db: sqlite3.Connection) -> bool:
    """Load sqlite-vec into *db*.  Returns True on success.

    Sets the module-level ``VEC_AVAILABLE`` flag so downstream functions
    know whether vector operations are safe on *any* connection on this
    system (the extension is either installable or not).
    """
    global VEC_AVAILABLE
    try:
        import sqlite_vec  # type: ignore[import-untyped]

        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        VEC_AVAILABLE = True
        return True
    except (ImportError, sqlite3.OperationalError) as exc:
        log.warning("sqlite-vec unavailable: %s — semantic search disabled", exc)
        return False


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
    # Wait up to 5s for a contended write lock before raising "database is
    # locked". Needed because v0.3.0 runs the initial scan + backfill as a
    # background task alongside live tool calls (reads under WAL are lock-free,
    # but writers still serialize on the single writer lock).
    db.execute("PRAGMA busy_timeout=5000")
    # Cap the WAL file. Without this, the WAL only ever grows: a passive
    # autocheckpoint can copy frames into the main db but cannot shrink the
    # file while any other connection holds a read snapshot, and with several
    # server instances open that quiet moment never arrives. journal_size_limit
    # forces SQLite to truncate the WAL back to this ceiling after each
    # checkpoint; checkpoint_wal() (called after every scan) does the rest.
    db.execute("PRAGMA journal_size_limit=67108864")  # 64 MB
    db.executescript(_SCHEMA)

    # Attempt to enable vector search (non-fatal if unavailable).
    if _load_vec(db):
        db.executescript(_VEC_SCHEMA)
        log.info("sqlite-vec loaded — semantic search enabled")

    return db


def checkpoint_wal(db: sqlite3.Connection) -> None:
    """Flush committed WAL frames into the main db and truncate the WAL to
    zero. Safe and non-destructive: only already-committed data is moved, then
    the now-redundant WAL file is shrunk. Call after each scan so the WAL can
    never balloon (see PRAGMA journal_size_limit in init_db). If another
    connection is mid-read the truncate is skipped this round and retried on the
    next scan — never an error."""
    try:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError:
        log.warning("WAL checkpoint skipped (db busy); will retry next scan")


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
    embedding: bytes | None = None,
) -> int:
    """Store a memory note.  Returns the new row id.

    If *embedding* (serialized float32 blob) is provided and sqlite-vec is
    loaded, the vector is stored in ``vec_memories`` for semantic search.
    """
    tags_json = json.dumps(tags) if tags else None
    cur = db.execute(
        "INSERT INTO memories (content, tags, context, source_session_id) VALUES (?, ?, ?, ?)",
        (content, tags_json, context, source_session_id),
    )
    memory_id = cur.lastrowid

    if embedding and VEC_AVAILABLE:
        try:
            db.execute(
                "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                (memory_id, embedding),
            )
        except sqlite3.OperationalError:
            log.warning("Failed to store embedding for memory %d", memory_id)

    db.commit()
    return memory_id  # type: ignore[return-value]


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


def semantic_search_memories(
    db: sqlite3.Connection,
    query_embedding: bytes,
    tags: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Vector similarity search across memories.

    Returns memories ordered by cosine distance (lower = more similar).
    Requires sqlite-vec; returns empty list if unavailable.
    """
    if not VEC_AVAILABLE:
        return []

    try:
        # vec0 KNN requires k=? in the WHERE clause (LIMIT doesn't propagate).
        k = limit * 2 if tags else limit
        rows = db.execute(
            "SELECT v.rowid, v.distance, "
            "m.id, m.content, m.tags, m.context, m.created_at "
            "FROM vec_memories v "
            "JOIN memories m ON m.id = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ?",
            (query_embedding, k),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("Semantic memory search failed: %s", exc)
        return []

    results = [dict(r) for r in rows]

    # Apply tag filter in Python (vec0 MATCH doesn't support compound WHERE).
    if tags:
        filtered = []
        for r in results:
            tags_raw = r.get("tags") or ""
            if all(f'"{t}"' in tags_raw for t in tags):
                filtered.append(r)
        results = filtered[:limit]

    return results


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
    # Clean up vector entry first (vec0 has no triggers).
    if VEC_AVAILABLE:
        try:
            db.execute("DELETE FROM vec_memories WHERE rowid = ?", (memory_id,))
        except sqlite3.OperationalError:
            pass
    cur = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


def upsert_session(db: sqlite3.Connection, session: dict) -> None:
    """Insert (or replace) a parsed session and all its messages.

    Deletes existing messages first so the FTS delete-triggers fire and the
    index stays consistent.  Also cleans up any vec_messages entries for the
    deleted rows (vec0 has no automatic triggers).
    """
    sid = session["id"]

    # Clean up vector entries for messages being replaced.
    if VEC_AVAILABLE:
        try:
            rowids = db.execute(
                "SELECT rowid FROM messages WHERE session_id = ?", (sid,)
            ).fetchall()
            if rowids:
                placeholders = ",".join("?" * len(rowids))
                db.execute(
                    f"DELETE FROM vec_messages WHERE rowid IN ({placeholders})",
                    [r[0] for r in rowids],
                )
        except sqlite3.OperationalError:
            pass

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


def get_tool_calls(
    db: sqlite3.Connection,
    session_id: str,
    tool_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Return (tool_calls_page, total_count) for a session.

    When tool_name is None, returns all tool_use and tool_result messages.
    Otherwise filters to the specified tool name (case-insensitive prefix match).
    """
    where_clauses = ["session_id = ?", "role IN ('tool_use', 'tool_result')"]
    params: list = [session_id]

    if tool_name:
        where_clauses.append("tool_name LIKE ?")
        params.append(f"{tool_name}%")

    where = " AND ".join(where_clauses)
    total = db.execute(
        f"SELECT count(*) FROM messages WHERE {where}",
        params,
    ).fetchone()[0]

    params.extend([limit, offset])
    rows = db.execute(
        f"""SELECT id, session_id, parent_id, role, content, thinking,
               tool_name, tool_input, tool_output, timestamp, model, cost_usd
               FROM messages WHERE {where}
               ORDER BY rowid LIMIT ? OFFSET ?""",
        params,
    ).fetchall()
    return [dict(r) for r in rows], total


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


def semantic_search_messages(
    db: sqlite3.Connection,
    query_embedding: bytes,
    limit: int = 10,
    offset: int = 0,
) -> list[dict]:
    """Vector similarity search across session messages.

    Returns messages ordered by cosine distance (lower = more similar),
    joined with session metadata for context.  Requires sqlite-vec.
    """
    if not VEC_AVAILABLE:
        return []

    # vec0 KNN requires k=? in WHERE clause; doesn't support OFFSET.
    fetch_count = limit + offset
    try:
        rows = db.execute(
            "SELECT v.rowid, v.distance, "
            "m.id, m.session_id, m.role, m.content, m.thinking, "
            "m.tool_name, m.tool_output, m.timestamp, "
            "s.title AS session_title, s.cwd AS session_cwd, "
            "s.started_at AS session_date, s.source AS session_source "
            "FROM vec_messages v "
            "JOIN messages m ON m.rowid = v.rowid "
            "JOIN sessions s ON s.id = m.session_id "
            "WHERE v.embedding MATCH ? AND k = ?",
            (query_embedding, fetch_count),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("Semantic message search failed: %s", exc)
        return []

    return [dict(r) for r in rows[offset:]]


# ---------------------------------------------------------------------------
# Embedding backfill
# ---------------------------------------------------------------------------


def backfill_embeddings(
    db: sqlite3.Connection,
    embedder,
    batch_size: int = 256,
) -> dict:
    """Embed memories and messages that don't yet have vectors.

    Called after scan on server startup and after periodic re-scans.
    The *embedder* must have an ``embed_batch(texts) -> list[bytes]``
    method (see ``embeddings.Embedder``).

    Returns counts of newly embedded rows.
    """
    if not VEC_AVAILABLE:
        return {"memories_embedded": 0, "messages_embedded": 0}

    mem_count = _backfill_table(
        db, embedder, batch_size,
        source_table="memories",
        id_column="id",
        text_column="content",
        vec_table="vec_memories",
    )

    msg_count = _backfill_table(
        db, embedder, batch_size,
        source_table="messages",
        id_column="rowid",
        text_column="content",
        vec_table="vec_messages",
    )

    return {"memories_embedded": mem_count, "messages_embedded": msg_count}


def _backfill_table(
    db: sqlite3.Connection,
    embedder,
    batch_size: int,
    *,
    source_table: str,
    id_column: str,
    text_column: str,
    vec_table: str,
) -> int:
    """Embed rows in *source_table* missing from *vec_table*.

    Processes in batches of *batch_size* to cap memory usage.
    Returns total rows embedded.
    """
    # Collect IDs already embedded.
    try:
        existing = set(
            r[0] for r in db.execute(f"SELECT rowid FROM {vec_table}").fetchall()
        )
    except sqlite3.OperationalError:
        existing = set()

    # Fetch rows with non-empty text that lack an embedding.
    rows = db.execute(
        f"SELECT {id_column} AS rid, {text_column} AS txt "
        f"FROM {source_table} "
        f"WHERE {text_column} IS NOT NULL AND {text_column} != ''"
    ).fetchall()

    to_embed = [(r["rid"], r["txt"]) for r in rows if r["rid"] not in existing]
    if not to_embed:
        return 0

    total = 0
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i : i + batch_size]
        ids = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        try:
            embeddings = embedder.embed_batch(texts)
        except Exception:
            log.exception("Embedding batch failed at offset %d", i)
            continue

        for rid, emb in zip(ids, embeddings):
            try:
                db.execute(
                    f"INSERT OR IGNORE INTO {vec_table}(rowid, embedding) VALUES (?, ?)",
                    (rid, emb),
                )
            except sqlite3.OperationalError:
                log.warning("Failed to insert vec row %d into %s", rid, vec_table)
        db.commit()
        total += len(batch)

    log.info("Backfilled %d rows into %s", total, vec_table)
    return total
