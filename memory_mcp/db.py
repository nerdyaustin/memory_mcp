"""SQLite database with FTS5 and sqlite-vec for memory and session storage.

Every public function takes an explicit ``sqlite3.Connection`` so callers
control the lifetime (the MCP server passes it via lifespan context).

Vector search (semantic) is optional: if sqlite-vec or fastembed is not
installed, the server falls back to FTS5-only mode with no loss of
existing functionality.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import sqlite3
from pathlib import Path
from time import monotonic, sleep

from .config import get_db_path

log = logging.getLogger(__name__)

# Set True once sqlite-vec loads successfully on any connection.
# Checked by vec-dependent functions so they can no-op gracefully.
VEC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA_VERSION = 3


_SCHEMA = """\
-- Explicit memories (the primary feature)
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id   TEXT,           -- UUID for cross-machine sync
    content     TEXT NOT NULL,
    tags        TEXT,           -- JSON array of strings
    context     TEXT,           -- what prompted this memory
    source_session_id TEXT,
    machine_id  TEXT NOT NULL DEFAULT '',
    sync_status TEXT NOT NULL DEFAULT 'pending_push',
    server_updated_at TEXT,
    updated_at  TEXT,
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
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content, tags, context ON memories BEGIN
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
    file_mtime      REAL    NOT NULL,
    machine_id      TEXT    NOT NULL DEFAULT '',
    sync_status     TEXT    NOT NULL DEFAULT 'pending_push',
    server_updated_at TEXT
);

-- File-level scan state. A logical session may have several source files
-- (for example a parent conversation and multiple agent logs), so file mtimes
-- cannot be stored reliably on the sessions row alone.
CREATE TABLE IF NOT EXISTS indexed_files (
    file_path   TEXT PRIMARY KEY,
    file_mtime REAL NOT NULL,
    source      TEXT NOT NULL
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
    cost_usd    REAL,
    machine_id  TEXT NOT NULL DEFAULT ''
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
CREATE TRIGGER IF NOT EXISTS messages_au
AFTER UPDATE OF content, thinking, tool_name, tool_input, tool_output ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, thinking, tool_name, tool_input, tool_output)
    VALUES ('delete', old.rowid, old.content, old.thinking, old.tool_name, old.tool_input, old.tool_output);
    INSERT INTO messages_fts(rowid, content, thinking, tool_name, tool_input, tool_output)
    VALUES (new.rowid, new.content, new.thinking, new.tool_name, new.tool_input, new.tool_output);
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


def _enable_wal(db: sqlite3.Connection) -> None:
    """Enable WAL, tolerating concurrent first-time database openers."""
    deadline = monotonic() + 5.0
    while True:
        mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        if mode.lower() == "wal":
            return
        try:
            mode = db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode.lower() == "wal":
                return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or monotonic() >= deadline:
                raise
        if monotonic() >= deadline:
            raise sqlite3.OperationalError("timed out enabling WAL journal mode")
        sleep(0.05)


def _open_db(path: Path | None = None) -> tuple[sqlite3.Connection, bool]:
    """Open and configure a connection without changing the schema."""
    db_path = path or get_db_path()
    db = sqlite3.connect(str(db_path))
    try:
        db.row_factory = sqlite3.Row
        # Install the wait policy before any pragma that may need a lock.
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("PRAGMA foreign_keys=ON")
        _enable_wal(db)
        db.execute("PRAGMA journal_size_limit=67108864")  # 64 MB
        vec_loaded = _load_vec(db)
        if vec_loaded:
            log.info("sqlite-vec loaded — semantic search enabled")
        return db, vec_loaded
    except BaseException:
        db.close()
        raise


def connect_db(path: Path | None = None) -> sqlite3.Connection:
    """Open an existing database without running schema setup or migrations."""
    db, _vec_loaded = _open_db(path)
    return db


def init_db(path: Path | None = None) -> sqlite3.Connection:
    """Open a database and apply pending schema setup exactly once per version."""
    db, vec_loaded = _open_db(path)
    try:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version >= _SCHEMA_VERSION:
            return db

        # Serialize first-time setup and migrations across concurrent servers,
        # then re-check because another process may have upgraded while we waited.
        db.execute("BEGIN IMMEDIATE")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version < _SCHEMA_VERSION:
            _execute_schema_script(db, _SCHEMA)
            _refresh_fts_triggers(db)
            _migrate_schema(db)
            db.execute(
                "CREATE TABLE IF NOT EXISTS sync_state ("
                " key TEXT PRIMARY KEY, value TEXT)"
            )
            if vec_loaded:
                _execute_schema_script(db, _VEC_SCHEMA)
            db.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        db.commit()
    except BaseException:
        db.rollback()
        db.close()
        raise
    return db


def _execute_schema_script(db: sqlite3.Connection, script: str) -> None:
    """Execute a SQL script without ``executescript``'s implicit commit."""
    pending: list[str] = []
    for line in script.splitlines():
        pending.append(line)
        statement = "\n".join(pending).strip()
        if statement and sqlite3.complete_statement(statement):
            db.execute(statement)
            pending.clear()
    if any(line.strip() for line in pending):
        raise sqlite3.OperationalError("incomplete schema statement")


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

def _refresh_fts_triggers(db: sqlite3.Connection) -> None:
    """Replace broad FTS update triggers with content-only variants.

    Sync metadata updates should not churn FTS rows. Recreating the trigger is
    also required for migrated databases that already have the older broad
    ``AFTER UPDATE`` trigger.
    """
    db.execute("DROP TRIGGER IF EXISTS memories_au")
    db.execute(
        """
        CREATE TRIGGER memories_au
        AFTER UPDATE OF content, tags, context ON memories
        BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags, context)
            VALUES ('delete', old.id, old.content, old.tags, old.context);
            INSERT INTO memories_fts(rowid, content, tags, context)
            VALUES (new.id, new.content, new.tags, new.context);
        END
        """
    )


# ---------------------------------------------------------------------------
# Schema migration (v0.3.0 → v0.4.0 sync columns)
# ---------------------------------------------------------------------------


def _migrate_schema(db: sqlite3.Connection) -> None:
    """Apply idempotent schema migrations for existing databases."""
    _add_column_if_missing(db, "sessions", "machine_id",
                           "TEXT NOT NULL DEFAULT ''")
    _add_column_if_missing(db, "sessions", "sync_status",
                           "TEXT NOT NULL DEFAULT 'pending_push'")
    _add_column_if_missing(db, "sessions", "server_updated_at", "TEXT")
    _add_column_if_missing(db, "messages", "machine_id",
                           "TEXT NOT NULL DEFAULT ''")
    _add_column_if_missing(db, "memories", "global_id", "TEXT")
    _add_column_if_missing(db, "memories", "machine_id",
                           "TEXT NOT NULL DEFAULT ''")
    _add_column_if_missing(db, "memories", "sync_status",
                           "TEXT NOT NULL DEFAULT 'pending_push'")
    _add_column_if_missing(db, "memories", "server_updated_at", "TEXT")
    _add_column_if_missing(db, "memories", "updated_at", "TEXT")

    # Ensure unique index on global_id (CREATE UNIQUE INDEX IF NOT EXISTS
    # is safe for new DBs; for migrated DBs we create it if missing).
    db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS "
        "idx_memories_global_id ON memories(global_id)"
    )

    # Backfill global_id for existing memories that lack one.
    import uuid as _uuid
    rows = db.execute(
        "SELECT id FROM memories WHERE global_id IS NULL"
    ).fetchall()
    for (mid,) in rows:
        db.execute(
            "UPDATE memories SET global_id = ? WHERE id = ?",
            (str(_uuid.uuid4()), mid),
        )
    if rows:
        log.info("Backfilled global_id for %d existing memories", len(rows))

    db.execute(
        "CREATE TABLE IF NOT EXISTS indexed_files ("
        "file_path TEXT PRIMARY KEY, "
        "file_mtime REAL NOT NULL, "
        "source TEXT NOT NULL)"
    )
    # Seed the independent file ledger from legacy session rows. Collision
    # losers are intentionally absent and will be parsed once after upgrade.
    db.execute(
        "INSERT OR IGNORE INTO indexed_files(file_path, file_mtime, source) "
        "SELECT file_path, file_mtime, source FROM sessions"
    )
    # Before canonical agent IDs existed, a subagent could be the last writer
    # of its parent's session row. Do not let that legacy path look processed:
    # it must be parsed once under its own agent identity after upgrade.
    db.execute(
        "DELETE FROM indexed_files "
        "WHERE source = 'claude_code' "
        "AND replace(file_path, '\\', '/') LIKE '%/subagents/%' "
        "AND NOT EXISTS ("
        "SELECT 1 FROM sessions s "
        "WHERE s.file_path = indexed_files.file_path "
        "AND s.id LIKE '%:agent:%'"
        ")"
    )


def claim_legacy_sync_rows(db: sqlite3.Connection, machine_id: str) -> dict[str, int]:
    """Assign pre-sync local rows to *machine_id* so first sync uploads them.

    Databases created before sync support have blank ``machine_id`` values after
    migration.  The scanner skips unchanged files, so those rows would otherwise
    stay invisible to ``get_pending_*`` queries and never reach the hosted
    service.  Blank machine IDs can only be legacy local rows, so claiming them
    is safe and idempotent.
    """
    if not machine_id:
        return {"sessions": 0, "messages": 0, "memories": 0}

    # A no-op UPDATE still opens a SQLite write transaction. The connection
    # context commits even when every rowcount is zero and rolls back on error.
    with db:
        sessions_cur = db.execute(
            "UPDATE sessions "
            "SET machine_id = ?, sync_status = 'pending_push' "
            "WHERE machine_id = '' OR machine_id IS NULL",
            (machine_id,),
        )
        messages_cur = db.execute(
            "UPDATE messages SET machine_id = ? "
            "WHERE machine_id = '' OR machine_id IS NULL",
            (machine_id,),
        )
        memories_cur = db.execute(
            "UPDATE memories "
            "SET machine_id = ?, sync_status = 'pending_push' "
            "WHERE machine_id = '' OR machine_id IS NULL",
            (machine_id,),
        )

        counts = {
            "sessions": max(sessions_cur.rowcount, 0),
            "messages": max(messages_cur.rowcount, 0),
            "memories": max(memories_cur.rowcount, 0),
        }

    if any(counts.values()):
        log.info("Claimed legacy sync rows for %s: %s", machine_id, counts)
    return counts


def _add_column_if_missing(
    db: sqlite3.Connection, table: str, column: str, typedef: str,
) -> None:
    """Add *column* to *table* if it doesn't already exist."""
    existing = {
        r[1]
        for r in db.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {typedef}")
        log.info("Migrated: added %s.%s (%s)", table, column, typedef)


# ---------------------------------------------------------------------------
# Sync state helpers
# ---------------------------------------------------------------------------


def get_sync_state(db: sqlite3.Connection, key: str) -> str | None:
    """Read a value from the sync_state table."""
    row = db.execute(
        "SELECT value FROM sync_state WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else None


def set_sync_state(db: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert a key-value pair into sync_state."""
    db.execute(
        "INSERT OR REPLACE INTO sync_state (key, value) VALUES (?, ?)",
        (key, value),
    )
    db.commit()


# ---------------------------------------------------------------------------
# Sync push queries
# ---------------------------------------------------------------------------


def get_pending_sessions(
    db: sqlite3.Connection, machine_id: str,
) -> list[dict]:
    """Return sessions with pending_push status for this machine."""
    rows = db.execute(
        "SELECT id, source, title, cwd, model, started_at, "
        "message_count, total_cost_usd, file_path, file_mtime "
        "FROM sessions "
        "WHERE sync_status = 'pending_push' AND machine_id = ?",
        (machine_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_messages_for_session(
    db: sqlite3.Connection, session_id: str,
) -> list[dict]:
    """Return all messages for a session."""
    rows = db.execute(
        "SELECT id, session_id, parent_id, role, content, thinking, "
        "tool_name, tool_input, tool_output, timestamp, model, cost_usd "
        "FROM messages WHERE session_id = ? ORDER BY rowid",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_pending_memories(
    db: sqlite3.Connection, machine_id: str,
) -> list[dict]:
    """Return memories with pending_push status for this machine."""
    rows = db.execute(
        "SELECT id, global_id, content, tags, context, "
        "source_session_id, created_at, updated_at "
        "FROM memories "
        "WHERE sync_status = 'pending_push' AND machine_id = ?",
        (machine_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_sessions_synced(
    db: sqlite3.Connection, session_ids: list[str],
) -> None:
    """Mark sessions as synced after successful push."""
    if not session_ids:
        return
    placeholders = ",".join("?" * len(session_ids))
    db.execute(
        f"UPDATE sessions SET sync_status = 'synced' "
        f"WHERE id IN ({placeholders})",
        session_ids,
    )
    db.commit()


def mark_memories_synced(
    db: sqlite3.Connection, global_ids: list[str],
) -> None:
    """Mark memories as synced after successful push."""
    if not global_ids:
        return
    placeholders = ",".join("?" * len(global_ids))
    db.execute(
        f"UPDATE memories SET sync_status = 'synced' "
        f"WHERE global_id IN ({placeholders})",
        global_ids,
    )
    db.commit()


# ---------------------------------------------------------------------------
# Sync pull helpers
# ---------------------------------------------------------------------------


def upsert_pulled_session(
    db: sqlite3.Connection, session: dict, machine_id: str,
    messages: list[dict],
) -> None:
    """Insert a session + messages pulled from the sync server.

    Unlike local ``upsert_session`` this does NOT delete existing
    messages — pulled sessions are new to this machine.  Uses
    INSERT OR REPLACE so re-pulls are idempotent.
    """
    sid = session["id"]

    db.execute(
        "INSERT OR REPLACE INTO sessions "
        "(id, source, title, cwd, model, started_at, message_count, "
        "total_cost_usd, file_path, file_mtime, machine_id, sync_status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'synced')",
        (
            sid,
            session.get("source", ""),
            session.get("title"),
            session.get("cwd"),
            session.get("model"),
            session.get("started_at"),
            session.get("message_count", 0),
            session.get("total_cost_usd", 0.0),
            session.get("file_path", ""),
            session.get("file_mtime", 0.0),
            machine_id,
        ),
    )

    for msg in messages:
        db.execute(
            "INSERT OR IGNORE INTO messages "
            "(id, session_id, parent_id, role, content, thinking, "
            "tool_name, tool_input, tool_output, timestamp, model, "
            "cost_usd, machine_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                machine_id,
            ),
        )
    db.commit()


def upsert_pulled_memory(
    db: sqlite3.Connection, memory: dict, machine_id: str,
) -> bool:
    """Insert or update a memory pulled from the sync server.

    Returns True if the memory was inserted/updated, False if the
    local version is newer (last-write-wins conflict resolution).
    """
    gid = memory["global_id"]

    # Check if a newer local version exists.
    existing = db.execute(
        "SELECT updated_at FROM memories WHERE global_id = ?", (gid,)
    ).fetchone()

    if existing and existing["updated_at"]:
        local_ts = existing["updated_at"]
        remote_ts = memory.get("updated_at", "")
        if local_ts and remote_ts and local_ts > remote_ts:
            return False  # Local is newer, keep it.

    db.execute(
        "INSERT OR REPLACE INTO memories "
        "(global_id, content, tags, context, source_session_id, "
        "machine_id, sync_status, updated_at, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'synced', ?, ?)",
        (
            gid,
            memory.get("content", ""),
            memory.get("tags"),
            memory.get("context"),
            memory.get("source_session_id"),
            machine_id,
            memory.get("updated_at"),
            memory.get("created_at"),
        ),
    )
    db.commit()
    return True


def session_exists(db: sqlite3.Connection, session_id: str) -> bool:
    """Check if a session already exists locally."""
    row = db.execute(
        "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    return row is not None


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


# Machine identity helper
def _machine_id_or_default(machine_id: str) -> str:
    """Return an explicit machine_id or the host's persistent identity."""
    if machine_id:
        return machine_id
    try:
        from memory_mcp.machine_id import get_machine_id

        return get_machine_id()
    except Exception:
        log.debug("Could not determine machine_id", exc_info=True)
        return ""


# Memory CRUD
# ---------------------------------------------------------------------------


def save_memory(
    db: sqlite3.Connection,
    content: str,
    tags: list[str] | None = None,
    context: str | None = None,
    source_session_id: str | None = None,
    embedding: bytes | None = None,
    machine_id: str = "",
) -> int:
    """Store a memory note.  Returns the new row id.

    If *embedding* (serialized float32 blob) is provided and sqlite-vec is
    loaded, the vector is stored in ``vec_memories`` for semantic search.
    """
    import uuid

    machine_id = _machine_id_or_default(machine_id)
    global_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tags_json = json.dumps(tags) if tags else None
    cur = db.execute(
        "INSERT INTO memories (global_id, content, tags, context, "
        "source_session_id, machine_id, sync_status, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'pending_push', ?)",
        (global_id, content, tags_json, context, source_session_id,
         machine_id, now),
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


# KNN always returns the k nearest rows no matter how far away they are, so
# without a cutoff an unrelated memory tops the list whenever nothing better
# exists. vec0 distance here is L2 over unit-normalised bge vectors, so
# d = sqrt(2 - 2*cos_sim): 0.95 corresponds to cosine similarity ~0.55 —
# loosely related. Anything beyond that is noise.
DEFAULT_MAX_DISTANCE = 0.95


def semantic_search_memories(
    db: sqlite3.Connection,
    query_embedding: bytes,
    tags: list[str] | None = None,
    limit: int = 10,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
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
    if max_distance is not None:
        results = [r for r in results if r["distance"] <= max_distance]

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


# Message columns compared to decide whether a re-parsed message actually
# changed. Order matches the SELECT in upsert_session and the UPDATE below.
_MSG_FIELDS = (
    "parent_id", "role", "content", "thinking", "tool_name",
    "tool_input", "tool_output", "timestamp", "model", "cost_usd",
)


def _delete_vec_rows(db: sqlite3.Connection, vec_table: str, rowids: list[int]) -> None:
    """Remove vector entries for the given rowids (vec0 has no triggers)."""
    if not VEC_AVAILABLE or not rowids:
        return
    try:
        for i in range(0, len(rowids), 500):
            batch = rowids[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            db.execute(
                f"DELETE FROM {vec_table} WHERE rowid IN ({placeholders})",
                batch,
            )
    except sqlite3.OperationalError:
        pass


def upsert_session(
    db: sqlite3.Connection, session: dict, machine_id: str = "",
) -> None:
    """Insert or update a parsed session and its messages.

    Diffs against existing rows keyed on the stable message id instead of
    deleting and re-inserting everything. Unchanged messages keep their
    rowid, so their FTS entries and vec_messages embeddings survive a
    re-scan — previously every re-scan of an active session orphaned all
    its vectors and forced a full re-embed.
    """
    sid = session["id"]
    machine_id = _machine_id_or_default(machine_id)

    db.execute(
        "INSERT INTO sessions "
        "(id, source, title, cwd, model, started_at, message_count, "
        "total_cost_usd, file_path, file_mtime, machine_id, sync_status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending_push') "
        "ON CONFLICT(id) DO UPDATE SET "
        "source=excluded.source, title=excluded.title, cwd=excluded.cwd, "
        "model=excluded.model, started_at=excluded.started_at, "
        "message_count=excluded.message_count, "
        "total_cost_usd=excluded.total_cost_usd, "
        "file_path=excluded.file_path, file_mtime=excluded.file_mtime, "
        "machine_id=excluded.machine_id, sync_status='pending_push'",
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
            machine_id,
        ),
    )

    existing = {
        r["id"]: r
        for r in db.execute(
            f"SELECT rowid, id, {', '.join(_MSG_FIELDS)} "
            "FROM messages WHERE session_id = ?",
            (sid,),
        )
    }

    seen_ids: set[str] = set()
    stale_vec_rowids: list[int] = []

    for msg in session.get("messages", []):
        mid = msg["id"]
        seen_ids.add(mid)
        values = (
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
        )
        old = existing.get(mid)
        if old is None:
            db.execute(
                "INSERT OR IGNORE INTO messages "
                "(id, session_id, parent_id, role, content, thinking, "
                "tool_name, tool_input, tool_output, timestamp, model, "
                "cost_usd, machine_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mid, sid, *values, machine_id),
            )
        elif tuple(old[f] for f in _MSG_FIELDS) != values:
            # Content changed (e.g. a streamed assistant message finished):
            # update in place so the rowid survives, and drop the now-stale
            # vector so backfill re-embeds just this message.
            db.execute(
                f"UPDATE messages SET {', '.join(f'{f} = ?' for f in _MSG_FIELDS)} "
                "WHERE rowid = ?",
                (*values, old["rowid"]),
            )
            if old["content"] != msg.get("content"):
                stale_vec_rowids.append(old["rowid"])

    # Messages that disappeared from the file (rare, e.g. truncation).
    removed = [r["rowid"] for mid, r in existing.items() if mid not in seen_ids]
    if removed:
        for i in range(0, len(removed), 500):
            batch = removed[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            db.execute(
                f"DELETE FROM messages WHERE rowid IN ({placeholders})", batch,
            )
        stale_vec_rowids.extend(removed)

    _delete_vec_rows(db, "vec_messages", stale_vec_rowids)
    db.commit()


def get_session_mtime(db: sqlite3.Connection, file_path: str) -> float | None:
    """Return the stored mtime for a source file, or None if unseen."""
    row = db.execute(
        "SELECT file_mtime FROM indexed_files WHERE file_path = ?", (file_path,)
    ).fetchone()
    return float(row["file_mtime"]) if row else None


def record_indexed_file(
    db: sqlite3.Connection, file_path: str, file_mtime: float, source: str,
) -> None:
    """Record a successfully processed source file independently of sessions."""
    db.execute(
        "INSERT INTO indexed_files(file_path, file_mtime, source) VALUES (?, ?, ?) "
        "ON CONFLICT(file_path) DO UPDATE SET "
        "file_mtime=excluded.file_mtime, source=excluded.source",
        (file_path, file_mtime, source),
    )
    db.commit()


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
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
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

    results = [dict(r) for r in rows[offset:]]
    if max_distance is not None:
        results = [r for r in results if r["distance"] <= max_distance]
    return results


# ---------------------------------------------------------------------------
# Embedding backfill
# ---------------------------------------------------------------------------


def prune_orphan_vectors(db: sqlite3.Connection) -> dict:
    """Delete vector rows whose source row no longer exists.

    vec0 tables have no triggers, so any code path that removes a message
    or memory without cleaning its vector leaves an orphan behind. Orphans
    slow every KNN scan and — worse — silently shrink result pages, because
    orphaned neighbours vanish in the JOIN back to the source table.
    """
    if not VEC_AVAILABLE:
        return {"vec_messages": 0, "vec_memories": 0}

    pruned = {}
    for vec_table, src_table, id_col in (
        ("vec_messages", "messages", "rowid"),
        ("vec_memories", "memories", "id"),
    ):
        try:
            vec_ids = set(
                r[0] for r in db.execute(f"SELECT rowid FROM {vec_table}")
            )
            src_ids = set(
                r[0] for r in db.execute(f"SELECT {id_col} FROM {src_table}")
            )
        except sqlite3.OperationalError as exc:
            log.warning("Vector prune skipped for %s: %s", vec_table, exc)
            pruned[vec_table] = 0
            continue

        orphans = list(vec_ids - src_ids)
        for i in range(0, len(orphans), 500):
            batch = orphans[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            db.execute(
                f"DELETE FROM {vec_table} WHERE rowid IN ({placeholders})",
                batch,
            )
            db.commit()
        pruned[vec_table] = len(orphans)
        if orphans:
            log.info("Pruned %d orphaned vectors from %s", len(orphans), vec_table)
    return pruned


def backfill_embeddings(
    db: sqlite3.Connection,
    embedder,
    batch_size: int = 32,
) -> dict:
    """Embed memories and messages that don't yet have vectors.

    Called after scan on server startup and after periodic re-scans.
    The *embedder* must have an ``embed_batch(texts) -> list[bytes]``
    method (see ``embeddings.Embedder``).

    Returns counts of newly embedded rows.
    """
    if not VEC_AVAILABLE:
        return {"memories_embedded": 0, "messages_embedded": 0}

    prune_orphan_vectors(db)

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
    total = 0
    last_id = -1
    while True:
        try:
            rows = db.execute(
                f"SELECT s.{id_column} AS rid, s.{text_column} AS txt "
                f"FROM {source_table} s "
                f"LEFT JOIN {vec_table} v ON v.rowid = s.{id_column} "
                f"WHERE s.{id_column} > ? "
                f"AND s.{text_column} IS NOT NULL "
                f"AND s.{text_column} != '' "
                f"AND v.rowid IS NULL "
                f"ORDER BY s.{id_column} LIMIT ?",
                (last_id, batch_size),
            ).fetchall()
        except sqlite3.OperationalError:
            log.warning("Failed to query missing vectors in %s", vec_table)
            return total

        if not rows:
            break

        last_id = rows[-1]["rid"]
        ids = [r["rid"] for r in rows]
        texts = [r["txt"] for r in rows]
        try:
            embeddings = embedder.embed_batch(texts)
        except Exception:
            log.exception("Embedding batch failed after rowid %d", last_id)
            continue

        inserted = 0
        for rid, emb in zip(ids, embeddings):
            try:
                db.execute(
                    f"INSERT OR IGNORE INTO {vec_table}(rowid, embedding) VALUES (?, ?)",
                    (rid, emb),
                )
                inserted += 1
            except sqlite3.OperationalError:
                log.warning("Failed to insert vec row %d into %s", rid, vec_table)
        db.commit()
        total += inserted
    log.info("Backfilled %d rows into %s", total, vec_table)
    return total
