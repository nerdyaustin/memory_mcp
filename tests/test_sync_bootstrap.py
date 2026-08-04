"""Regression tests for initial sync bootstrap of pre-sync local databases."""

import sqlite3

from memory_mcp import db as db_mod


def test_claim_legacy_rows_for_first_sync(tmp_path):
    """Migrated local rows with blank machine_id become pending upload rows."""
    db_path = tmp_path / "legacy.db"

    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            tags TEXT,
            context TEXT,
            source_session_id TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT,
            cwd TEXT,
            model TEXT,
            started_at TEXT,
            message_count INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0.0,
            file_path TEXT NOT NULL,
            file_mtime REAL NOT NULL
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            parent_id TEXT,
            role TEXT NOT NULL,
            content TEXT,
            thinking TEXT,
            tool_name TEXT,
            tool_input TEXT,
            tool_output TEXT,
            timestamp TEXT,
            model TEXT,
            cost_usd REAL
        );
        INSERT INTO sessions (
            id, source, title, cwd, model, started_at, message_count,
            total_cost_usd, file_path, file_mtime
        ) VALUES (
            'session-1', 'omp', 'Legacy session', '/repo', 'model',
            '2026-06-27T00:00:00+00:00', 1, 0.0, '/logs/session.jsonl', 123.0
        );
        INSERT INTO messages (
            id, session_id, role, content, timestamp, model
        ) VALUES (
            'message-1', 'session-1', 'user', 'hello from old db',
            '2026-06-27T00:00:01+00:00', 'model'
        );
        INSERT INTO memories (content, tags, context, source_session_id)
        VALUES ('legacy memory', '["sync"]', 'test', 'session-1');
        """
    )
    legacy.commit()
    legacy.close()

    conn = db_mod.init_db(db_path)
    try:
        counts = db_mod.claim_legacy_sync_rows(conn, "machine-1")

        assert counts == {"sessions": 1, "messages": 1, "memories": 1}
        assert db_mod.get_pending_sessions(conn, "machine-1")[0]["id"] == "session-1"
        row = conn.execute(
            "SELECT machine_id FROM messages WHERE id = 'message-1'"
        ).fetchone()
        assert row["machine_id"] == "machine-1"

        pending_memories = db_mod.get_pending_memories(conn, "machine-1")
        assert pending_memories[0]["content"] == "legacy memory"
        assert pending_memories[0]["global_id"]
    finally:
        conn.close()


def test_zero_row_legacy_claim_closes_write_transaction(tmp_path):
    """An idempotent claim must not retain SQLite's single writer lock."""
    db_path = tmp_path / "claimed.db"
    conn = db_mod.init_db(db_path)
    try:
        counts = db_mod.claim_legacy_sync_rows(conn, "machine-1")

        assert counts == {"sessions": 0, "messages": 0, "memories": 0}
        assert not conn.in_transaction

        peer = db_mod.connect_db(db_path)
        try:
            peer.execute("BEGIN IMMEDIATE")
            peer.rollback()
        finally:
            peer.close()
    finally:
        conn.close()


def test_initialized_database_reopens_during_active_writer(tmp_path):
    """Ordinary startup must not repeat schema DDL after initialization."""
    db_path = tmp_path / "initialized.db"
    owner = db_mod.init_db(db_path)
    try:
        owner.execute("BEGIN IMMEDIATE")
        peer = db_mod.init_db(db_path)
        try:
            assert peer.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        finally:
            peer.close()
            owner.rollback()
    finally:
        owner.close()


def test_concurrent_first_openers_initialize_once(tmp_path):
    """Concurrent server openers must safely share first-time initialization."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    db_path = tmp_path / "concurrent.db"
    barrier = Barrier(4)

    def open_once() -> int:
        barrier.wait()
        conn = db_mod.init_db(db_path)
        try:
            return conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(lambda _index: open_once(), range(4)))

    assert counts == [0, 0, 0, 0]


def test_sync_now_keeps_sqlite_on_owner_thread(tmp_path, monkeypatch):
    """Configured sync must not move the SQLite connection into a worker."""
    import asyncio

    from memory_mcp import sync_engine

    class RecordingClient:
        def __init__(self, api_url, api_key):
            self.pushed_sessions = []

        def register_machine(self, machine_id, hostname):
            return {"registered": True, "machine_id": machine_id}

        def push(self, machine_id, sessions, memories):
            self.pushed_sessions.extend(sessions)
            return {
                "server_ts": "2026-06-27T00:00:02+00:00",
                "sessions_accepted": [s["id"] for s in sessions],
                "memories_accepted": [m["global_id"] for m in memories],
            }

        def pull(self, machine_id, since=None):
            return {
                "server_ts": "2026-06-27T00:00:03+00:00",
                "sessions": [],
                "memories": [],
            }

    client = RecordingClient("http://sync", "secret")
    monkeypatch.setattr(
        sync_engine, "get_sync_config",
        lambda: {"api_url": "http://sync", "api_key": "secret"},
    )
    monkeypatch.setattr(sync_engine, "SyncClient", lambda *_args: client)

    conn = db_mod.init_db(tmp_path / "sync.db")
    try:
        db_mod.upsert_session(conn, {
            "id": "session-1",
            "source": "omp",
            "title": "Thread ownership",
            "cwd": "/repo",
            "model": "model",
            "started_at": "2026-06-27T00:00:00+00:00",
            "message_count": 1,
            "total_cost_usd": 0.0,
            "file_path": "/logs/session.jsonl",
            "file_mtime": 123.0,
            "messages": [
                {
                    "id": "message-1",
                    "session_id": "session-1",
                    "role": "user",
                    "content": "hello",
                }
            ],
        }, "machine-1")

        result = asyncio.run(sync_engine.sync_now(conn, "machine-1"))

        assert "push failed" not in result
        assert "pushed 1 sessions" in result
        assert client.pushed_sessions[0]["messages"][0]["id"] == "message-1"
        assert db_mod.get_pending_sessions(conn, "machine-1") == []
    finally:
        conn.close()
