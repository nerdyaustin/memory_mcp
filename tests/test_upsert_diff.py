"""Diff-based upsert_session: rowids, vectors, and FTS must survive re-scans.

These tests exercise the real schema against a real SQLite file — no mocks.
They encode the v0.5.0 fix: re-indexing a changed session file must not
delete-and-reinsert messages, because that orphans every vector and forces
a full re-embed of the session on each scan of an active conversation.
"""

from __future__ import annotations

import struct

import pytest

from memory_mcp import db as db_mod


def _msg(mid: str, content: str, role: str = "user", **extra) -> dict:
    return {"id": mid, "role": role, "content": content, **extra}


def _session(messages: list[dict], mtime: float = 1.0) -> dict:
    return {
        "id": "sess-1",
        "source": "claude_code",
        "title": "test session",
        "cwd": "/repo",
        "model": "m",
        "started_at": "2026-07-10T00:00:00+00:00",
        "message_count": len(messages),
        "total_cost_usd": 0.0,
        "file_path": "/logs/sess-1.jsonl",
        "file_mtime": mtime,
        "messages": messages,
    }


def _vec(seed: float) -> bytes:
    v = [seed] * db_mod.EMBEDDING_DIM if hasattr(db_mod, "EMBEDDING_DIM") else [seed] * 384
    return struct.pack(f"<{len(v)}f", *v)


@pytest.fixture()
def conn(tmp_path):
    c = db_mod.init_db(tmp_path / "test.db")
    yield c
    c.close()


def _rowids(conn) -> dict[str, int]:
    return {
        r["id"]: r["rowid"]
        for r in conn.execute("SELECT rowid, id FROM messages")
    }


def test_unchanged_messages_keep_rowids_and_vectors(conn):
    db_mod.upsert_session(conn, _session([
        _msg("m1", "first message"),
        _msg("m2", "second message"),
    ]), "machine-1")
    before = _rowids(conn)

    if db_mod.VEC_AVAILABLE:
        conn.execute(
            "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
            (before["m1"], _vec(0.1)),
        )
        conn.execute(
            "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
            (before["m2"], _vec(0.2)),
        )
        conn.commit()

    # Re-scan: same messages plus a new one appended (the common case for
    # an active session file).
    db_mod.upsert_session(conn, _session([
        _msg("m1", "first message"),
        _msg("m2", "second message"),
        _msg("m3", "third message"),
    ], mtime=2.0), "machine-1")

    after = _rowids(conn)
    assert after["m1"] == before["m1"]
    assert after["m2"] == before["m2"]
    assert "m3" in after

    if db_mod.VEC_AVAILABLE:
        kept = {
            r[0] for r in conn.execute("SELECT rowid FROM vec_messages")
        }
        assert kept == {before["m1"], before["m2"]}, "vectors must survive re-scan"


def test_changed_message_updates_in_place_and_drops_stale_vector(conn):
    db_mod.upsert_session(conn, _session([
        _msg("m1", "partial streamed xyzzy", role="assistant"),
    ]), "machine-1")
    before = _rowids(conn)

    if db_mod.VEC_AVAILABLE:
        conn.execute(
            "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
            (before["m1"], _vec(0.3)),
        )
        conn.commit()

    db_mod.upsert_session(conn, _session([
        _msg("m1", "partial streamed response, now complete", role="assistant"),
    ], mtime=2.0), "machine-1")

    after = _rowids(conn)
    assert after["m1"] == before["m1"], "update must preserve rowid"

    row = conn.execute("SELECT content FROM messages WHERE id='m1'").fetchone()
    assert row["content"] == "partial streamed response, now complete"

    # FTS must reflect the new content, not the old: "xyzzy" appeared only
    # in the old content (search adds prefix wildcards, so the probe token
    # must not be a prefix of any new-content word).
    hits = db_mod.search_messages(conn, "complete")
    assert len(hits) == 1
    assert db_mod.search_messages(conn, "xyzzy") == []

    if db_mod.VEC_AVAILABLE:
        n = conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0]
        assert n == 0, "stale vector must be dropped so backfill re-embeds"


def test_removed_messages_are_deleted_with_vectors(conn):
    db_mod.upsert_session(conn, _session([
        _msg("m1", "keep me"),
        _msg("m2", "remove me"),
    ]), "machine-1")
    before = _rowids(conn)

    if db_mod.VEC_AVAILABLE:
        for mid in ("m1", "m2"):
            conn.execute(
                "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
                (before[mid], _vec(0.4)),
            )
        conn.commit()

    db_mod.upsert_session(
        conn, _session([_msg("m1", "keep me")], mtime=2.0), "machine-1",
    )

    assert set(_rowids(conn)) == {"m1"}
    assert db_mod.search_messages(conn, "remove") == []
    if db_mod.VEC_AVAILABLE:
        kept = {r[0] for r in conn.execute("SELECT rowid FROM vec_messages")}
        assert kept == {before["m1"]}


def test_rescan_marks_session_pending_push_again(conn):
    db_mod.upsert_session(conn, _session([_msg("m1", "hi")]), "machine-1")
    db_mod.mark_sessions_synced(conn, ["sess-1"])

    db_mod.upsert_session(
        conn, _session([_msg("m1", "hi"), _msg("m2", "more")], mtime=2.0),
        "machine-1",
    )
    row = conn.execute(
        "SELECT sync_status FROM sessions WHERE id='sess-1'"
    ).fetchone()
    assert row["sync_status"] == "pending_push"


def test_prune_orphan_vectors(conn):
    # VEC_AVAILABLE is only set once init_db has run (the conn fixture),
    # so this check must live in the test body, not a skipif decorator.
    if not db_mod.VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    db_mod.upsert_session(conn, _session([_msg("m1", "hello")]), "machine-1")
    rid = _rowids(conn)["m1"]

    conn.execute(
        "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)", (rid, _vec(0.5)),
    )
    # Orphans: rowids with no message behind them.
    for orphan_rid in (rid + 1000, rid + 1001):
        conn.execute(
            "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
            (orphan_rid, _vec(0.6)),
        )
    conn.commit()

    pruned = db_mod.prune_orphan_vectors(conn)
    assert pruned["vec_messages"] == 2
    kept = {r[0] for r in conn.execute("SELECT rowid FROM vec_messages")}
    assert kept == {rid}


def test_semantic_search_distance_cutoff(conn):
    if not db_mod.VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    db_mod.upsert_session(conn, _session([_msg("m1", "hello")]), "machine-1")
    rid = _rowids(conn)["m1"]

    # Unit vector along axis 0 for the stored message.
    stored = [0.0] * 384
    stored[0] = 1.0
    conn.execute(
        "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
        (rid, struct.pack("<384f", *stored)),
    )
    conn.commit()

    # Identical query -> distance 0 -> returned.
    near = db_mod.semantic_search_messages(conn, struct.pack("<384f", *stored))
    assert len(near) == 1 and near[0]["distance"] == pytest.approx(0.0)

    # Orthogonal query -> distance sqrt(2) ~ 1.41 -> filtered by the cutoff.
    ortho = [0.0] * 384
    ortho[1] = 1.0
    far = db_mod.semantic_search_messages(conn, struct.pack("<384f", *ortho))
    assert far == []

    # But still returned when the caller disables the cutoff.
    unfiltered = db_mod.semantic_search_messages(
        conn, struct.pack("<384f", *ortho), max_distance=None,
    )
    assert len(unfiltered) == 1


def test_failed_scan_upsert_rolls_back_partial_session(conn):
    """A rejected session must not poison the scanner's shared connection."""
    from memory_mcp.scanner import _index_session

    broken = _session([{"id": "missing-role"}])
    stats = {"files_indexed": 0, "errors": 0}

    _index_session(conn, broken, stats, "machine-1")

    assert stats == {"files_indexed": 0, "errors": 1}
    assert not conn.in_transaction
    assert conn.execute(
        "SELECT count(*) FROM sessions WHERE id = ?", (broken["id"],)
    ).fetchone()[0] == 0


def test_backfill_reads_and_embeds_in_bounded_batches(conn):
    if not db_mod.VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")

    messages = [_msg(f"m{i}", f"message {i}") for i in range(70)]
    db_mod.upsert_session(conn, _session(messages), "machine-1")

    class RecordingEmbedder:
        def __init__(self):
            self.batch_sizes: list[int] = []

        def embed_batch(self, texts: list[str]) -> list[bytes]:
            self.batch_sizes.append(len(texts))
            return [_vec(0.1) for _ in texts]

    embedder = RecordingEmbedder()
    result = db_mod.backfill_embeddings(conn, embedder)

    assert result["messages_embedded"] == 70
    assert embedder.batch_sizes == [32, 32, 6]

    embedder.batch_sizes.clear()
    assert db_mod.backfill_embeddings(conn, embedder)["messages_embedded"] == 0
    assert embedder.batch_sizes == []
