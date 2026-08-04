"""Cross-source session identity and file-ledger regressions."""

from __future__ import annotations

import json

from memory_mcp import db as db_mod
from memory_mcp.parsers.claude_code import ClaudeCodeParser
from memory_mcp.parsers.omp import OmpParser
from memory_mcp.scanner import scan_source


def _write_jsonl(path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


def _claude_user(session_id: str, message_id: str, *, agent_id: str | None = None) -> dict:
    entry = {
        "type": "user",
        "sessionId": session_id,
        "uuid": message_id,
        "timestamp": "2026-08-04T00:00:00Z",
        "message": {"role": "user", "content": f"message {message_id}"},
    }
    if agent_id is not None:
        entry["agentId"] = agent_id
        entry["isSidechain"] = True
    return entry


def test_claude_subagent_identity_is_scoped_to_parent(tmp_path):
    parent_id = "12181a18-45ad-4c48-91e5-849e6a71f78e"
    first = tmp_path / "subagents" / "agent-first.jsonl"
    second = tmp_path / "subagents" / "agent-second.jsonl"
    _write_jsonl(first, [_claude_user(parent_id, "m1", agent_id="first")])
    _write_jsonl(second, [_claude_user(parent_id, "m2", agent_id="second")])

    parser = ClaudeCodeParser()
    first_session = parser.parse_file(str(first))
    second_session = parser.parse_file(str(second))

    assert first_session is not None
    assert second_session is not None
    assert first_session.id == f"{parent_id}:agent:first"
    assert second_session.id == f"{parent_id}:agent:second"


def test_omp_uses_native_session_id_and_scopes_headerless_agent(tmp_path):
    parent_id = "019f28a9-eb24-7000-9b2e-16632bab4501"
    agent_dir = tmp_path / f"2026-07-03T15-47-32-262Z_{parent_id}"

    native = agent_dir / "NativeAgent.jsonl"
    native_id = "019f28d0-0d4b-7000-8a55-bfbe0e856fff"
    _write_jsonl(native, [
        {"type": "session", "id": native_id, "timestamp": "2026-08-04T00:00:00Z"},
        {
            "type": "message", "id": "m1", "parentId": None,
            "timestamp": "2026-08-04T00:00:01Z",
            "message": {"role": "user", "content": "native"},
        },
    ])

    legacy = agent_dir / "ReusableAgentName.jsonl"
    _write_jsonl(legacy, [{
        "type": "message", "id": "m2", "parentId": None,
        "timestamp": "2026-08-04T00:00:01Z",
        "message": {"role": "user", "content": "legacy"},
    }])

    parser = OmpParser()
    native_session = parser.parse_file(str(native))
    legacy_session = parser.parse_file(str(legacy))

    assert native_session is not None
    assert legacy_session is not None
    assert native_session.id == native_id
    assert legacy_session.id == f"{parent_id}:agent:ReusableAgentName"


def test_repeated_scan_does_not_reindex_sibling_agent_files(tmp_path):
    parent_id = "12181a18-45ad-4c48-91e5-849e6a71f78e"
    root = tmp_path / "claude-projects"
    _write_jsonl(root / f"{parent_id}.jsonl", [
        _claude_user(parent_id, "parent-message"),
    ])
    _write_jsonl(root / parent_id / "subagents" / "agent-first.jsonl", [
        _claude_user(parent_id, "first-message", agent_id="first"),
    ])
    _write_jsonl(root / parent_id / "subagents" / "agent-second.jsonl", [
        _claude_user(parent_id, "second-message", agent_id="second"),
    ])

    conn = db_mod.init_db(tmp_path / "memory.db")
    try:
        first = scan_source(conn, "claude_code", str(root), "machine")
        before = conn.execute(
            "SELECT max(rowid), count(*) FROM messages"
        ).fetchone()
        second = scan_source(conn, "claude_code", str(root), "machine")
        after = conn.execute(
            "SELECT max(rowid), count(*) FROM messages"
        ).fetchone()

        assert first["files_indexed"] == 3
        assert first["errors"] == 0
        assert second["files_indexed"] == 0
        assert second["files_skipped"] == 3
        assert tuple(after) == tuple(before)
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM indexed_files").fetchone()[0] == 3
    finally:
        conn.close()
