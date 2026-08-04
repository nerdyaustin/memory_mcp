"""Background sync engine for memory_mcp.

Runs a periodic push/pull loop alongside the MCP server.  All DB writes
go through the lifespan connection; HTTP I/O is offloaded to a thread
pool so the event loop never stalls.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import sqlite3

from memory_mcp import db
from memory_mcp.client import SyncClient
from memory_mcp.config import get_sync_config, is_sync_enabled
from memory_mcp.machine_id import get_machine_id

log = logging.getLogger(__name__)

# Sync at most once per interval.  The periodic loop also calls this,
# so it's fine to be conservative.
SYNC_INTERVAL_SECONDS = 60

# Push batches — avoid huge single requests.
PUSH_BATCH_SIZE = 50


async def run_sync_loop(
    db_conn: sqlite3.Connection,
    stop: asyncio.Event,
    machine_id: str,
) -> None:
    """Run the push/pull sync loop until *stop* is set.

    Intended to run as a background asyncio task.  Sleeps
    SYNC_INTERVAL_SECONDS between cycles.
    """
    cfg = get_sync_config()
    if not cfg:
        log.debug("Sync not configured; sync loop exiting")
        return


    db.claim_legacy_sync_rows(db_conn, machine_id)

    client = SyncClient(cfg["api_url"], cfg["api_key"])

    # Register this machine on first connection.
    hostname = platform.node()
    try:
        await asyncio.to_thread(client.register_machine, machine_id, hostname)
    except Exception:
        log.warning("Machine registration failed (will retry next cycle)")

    log.info(
        "Sync engine started — pushing to %s every %ds",
        cfg["api_url"], SYNC_INTERVAL_SECONDS,
    )

    while not stop.is_set():
        try:
            await _push(db_conn, client, machine_id)
        except Exception as exc:
            db_conn.rollback()
            log.warning("Sync push failed this cycle: %s", exc, exc_info=True)

        try:
            await _pull(db_conn, client, machine_id)
        except Exception as exc:
            db_conn.rollback()
            log.warning("Sync pull failed this cycle: %s", exc, exc_info=True)

        # Wait for the next cycle or early stop.
        try:
            await asyncio.wait_for(stop.wait(), timeout=SYNC_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass

    log.info("Sync engine stopped")


async def sync_now(
    db_conn: sqlite3.Connection, machine_id: str,
) -> str:
    """One-shot full sync (push + pull).  Returns a human-readable summary.

    Called by the ``sync_now`` MCP tool, and also on startup.
    """
    cfg = get_sync_config()
    if not cfg:
        return "Sync not configured. Set MEMORY_MCP_SYNC_URL and MEMORY_MCP_SYNC_KEY."


    db.claim_legacy_sync_rows(db_conn, machine_id)

    client = SyncClient(cfg["api_url"], cfg["api_key"])

    try:
        await asyncio.to_thread(client.register_machine, machine_id, platform.node())
    except Exception as exc:
        log.warning("Machine registration failed: %s", exc)

    push_result = "no data"
    try:
        push_result = await _push(db_conn, client, machine_id)
    except Exception as exc:
        db_conn.rollback()
        push_result = f"push failed: {exc}"

    pull_result = "no data"
    try:
        pull_result = await _pull(db_conn, client, machine_id)
    except Exception as exc:
        db_conn.rollback()
        pull_result = f"pull failed: {exc}"

    return f"Sync complete. Push: {push_result}. Pull: {pull_result}."


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------


async def _push(
    db_conn: sqlite3.Connection, client: SyncClient, machine_id: str,
) -> str:
    """Push pending sessions and memories to the sync server."""
    pending_sessions = db.get_pending_sessions(db_conn, machine_id)
    pending_memories = db.get_pending_memories(db_conn, machine_id)

    if not pending_sessions and not pending_memories:
        return "nothing to push"

    # Attach messages to sessions for the server.  Keep SQLite work on the
    # owning thread; only the blocking HTTP calls move to a worker thread.
    sessions_with_msgs: list[dict] = []
    for sess in pending_sessions:
        msgs = db.get_messages_for_session(db_conn, sess["id"])
        sess_copy = dict(sess)
        sess_copy["messages"] = msgs
        sessions_with_msgs.append(sess_copy)

    session_count = 0
    memory_count = 0

    # Push in batches to avoid giant requests.
    for i in range(0, len(sessions_with_msgs), PUSH_BATCH_SIZE):
        batch_sessions = sessions_with_msgs[i : i + PUSH_BATCH_SIZE]
        batch_memories = pending_memories[i : i + PUSH_BATCH_SIZE]

        resp = await asyncio.to_thread(
            client.push, machine_id, batch_sessions, batch_memories
        )
        accepted_sessions = resp.get("sessions_accepted", [])
        accepted_memories = resp.get("memories_accepted", [])

        if accepted_sessions:
            db.mark_sessions_synced(db_conn, accepted_sessions)
            session_count += len(accepted_sessions)
        if accepted_memories:
            db.mark_memories_synced(db_conn, accepted_memories)
            memory_count += len(accepted_memories)

    # Push any remaining memories that weren't batched with sessions.
    remaining_memories = pending_memories[len(sessions_with_msgs):]
    if remaining_memories:
        for i in range(0, len(remaining_memories), PUSH_BATCH_SIZE):
            batch = remaining_memories[i : i + PUSH_BATCH_SIZE]
            resp = await asyncio.to_thread(client.push, machine_id, [], batch)
            accepted = resp.get("memories_accepted", [])
            if accepted:
                db.mark_memories_synced(db_conn, accepted)
                memory_count += len(accepted)

    result = f"pushed {session_count} sessions"
    if memory_count:
        result += f", {memory_count} memories"
    log.info("Sync push: %s", result)
    return result


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------


async def _pull(
    db_conn: sqlite3.Connection, client: SyncClient, machine_id: str,
) -> str:
    """Pull sessions and memories from other machines."""
    last_pull = db.get_sync_state(db_conn, "last_pull_at")

    resp = await asyncio.to_thread(client.pull, machine_id, last_pull)
    server_ts = resp.get("server_ts", "")
    sessions = resp.get("sessions", [])
    memories = resp.get("memories", [])

    session_count = 0
    for sess in sessions:
        sid = sess["id"]
        remote_machine = sess.get("machine_id", "")
        msgs = sess.pop("messages", [])

        if not db.session_exists(db_conn, sid):
            db.upsert_pulled_session(db_conn, sess, remote_machine, msgs)
            session_count += 1

    memory_count = 0
    for mem in memories:
        remote_machine = mem.get("machine_id", "")
        if db.upsert_pulled_memory(db_conn, mem, remote_machine):
            memory_count += 1

    if server_ts:
        db.set_sync_state(db_conn, "last_pull_at", server_ts)

    if sessions or memories:
        result = f"pulled {session_count} sessions, {memory_count} memories"
        log.info("Sync pull: %s", result)
        return result
    return "no new data"
