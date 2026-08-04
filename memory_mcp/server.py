from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP, Context

from memory_mcp.config import is_sync_enabled
from memory_mcp.db import backfill_embeddings, checkpoint_wal, connect_db, init_db
from memory_mcp.machine_id import get_machine_id
from memory_mcp.parent_watchdog import start_parent_watchdog
from memory_mcp.readiness import ReadinessState, init_readiness
from memory_mcp.scanner import scan_sessions
from memory_mcp.sync_engine import run_sync_loop, sync_now as _sync_now
from memory_mcp.tools.memory import register_memory_tools
from memory_mcp.tools.sessions import register_session_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
PERIODIC_SCAN_INTERVAL_SECONDS = 120


def _make_scan_fresh_conn(machine_id: str):
    """Return a callable that scans sessions with the given machine_id.

    Worker-thread entrypoint — SQLite connections are thread-bound, so
    this opens its own via connect_db() rather than sharing the lifespan
    connection.
    """
    def _run():
        conn = connect_db()
        try:
            stats = scan_sessions(conn, machine_id)
            checkpoint_wal(conn)
            return stats
        finally:
            conn.close()
    return _run


def _run_backfill_fresh_conn(embedder) -> dict:
    """Worker-thread entrypoint mirroring _run_scan_fresh_conn."""
    conn = connect_db()
    try:
        stats = backfill_embeddings(conn, embedder)
        checkpoint_wal(conn)
        return stats
    finally:
        conn.close()


async def _initial_scan(
    state: ReadinessState, logger: logging.Logger, scan_func,
) -> None:
    """First scan after lifespan yields. Failure is non-fatal — scan_done is
    set unconditionally so a waiting semantic caller is never blocked forever
    by a transient scan error; the periodic loop will retry."""
    async with state.maintenance_lock:
        try:
            stats = await asyncio.to_thread(scan_func)
            logger.info("Initial scan complete: %s", stats)
        except Exception:
            logger.exception("Initial scan failed; semantic search may be incomplete")
    state.scan_done.set()


async def _periodic_loop(
    state: ReadinessState, logger: logging.Logger, scan_func,
) -> None:
    """Scan + conditional backfill every PERIODIC_SCAN_INTERVAL_SECONDS.
    Backfill only runs once the embedder has been loaded by a real semantic
    call — keeps cold sessions cheap when semantic search is never used."""
    while True:
        try:
            await asyncio.sleep(PERIODIC_SCAN_INTERVAL_SECONDS)
            async with state.maintenance_lock:
                try:
                    stats = await asyncio.to_thread(scan_func)
                except Exception:
                    logger.exception("Periodic scan crashed; will retry")
                    continue
                if stats["files_indexed"] > 0 or stats["errors"] > 0:
                    logger.info("Periodic scan: %s", stats)
                if state.embedder is not None and stats["files_indexed"] > 0:
                    try:
                        bf = await asyncio.to_thread(
                            _run_backfill_fresh_conn, state.embedder
                        )
                        if bf.get("memories_embedded") or bf.get("messages_embedded"):
                            logger.info("Periodic backfill: %s", bf)
                    except Exception:
                        logger.exception("Periodic backfill crashed; will retry")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Periodic loop iteration crashed; will retry")


async def _background_startup(
    state: ReadinessState,
    logger: logging.Logger,
    scan_func,
    db_conn,
    machine_id: str,
) -> None:
    await _initial_scan(state, logger, scan_func)


    logger.info(
        "Periodic scan loop started (interval=%ds)",
        PERIODIC_SCAN_INTERVAL_SECONDS,
    )

    # Optionally start the sync engine.
    sync_task = None
    if is_sync_enabled():
        stop_sync = asyncio.Event()
        sync_task = asyncio.create_task(
            run_sync_loop(db_conn, stop_sync, machine_id)
        )
        logger.info("Sync engine started")

    try:
        await _periodic_loop(state, logger, scan_func)
    finally:
        if sync_task:
            stop_sync.set()
            sync_task.cancel()
            try:
                await sync_task
            except asyncio.CancelledError:
                pass


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Yields in milliseconds. Heavy work — initial scan, embedding model
    load, vector backfill — runs in a background task so MCP clients see
    tools/list immediately on startup."""
    logger = logging.getLogger("memory_mcp")
    logger.info("Initializing database...")
    db = init_db()

    machine_id = get_machine_id()
    logger.info("Machine ID: %s", machine_id)

    state = ReadinessState.new()
    init_readiness(state)

    scan_func = _make_scan_fresh_conn(machine_id)
    bg_task = asyncio.create_task(
        _background_startup(state, logger, scan_func, db, machine_id)
    )
    logger.info("Background startup task scheduled; lifespan yielding")

    try:
        yield {"db": db, "machine_id": machine_id}
    finally:
        bg_task.cancel()
        if state.backfill_task is not None:
            state.backfill_task.cancel()
            try:
                await state.backfill_task
            except asyncio.CancelledError:
                pass
        try:
            await bg_task
        except asyncio.CancelledError:
            pass
        db.close()
        logger.info("Database connection closed.")


mcp = FastMCP("MemoryMCP", lifespan=lifespan)
register_memory_tools(mcp)
register_session_tools(mcp)


def _register_sync_tool(mcp: FastMCP) -> None:
    """Register the sync_now tool (only functional when sync is configured)."""

    @mcp.tool(
        description=(
            "Manually trigger a full sync cycle (push + pull) with the "
            "configured sync server.  Returns a summary of what was "
            "pushed and pulled.  Does nothing if sync is not configured."
        ),
    )
    async def sync_now(ctx: Context = None) -> str:
        conn = ctx.request_context.lifespan_context["db"]
        machine_id = ctx.request_context.lifespan_context["machine_id"]
        return await _sync_now(conn, machine_id)

    return sync_now

_register_sync_tool(mcp)


def main():
    # Must come before mcp.run(): a stdio server whose client dies without
    # closing our stdin would otherwise linger and hold the SQLite write lock.
    start_parent_watchdog()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
