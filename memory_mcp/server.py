from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from memory_mcp.db import backfill_embeddings, checkpoint_wal, init_db
from memory_mcp.readiness import ReadinessState, init_readiness
from memory_mcp.scanner import scan_sessions
from memory_mcp.tools.memory import register_memory_tools
from memory_mcp.tools.sessions import register_session_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

PERIODIC_SCAN_INTERVAL_SECONDS = 120


def _run_scan_fresh_conn() -> dict:
    """Worker-thread entrypoint. SQLite connections are thread-bound, so this
    opens its own via init_db() rather than sharing the lifespan connection."""
    conn = init_db()
    try:
        stats = scan_sessions(conn)
        # Empty the WAL after each scan so it can never grow unbounded.
        checkpoint_wal(conn)
        return stats
    finally:
        conn.close()


def _run_backfill_fresh_conn(embedder) -> dict:
    """Worker-thread entrypoint mirroring _run_scan_fresh_conn."""
    conn = init_db()
    try:
        return backfill_embeddings(conn, embedder)
    finally:
        conn.close()


async def _initial_scan(state: ReadinessState, logger: logging.Logger) -> None:
    """First scan after lifespan yields. Failure is non-fatal — scan_done is
    set unconditionally so a waiting semantic caller is never blocked forever
    by a transient scan error; the periodic loop will retry."""
    async with state.maintenance_lock:
        try:
            stats = await asyncio.to_thread(_run_scan_fresh_conn)
            logger.info("Initial scan complete: %s", stats)
        except Exception:
            logger.exception("Initial scan failed; semantic search may be incomplete")
    state.scan_done.set()


async def _periodic_loop(state: ReadinessState, logger: logging.Logger) -> None:
    """Scan + conditional backfill every PERIODIC_SCAN_INTERVAL_SECONDS.
    Backfill only runs once the embedder has been loaded by a real semantic
    call — keeps cold sessions cheap when semantic search is never used."""
    while True:
        try:
            await asyncio.sleep(PERIODIC_SCAN_INTERVAL_SECONDS)
            async with state.maintenance_lock:
                try:
                    stats = await asyncio.to_thread(_run_scan_fresh_conn)
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


async def _background_startup(state: ReadinessState, logger: logging.Logger) -> None:
    await _initial_scan(state, logger)
    logger.info(
        "Periodic scan loop started (interval=%ds)",
        PERIODIC_SCAN_INTERVAL_SECONDS,
    )
    await _periodic_loop(state, logger)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Yields in milliseconds. Heavy work — initial scan, embedding model
    load, vector backfill — runs in a background task so MCP clients see
    tools/list immediately on startup."""
    logger = logging.getLogger("memory_mcp")
    logger.info("Initializing database...")
    db = init_db()

    state = ReadinessState.new()
    init_readiness(state)

    bg_task = asyncio.create_task(_background_startup(state, logger))
    logger.info("Background startup task scheduled; lifespan yielding")

    try:
        yield {"db": db}
    finally:
        bg_task.cancel()
        try:
            await bg_task
        except asyncio.CancelledError:
            pass
        db.close()
        logger.info("Database connection closed.")


mcp = FastMCP("MemoryMCP", lifespan=lifespan)
register_memory_tools(mcp)
register_session_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
