"""Semantic-readiness coordination for lazy startup.

Holds a lifespan-owned ReadinessState carrying the lazy Embedder, an
initial-scan gate, a backfill gate, and a maintenance lock shared between
the periodic scan and background backfill. State is constructed inside
``lifespan`` (not at module import) so asyncio primitives bind to the
running event loop. Tools reach the state via module-level helpers after
``init_readiness`` has been called once during startup.

The embedding model stays unloaded until the first semantic query. That query
waits only for the model load; vector backfill runs as a separate task and
searches use whatever vectors already exist. This keeps ordinary MCP startup
cheap and avoids making a first semantic call wait for a potentially
multi-minute corpus backfill.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from memory_mcp import db as db_mod
from memory_mcp.db import backfill_embeddings, checkpoint_wal, connect_db
from memory_mcp.embeddings import AVAILABLE as EMBED_AVAILABLE, Embedder

log = logging.getLogger(__name__)


@dataclass
class ReadinessState:
    scan_done: asyncio.Event
    backfill_done: asyncio.Event
    maintenance_lock: asyncio.Lock
    embed_load_lock: asyncio.Lock
    embedder: Optional[Embedder] = None
    backfill_task: Optional[asyncio.Task[None]] = None

    @classmethod
    def new(cls) -> "ReadinessState":
        """Construct inside a running loop so asyncio primitives bind correctly."""
        return cls(
            scan_done=asyncio.Event(),
            backfill_done=asyncio.Event(),
            maintenance_lock=asyncio.Lock(),
            embed_load_lock=asyncio.Lock(),
        )


_state: Optional[ReadinessState] = None


def init_readiness(state: ReadinessState) -> None:
    """Register the lifespan-owned state so tools can reach it."""
    global _state
    _state = state


def get_state() -> ReadinessState:
    if _state is None:
        raise RuntimeError(
            "readiness.init_readiness() was not called; lifespan did not initialise."
        )
    return _state


def _run_backfill_fresh_conn(embedder: Embedder) -> dict:
    """Worker-thread entrypoint. SQLite connections are thread-bound, so the
    background thread opens its own via connect_db() rather than sharing the
    lifespan connection."""
    conn = connect_db()
    try:
        stats = backfill_embeddings(conn, embedder)
        checkpoint_wal(conn)
        return stats
    finally:
        conn.close()


async def ensure_semantic_ready() -> Embedder:
    """Return the loaded Embedder, loading it on the first semantic query.

    Deliberately does not wait for the initial scan or vector backfill: a
    query over a partially embedded corpus returns useful results immediately,
    while the background task catches up missing vectors.
    """
    state = get_state()

    if state.embedder is not None:
        return state.embedder

    async with state.embed_load_lock:
        if state.embedder is None:
            if not EMBED_AVAILABLE or not db_mod.VEC_AVAILABLE:
                raise RuntimeError(
                    "Semantic search unavailable: fastembed or sqlite-vec missing."
                )
            state.embedder = await asyncio.to_thread(Embedder)

    embedder = state.embedder
    if embedder is None:
        raise RuntimeError("Embedding model failed to initialize.")

    if state.backfill_task is None:
        state.backfill_task = asyncio.create_task(
            _backfill_after_load(state, embedder)
        )

    return embedder


async def _backfill_after_load(
    state: ReadinessState, embedder: Embedder,
) -> None:
    """Backfill missing vectors without delaying the semantic query that
    triggered model loading."""
    try:
        async with state.maintenance_lock:
            stats = await asyncio.to_thread(_run_backfill_fresh_conn, embedder)
        log.info("Background embedding backfill complete: %s", stats)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Background embedding backfill failed; periodic scan will retry")
    finally:
        state.backfill_done.set()


def get_embedder_if_ready() -> Optional[Embedder]:
    """Non-blocking snapshot. Returns the embedder once the model is loaded.
    Used by save_memory so opportunistic embedding never triggers a cold
    load — the backfill catches up any rows saved before the model was up."""
    if _state is None:
        return None
    return _state.embedder
