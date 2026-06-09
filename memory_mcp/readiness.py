"""Semantic-readiness coordination for v0.3.0 lazy-startup.

Holds a lifespan-owned ReadinessState carrying the lazy Embedder, an
initial-scan gate, a backfill gate, and a maintenance lock shared between
the periodic scan and the first-semantic-call backfill. State is constructed
inside ``lifespan`` (not at module import) so asyncio primitives bind to the
running event loop. Tools reach the state via module-level helpers after
``init_readiness`` has been called once during startup.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from memory_mcp import db as db_mod
from memory_mcp.db import backfill_embeddings, init_db
from memory_mcp.embeddings import AVAILABLE as EMBED_AVAILABLE, Embedder

log = logging.getLogger(__name__)


@dataclass
class ReadinessState:
    scan_done: asyncio.Event
    backfill_done: asyncio.Event
    maintenance_lock: asyncio.Lock
    embedder: Optional[Embedder] = None

    @classmethod
    def new(cls) -> "ReadinessState":
        """Construct inside a running loop so asyncio primitives bind correctly."""
        return cls(
            scan_done=asyncio.Event(),
            backfill_done=asyncio.Event(),
            maintenance_lock=asyncio.Lock(),
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
    background thread opens its own via init_db() rather than sharing the
    lifespan connection."""
    conn = init_db()
    try:
        return backfill_embeddings(conn, embedder)
    finally:
        conn.close()


async def ensure_semantic_ready() -> Embedder:
    """Return an Embedder once the model is loaded and the initial vector
    backfill is complete. Fast path on subsequent calls. The first caller
    pays the cold-load cost; concurrent callers serialise on the lock and
    share the result.
    """
    state = get_state()

    if state.embedder is not None and state.backfill_done.is_set():
        return state.embedder

    # Backfill against a half-populated DB would leave newly-scanned rows
    # unembedded. Wait for the initial scan to complete first.
    await state.scan_done.wait()

    async with state.maintenance_lock:
        if state.embedder is None:
            if not EMBED_AVAILABLE or not db_mod.VEC_AVAILABLE:
                raise RuntimeError(
                    "Semantic search unavailable: fastembed or sqlite-vec missing."
                )
            state.embedder = await asyncio.to_thread(Embedder)

        if not state.backfill_done.is_set():
            try:
                stats = await asyncio.to_thread(
                    _run_backfill_fresh_conn, state.embedder
                )
                log.info("Initial backfill complete: %s", stats)
            except Exception:
                log.exception("Initial backfill failed; periodic scan will retry")
            finally:
                # Mark done regardless of success — periodic scan handles
                # retry; leaving the gate closed would block every future
                # semantic call on a persistently-failing backfill.
                state.backfill_done.set()

    return state.embedder


def get_embedder_if_ready() -> Optional[Embedder]:
    """Non-blocking snapshot. Returns the embedder only when fully ready.
    Used by save_memory so opportunistic embedding never triggers a cold
    load — periodic backfill will catch up unembedded rows later."""
    if _state is None:
        return None
    if _state.embedder is not None and _state.backfill_done.is_set():
        return _state.embedder
    return None
