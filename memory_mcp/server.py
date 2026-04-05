from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from memory_mcp.db import init_db
from memory_mcp.scanner import scan_sessions
from memory_mcp.tools.memory import register_memory_tools
from memory_mcp.tools.sessions import register_session_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    logger = logging.getLogger("memory_mcp")
    logger.info("Initializing database...")
    db = init_db()
    logger.info("Running initial session scan...")
    try:
        stats = scan_sessions(db)
        logger.info("Scan complete: %s", stats)
    except Exception:
        logger.exception("Session scan failed — starting without session data")
    try:
        yield {"db": db}
    finally:
        db.close()
        logger.info("Database connection closed.")


mcp = FastMCP("MemoryMCP", lifespan=lifespan)
register_memory_tools(mcp)
register_session_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
