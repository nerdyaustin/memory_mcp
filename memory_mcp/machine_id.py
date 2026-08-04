"""Persistent machine identity for sync-enabled deployments.

Each host generates a stable UUID on first run and stores it in
``~/.memory_mcp/machine_id``.  This UUID is the sync key for
identifying which machine authored a session or memory.
"""

from __future__ import annotations

import uuid
from pathlib import Path


def get_machine_id() -> str:
    """Return the persistent machine UUID, generating it if absent.

    The file lives in the same directory as the main database so it
    survives pip reinstalls and moves with any custom ``MEMORY_MCP_DB``
    path.
    """
    db_path = _db_dir()
    id_file = db_path / "machine_id"
    try:
        return id_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass

    mid = str(uuid.uuid4())
    db_path.mkdir(parents=True, exist_ok=True)
    id_file.write_text(mid, encoding="utf-8")
    return mid


def _db_dir() -> Path:
    """Return the directory that holds memory.db and machine_id."""
    import os

    custom = os.environ.get("MEMORY_MCP_DB")
    if custom:
        return Path(custom).parent
    return Path.home() / ".memory_mcp"
