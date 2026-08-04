"""Hosted sync server for memory_mcp.

Aggregates session and memory data from multiple machines behind a
REST API.  Uses PostgreSQL + pgvector for storage and vector search.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from hosted.auth import get_api_key, verify_api_key
from hosted.models import Base, Machine, Memory, Message, Session, SyncCursor

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/memory_mcp",
)

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    """Yield an async database session."""
    async with async_session() as session:
        yield session


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables ready")
    yield
    await engine.dispose()


app = FastAPI(title="memory_mcp Sync Server", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health(db: Annotated[AsyncSession, Depends(get_db)]) -> dict:
    """Health check — verifies database connectivity."""
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        return {"status": "degraded"}


# ---------------------------------------------------------------------------
# Machine registration
# ---------------------------------------------------------------------------


@app.post("/machines/register")
async def register_machine(
    request: Request,
    payload: dict,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Register (or re-register) a machine."""
    token_hash = await get_api_key(request, None)  # type: ignore[arg-type]
    user_id = await verify_api_key(db, token_hash)

    machine_id = payload.get("machine_id")
    hostname = payload.get("hostname", "unknown")

    if not machine_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="machine_id is required",
        )

    try:
        mid_uuid = uuid.UUID(machine_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="machine_id must be a valid UUID",
        )

    stmt = pg_insert(Machine).values(
        id=mid_uuid,
        user_id=uuid.UUID(user_id),
        hostname=hostname,
        registered_at=datetime.now(timezone.utc),
    ).on_conflict_do_update(
        index_elements=["id"],
        set_={"hostname": hostname, "last_sync_at": datetime.now(timezone.utc)},
    )
    await db.execute(stmt)
    await db.commit()

    return {"registered": True, "machine_id": machine_id}


# ---------------------------------------------------------------------------
# Sync push
# ---------------------------------------------------------------------------


@app.post("/sync/push")
async def sync_push(
    request: Request,
    payload: dict,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Accept a batch of sessions and memories from a machine."""
    token_hash = await get_api_key(request, None)  # type: ignore[arg-type]
    user_id = await verify_api_key(db, token_hash)

    machine_id = payload.get("machine_id", "")
    sessions_data = payload.get("sessions", [])
    memories_data = payload.get("memories", [])

    try:
        mid_uuid = uuid.UUID(machine_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="machine_id must be a valid UUID",
        )

    now = datetime.now(timezone.utc)
    sessions_accepted: list[str] = []
    memories_accepted: list[str] = []

    for sess in sessions_data:
        sid = sess.get("id", "")
        if not sid:
            continue

        # Upsert session header.
        stmt = pg_insert(Session).values(
            id=_pg_text(sid) or "",
            machine_id=mid_uuid,
            source=_pg_text(sess.get("source") or ""),
            title=_pg_text(sess.get("title")),
            cwd=_pg_text(sess.get("cwd")),
            model=_pg_text(sess.get("model")),
            started_at=_pg_text(sess.get("started_at")),
            message_count=sess.get("message_count", 0),
            total_cost_usd=sess.get("total_cost_usd", 0.0),
            file_path=_pg_text(sess.get("file_path") or ""),
            file_mtime=sess.get("file_mtime", 0.0),
            server_updated_at=now,
        ).on_conflict_do_update(
            index_elements=["id", "machine_id"],
            set_={
                "title": _pg_text(sess.get("title")),
                "message_count": sess.get("message_count", 0),
                "total_cost_usd": sess.get("total_cost_usd", 0.0),
                "server_updated_at": now,
            },
        )
        await db.execute(stmt)

        # Upsert messages.
        for msg in sess.get("messages", []):
            msg_stmt = pg_insert(Message).values(
                id=_pg_text(msg.get("id") or ""),
                session_id=_pg_text(sid) or "",
                machine_id=mid_uuid,
                parent_id=_pg_text(msg.get("parent_id")),
                role=_pg_text(msg.get("role") or ""),
                content=_pg_text(msg.get("content")),
                thinking=_pg_text(msg.get("thinking")),
                tool_name=_pg_text(msg.get("tool_name")),
                tool_input=_pg_text(msg.get("tool_input")),
                tool_output=_pg_text(msg.get("tool_output")),
                timestamp=_pg_text(msg.get("timestamp")),
                model=_pg_text(msg.get("model")),
                cost_usd=msg.get("cost_usd"),
            ).on_conflict_do_nothing()
            await db.execute(msg_stmt)

        sessions_accepted.append(sid)

    for mem in memories_data:
        gid = mem.get("global_id", "")
        if not gid:
            continue
        try:
            gid_uuid = uuid.UUID(gid)
        except ValueError:
            continue

        stmt = pg_insert(Memory).values(
            global_id=gid_uuid,
            machine_id=mid_uuid,
            user_id=uuid.UUID(user_id),
            content=_pg_text(mem.get("content") or ""),
            tags=_pg_text(mem.get("tags")),
            context=_pg_text(mem.get("context")),
            source_session_id=_pg_text(mem.get("source_session_id")),
            created_at=_parse_dt(mem.get("created_at")),
            updated_at=_parse_dt(mem.get("updated_at")),
            server_updated_at=now,
        ).on_conflict_do_update(
            index_elements=["global_id"],
            set_={
                "content": _pg_text(mem.get("content") or ""),
                "tags": _pg_text(mem.get("tags")),
                "context": _pg_text(mem.get("context")),
                "updated_at": _parse_dt(mem.get("updated_at")),
                "server_updated_at": now,
            },
        )
        await db.execute(stmt)
        memories_accepted.append(gid)

    # Update machine's last_sync_at.
    await db.execute(
        pg_insert(SyncCursor).values(
            machine_id=mid_uuid, last_pull_at=now,
        ).on_conflict_do_update(
            index_elements=["machine_id"],
            set_={"last_pull_at": now},
        ),
    )

    await db.commit()

    return {
        "server_ts": now.isoformat(),
        "sessions_accepted": sessions_accepted,
        "memories_accepted": memories_accepted,
    }


# ---------------------------------------------------------------------------
# Sync pull
# ---------------------------------------------------------------------------


@app.get("/sync/pull")
async def sync_pull(
    request: Request,
    machine_id: str,
    since: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return sessions and memories from *other* machines updated after *since*."""
    token_hash = await get_api_key(request, None)  # type: ignore[arg-type]
    user_id = await verify_api_key(db, token_hash)

    try:
        mid_uuid = uuid.UUID(machine_id)
        uid_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid UUID",
        )

    now = datetime.now(timezone.utc)
    since_dt = _parse_dt(since) if since else None

    # Pull sessions from other machines belonging to this user.
    sess_query = select(Session).where(
        Session.machine_id != mid_uuid,
    )
    if since_dt:
        sess_query = sess_query.where(Session.server_updated_at > since_dt)

    result = await db.execute(sess_query)
    sessions = result.scalars().all()

    sessions_data: list[dict] = []
    for sess in sessions:
        sess_dict = {
            "id": sess.id,
            "machine_id": str(sess.machine_id),
            "source": sess.source,
            "title": sess.title,
            "cwd": sess.cwd,
            "model": sess.model,
            "started_at": sess.started_at,
            "message_count": sess.message_count,
            "total_cost_usd": sess.total_cost_usd,
            "file_path": sess.file_path,
            "file_mtime": sess.file_mtime,
        }

        # Fetch messages for this session.
        msg_result = await db.execute(
            select(Message).where(
                Message.session_id == sess.id,
                Message.machine_id == sess.machine_id,
            ).order_by(Message.id),
        )
        msgs = msg_result.scalars().all()
        sess_dict["messages"] = [
            {
                "id": m.id,
                "session_id": m.session_id,
                "parent_id": m.parent_id,
                "role": m.role,
                "content": m.content,
                "thinking": m.thinking,
                "tool_name": m.tool_name,
                "tool_input": m.tool_input,
                "tool_output": m.tool_output,
                "timestamp": m.timestamp,
                "model": m.model,
                "cost_usd": m.cost_usd,
                "machine_id": str(m.machine_id),
            }
            for m in msgs
        ]
        sessions_data.append(sess_dict)

    # Pull memories from other machines.
    mem_query = select(Memory).where(
        Memory.machine_id != mid_uuid,
        Memory.user_id == uid_uuid,
    )
    if since_dt:
        mem_query = mem_query.where(Memory.server_updated_at > since_dt)

    result = await db.execute(mem_query)
    memories = result.scalars().all()

    memories_data = [
        {
            "global_id": str(m.global_id),
            "machine_id": str(m.machine_id),
            "content": m.content,
            "tags": m.tags,
            "context": m.context,
            "source_session_id": m.source_session_id,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "updated_at": m.updated_at.isoformat() if m.updated_at else None,
        }
        for m in memories
    ]

    return {
        "server_ts": now.isoformat(),
        "sessions": sessions_data,
        "memories": memories_data,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pg_text(value) -> str | None:
    """Return a PostgreSQL-safe text value.

    PostgreSQL text/varchar cannot store NUL bytes.  Some captured tool output
    can contain binary blobs, so normalize them at the API boundary instead of
    failing the whole sync batch.
    """
    if value is None:
        return None
    return str(value).replace("\x00", "\uFFFD")


def _parse_dt(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp, returning None on failure."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("hosted.server:app", host=host, port=port, reload=False)
