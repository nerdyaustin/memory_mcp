"""SQLAlchemy models for the memory_mcp sync server."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Users & machines
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    api_key_hash = Column(String(255), nullable=False, unique=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    machines = relationship("Machine", back_populates="user")
    memories = relationship("Memory", back_populates="user")


class Machine(Base):
    __tablename__ = "machines"

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False,
    )
    hostname = Column(String(255), nullable=False)
    registered_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_sync_at = Column(DateTime(timezone=True))

    user = relationship("User", back_populates="machines")
    sessions = relationship("Session", back_populates="machine")
    memories = relationship("Memory", back_populates="machine")


# ---------------------------------------------------------------------------
# Sessions & messages
# ---------------------------------------------------------------------------


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(255), primary_key=True)
    machine_id = Column(
        UUID(as_uuid=True), ForeignKey("machines.id"), primary_key=True,
    )
    source = Column(String(50), nullable=False)
    title = Column(Text)
    cwd = Column(Text)
    model = Column(String(255))
    started_at = Column(String(50))
    message_count = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    file_path = Column(Text)
    file_mtime = Column(Float, default=0.0)
    server_updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    machine = relationship("Machine", back_populates="sessions")
    messages = relationship(
        "Message", back_populates="session",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_sessions_source", "source"),
        Index("ix_sessions_started", "started_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(255), primary_key=True)
    session_id = Column(String(255), primary_key=True)
    machine_id = Column(
        UUID(as_uuid=True), ForeignKey("machines.id"), primary_key=True,
    )
    parent_id = Column(String(255))
    role = Column(String(50), nullable=False)
    content = Column(Text)
    thinking = Column(Text)
    tool_name = Column(String(255))
    tool_input = Column(Text)
    tool_output = Column(Text)
    timestamp = Column(String(50))
    model = Column(String(255))
    cost_usd = Column(Float)

    # Full-text search vector (populated via trigger or application code).
    search_vector = Column(
        "search_vector", Text,
    )

    # Vector embedding for semantic search (384-dim, matching BAAI/bge-small-en-v1.5).
    embedding = Column(Vector(384))

    session = relationship("Session", back_populates="messages")

    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "machine_id"],
            ["sessions.id", "sessions.machine_id"],
            ondelete="CASCADE",
        ),
        Index("ix_messages_session", "session_id"),
    )


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


class Memory(Base):
    __tablename__ = "memories"

    global_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    machine_id = Column(
        UUID(as_uuid=True), ForeignKey("machines.id"), nullable=False,
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False,
    )
    content = Column(Text, nullable=False)
    tags = Column(Text)  # JSON array of strings
    context = Column(Text)
    source_session_id = Column(String(255))
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    server_updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Vector embedding for semantic search.
    embedding = Column(Vector(384))

    machine = relationship("Machine", back_populates="memories")
    user = relationship("User", back_populates="memories")

    __table_args__ = ()


# ---------------------------------------------------------------------------
# Sync state (per-machine pull cursors)
# ---------------------------------------------------------------------------


class SyncCursor(Base):
    __tablename__ = "sync_cursors"

    machine_id = Column(
        UUID(as_uuid=True), ForeignKey("machines.id"), primary_key=True,
    )
    last_pull_at = Column(DateTime(timezone=True))
