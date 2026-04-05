"""Shared types for session parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ParsedMessage:
    """A single conversational turn extracted from a session file."""

    id: str
    session_id: str
    role: str  # 'user' | 'assistant' | 'tool_use' | 'tool_result'
    timestamp: str | None = None
    parent_id: str | None = None
    content: str | None = None
    thinking: str | None = None
    tool_name: str | None = None
    tool_input: str | None = None  # JSON string
    tool_output: str | None = None
    model: str | None = None
    cost_usd: float | None = None


@dataclass
class ParsedSession:
    """A complete session extracted from one JSONL file."""

    id: str
    source: str  # 'claude_code' | 'omp'
    file_path: str
    file_mtime: float
    title: str | None = None
    cwd: str | None = None
    model: str | None = None
    started_at: str | None = None
    total_cost_usd: float = 0.0
    messages: list[ParsedMessage] = field(default_factory=list)


class SessionParser(Protocol):
    """Interface every source-specific parser must satisfy."""

    source_type: str

    def parse_file(self, path: str) -> ParsedSession | None:
        """Parse a single JSONL file into a session. Returns None if unparseable."""
        ...
