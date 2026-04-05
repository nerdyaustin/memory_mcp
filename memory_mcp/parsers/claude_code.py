"""Parser for Claude Code session JSONL files."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

from .base import ParsedMessage, ParsedSession

logger = logging.getLogger(__name__)


class ClaudeCodeParser:
    """Parses Claude Code JSONL conversation files into ParsedSession objects."""

    source_type = "claude_code"

    def parse_file(self, path: str) -> ParsedSession | None:
        """Parse a Claude Code JSONL file. Returns None for summary-only or empty files."""
        try:
            entries = _read_entries(path)
        except OSError:
            logger.warning("Cannot read file: %s", path)
            return None

        if not entries:
            return None

        # Determine if summary-only: first non-snapshot entry is type "summary" → skip.
        first_meaningful = next(
            (e for e in entries if e.get("type") != "file-history-snapshot"),
            None,
        )
        if first_meaningful is None:
            return None
        if first_meaningful.get("type") == "summary":
            return None

        # Extract session metadata from first user/assistant entry.
        session_id: str | None = None
        cwd: str | None = None
        for e in entries:
            if e.get("type") in ("user", "assistant"):
                session_id = session_id or e.get("sessionId")
                cwd = cwd or e.get("cwd")
                if session_id and cwd:
                    break

        if session_id is None:
            # Fallback: derive from filename.
            session_id = os.path.splitext(os.path.basename(path))[0]

        messages: list[ParsedMessage] = []
        title: str | None = None

        # --- Process assistant entries: group by message.id, merge content blocks ---
        assistant_groups: dict[str, list[dict]] = defaultdict(list)
        assistant_order: list[str] = []  # preserve first-seen order

        for e in entries:
            etype = e.get("type")
            if etype == "assistant":
                msg = e.get("message")
                if not msg or "id" not in msg:
                    continue
                mid = msg["id"]
                if mid not in assistant_groups:
                    assistant_order.append(mid)
                assistant_groups[mid].append(e)
            elif etype == "user":
                msgs = _parse_user_entry(e, session_id)
                for m in msgs:
                    if title is None and m.role == "user" and m.content:
                        title = m.content[:80]
                messages.extend(msgs)

        for mid in assistant_order:
            group = assistant_groups[mid]
            m = _merge_assistant_group(group, session_id)
            if m is not None:
                messages.append(m)

        if not messages:
            return None

        # Sort by timestamp (None-timestamps sort last).
        messages.sort(key=lambda m: m.timestamp or "")

        file_mtime = os.path.getmtime(path)
        started_at = next((m.timestamp for m in messages if m.timestamp), None)

        return ParsedSession(
            id=session_id,
            source=self.source_type,
            file_path=path,
            file_mtime=file_mtime,
            title=title,
            cwd=cwd,
            model=next(
                (m.model for m in messages if m.role == "assistant" and m.model),
                None,
            ),
            started_at=started_at,
            total_cost_usd=0.0,
            messages=messages,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_entries(path: str) -> list[dict]:
    """Read all valid JSON entries from a JSONL file."""
    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s:%d: malformed JSON, skipping", path, lineno)
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries


def _extract_text_from_content(content) -> str | None:
    """Extract joined text from a content value (string or list of blocks)."""
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
        return "\n\n".join(parts) if parts else None
    return None


def _parse_user_entry(entry: dict, session_id: str) -> list[ParsedMessage]:
    """Parse a user-type entry, returning one or more ParsedMessages.

    A single user entry may contain both regular text and tool_result blocks;
    those are split into separate messages.
    """
    msg = entry.get("message")
    if not msg:
        return []

    uuid = entry.get("uuid", "")
    parent_id = entry.get("parentUuid")
    timestamp = entry.get("timestamp")
    content_raw = msg.get("content")
    results: list[ParsedMessage] = []

    if isinstance(content_raw, str):
        # Plain string content — always a user message.
        text = content_raw.strip() if content_raw else None
        results.append(ParsedMessage(
            id=uuid,
            session_id=session_id,
            role="user",
            timestamp=timestamp,
            parent_id=parent_id,
            content=text,
        ))
        return results

    if not isinstance(content_raw, list):
        return []

    # Separate text blocks from tool_result blocks.
    text_parts: list[str] = []
    tool_results: list[dict] = []

    for block in content_raw:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "tool_result":
            tool_results.append(block)
        elif btype == "text":
            t = block.get("text", "")
            if isinstance(t, str) and t.strip():
                text_parts.append(t.strip())

    if text_parts:
        results.append(ParsedMessage(
            id=uuid,
            session_id=session_id,
            role="user",
            timestamp=timestamp,
            parent_id=parent_id,
            content="\n\n".join(text_parts),
        ))

    for tr in tool_results:
        tr_content = tr.get("content", "")
        if isinstance(tr_content, list):
            # Content can be a list of blocks too.
            parts = []
            for b in tr_content:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
            tr_text = "\n".join(parts)
        elif isinstance(tr_content, str):
            tr_text = tr_content
        else:
            tr_text = str(tr_content)

        results.append(ParsedMessage(
            id=f"{uuid}_tr_{tr.get('tool_use_id', '')}",
            session_id=session_id,
            role="tool_result",
            timestamp=timestamp,
            parent_id=parent_id,
            tool_output=tr_text if tr_text else None,
        ))

    # If nothing was produced (all blocks were unrecognised), still emit an
    # empty user message so the session isn't silently truncated.
    if not results:
        results.append(ParsedMessage(
            id=uuid,
            session_id=session_id,
            role="user",
            timestamp=timestamp,
            parent_id=parent_id,
        ))

    return results


def _merge_assistant_group(group: list[dict], session_id: str) -> ParsedMessage | None:
    """Merge multiple JSONL entries that share the same assistant message.id."""
    if not group:
        return None

    first = group[0]
    uuid = first.get("uuid", "")
    parent_id = first.get("parentUuid")
    timestamp = first.get("timestamp")
    first_msg = first.get("message", {})
    model = first_msg.get("model")

    # Merge content blocks from all entries in order.
    all_blocks: list[dict] = []
    for entry in group:
        msg = entry.get("message")
        if not msg:
            continue
        blocks = msg.get("content")
        if isinstance(blocks, list):
            all_blocks.extend(blocks)

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_name: str | None = None
    tool_input: str | None = None

    for block in all_blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text", "")
            if isinstance(t, str) and t.strip():
                text_parts.append(t.strip())
        elif btype == "thinking":
            t = block.get("thinking", "")
            if isinstance(t, str) and t.strip():
                thinking_parts.append(t.strip())
        elif btype == "tool_use":
            if tool_name is None:
                tool_name = block.get("name")
                inp = block.get("input")
                if inp is not None:
                    tool_input = json.dumps(inp)

    return ParsedMessage(
        id=uuid,
        session_id=session_id,
        role="assistant",
        timestamp=timestamp,
        parent_id=parent_id,
        content="\n\n".join(text_parts) if text_parts else None,
        thinking="\n\n".join(thinking_parts) if thinking_parts else None,
        tool_name=tool_name,
        tool_input=tool_input,
        model=model,
        cost_usd=None,
    )
