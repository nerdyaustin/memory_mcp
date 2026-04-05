"""Parser for OMP (Oh My Pi) session JSONL files."""

from __future__ import annotations

import json
import logging
import os

from .base import ParsedMessage, ParsedSession

log = logging.getLogger(__name__)


def _extract_text(content: list[dict] | str | None) -> str | None:
    """Join all text blocks from a content array. Handles string content too."""
    if content is None:
        return None
    if isinstance(content, str):
        return content or None
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(parts) or None


class OmpParser:
    """Parses OMP agent session JSONL files into ParsedSession objects."""

    source_type = "omp"

    def parse_file(self, path: str) -> ParsedSession | None:
        """Parse a single OMP JSONL file into a ParsedSession."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            log.warning("Cannot stat file: %s", path)
            return None

        session_id: str | None = None
        title: str | None = None
        cwd: str | None = None
        started_at: str | None = None
        current_model: str | None = None
        messages: list[ParsedMessage] = []
        total_cost = 0.0

        try:
            with open(path, encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning("%s:%d: malformed JSON, skipping", path, line_no)
                        continue

                    if not isinstance(entry, dict):
                        continue

                    entry_type = entry.get("type")

                    if entry_type == "session":
                        session_id = entry.get("id")
                        title = entry.get("title")
                        cwd = entry.get("cwd")
                        started_at = entry.get("timestamp")

                    elif entry_type == "model_change":
                        current_model = entry.get("model")

                    elif entry_type == "message":
                        msg = self._parse_message(
                            entry, session_id or "", current_model,
                        )
                        if msg is not None:
                            if msg.cost_usd:
                                total_cost += msg.cost_usd
                            messages.append(msg)

        except OSError:
            log.warning("Cannot read file: %s", path)
            return None

        # Fallback session id from filename if no session header found.
        if session_id is None:
            session_id = os.path.splitext(os.path.basename(path))[0]

        return ParsedSession(
            id=session_id,
            source=self.source_type,
            file_path=path,
            file_mtime=mtime,
            title=title,
            cwd=cwd,
            model=current_model,
            started_at=started_at,
            total_cost_usd=total_cost,
            messages=messages,
        )

    @staticmethod
    def _parse_message(
        entry: dict, session_id: str, current_model: str | None,
    ) -> ParsedMessage | None:
        """Convert a single JSONL message entry to a ParsedMessage."""
        msg = entry.get("message")
        if not isinstance(msg, dict):
            return None

        role = msg.get("role")
        if role is None:
            return None

        entry_id = entry.get("id")
        if entry_id is None:
            return None

        parent_id = entry.get("parentId")
        timestamp = entry.get("timestamp")

        content: str | None = None
        thinking: str | None = None
        tool_name: str | None = None
        tool_input: str | None = None
        tool_output: str | None = None
        model: str | None = None
        cost_usd: float | None = None
        parsed_role = role

        if role == "user":
            content = _extract_text(msg.get("content"))

        elif role == "assistant":
            blocks = msg.get("content")
            if isinstance(blocks, list):
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text = block.get("text", "")
                        if text:
                            text_parts.append(text)
                    elif btype == "thinking":
                        t = block.get("thinking", "")
                        if t:
                            thinking_parts.append(t)
                    elif btype == "toolCall":
                        tool_name = block.get("name")
                        args = block.get("arguments")
                        if args is not None:
                            try:
                                tool_input = json.dumps(args)
                            except (TypeError, ValueError):
                                tool_input = str(args)

                content = "\n".join(text_parts) or None
                thinking = "\n".join(thinking_parts) or None

            model = msg.get("model") or current_model
            # Cost extraction: message.usage.cost.total
            usage = msg.get("usage")
            if isinstance(usage, dict):
                cost_obj = usage.get("cost")
                if isinstance(cost_obj, dict):
                    raw_cost = cost_obj.get("total")
                    if raw_cost is not None:
                        try:
                            cost_usd = float(raw_cost)
                        except (TypeError, ValueError):
                            pass

        elif role == "toolResult":
            parsed_role = "tool_result"
            tool_name = msg.get("toolName")
            tool_output = _extract_text(msg.get("content"))

        else:
            # Unknown role — still store it.
            content = _extract_text(msg.get("content"))

        return ParsedMessage(
            id=str(entry_id),
            session_id=session_id,
            role=parsed_role,
            timestamp=timestamp,
            parent_id=parent_id,
            content=content,
            thinking=thinking,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            model=model if role == "assistant" else current_model,
            cost_usd=cost_usd,
        )
