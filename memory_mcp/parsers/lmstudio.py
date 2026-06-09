"""Parser for LM Studio conversation JSON files.

LM Studio stores each conversation as a single JSON file.  Messages use a
versioned structure (to support response regeneration):

    messages: [
      {
        "versions": [
          {
            "type": "singleStep",          # user messages
            "role": "user",
            "content": [ { "type": "text", "text": "..." } ]
          }
        ],
        "currentlySelected": 0
      },
      {
        "versions": [
          {
            "type": "multiStep",           # assistant messages
            "role": "assistant",
            "senderInfo": { "senderName": "model-id" },
            "steps": [
              { "type": "contentBlock", "content": [ { "type": "text", "text": "..." } ] },
              { "type": "debugInfoBlock", ... }   # skip these
            ]
          }
        ],
        "currentlySelected": 0
      }
    ]

Files live under ~/.lmstudio/conversations/, possibly in model-named
subdirectories (e.g. Gemma4/).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from .base import ParsedMessage, ParsedSession

log = logging.getLogger(__name__)


def _join_text_blocks(content_array: list) -> str | None:
    """Extract and join text from an array of {type, text} content blocks."""
    parts: list[str] = []
    for block in content_array:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts) or None


def _extract_single_step(version: dict) -> str | None:
    """Extract text from a singleStep (user) message version."""
    content = version.get("content")
    if not isinstance(content, list):
        return None
    return _join_text_blocks(content)


def _extract_multi_step(version: dict) -> str | None:
    """Extract text from a multiStep (assistant) message version.

    Iterates steps, skips debugInfoBlock entries, joins contentBlock text.
    """
    steps = version.get("steps")
    if not isinstance(steps, list):
        return None
    parts: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("type") != "contentBlock":
            continue
        content = step.get("content")
        if isinstance(content, list):
            text = _join_text_blocks(content)
            if text:
                parts.append(text)
    return "\n".join(parts) or None


class LmStudioParser:
    """Parses LM Studio .conversation.json files into ParsedSession objects."""

    source_type = "lmstudio"
    file_extensions = (".conversation.json",)

    def parse_file(self, path: str) -> ParsedSession | None:
        """Parse a single LM Studio conversation file."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            log.warning("Cannot stat file: %s", path)
            return None

        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Cannot read/parse %s: %s", path, exc)
            return None

        if not isinstance(data, dict):
            return None

        messages_raw = data.get("messages")
        if not isinstance(messages_raw, list) or not messages_raw:
            return None  # Skip empty conversations.

        # Session ID from filename: "1775919649569.conversation.json" -> "1775919649569"
        basename = os.path.basename(path)
        session_id = basename
        for ext in self.file_extensions:
            if session_id.endswith(ext):
                session_id = session_id[: -len(ext)]
                break

        # Timestamp from createdAt (milliseconds since epoch).
        started_at: str | None = None
        created_at = data.get("createdAt")
        if isinstance(created_at, (int, float)) and created_at > 0:
            try:
                dt = datetime.fromtimestamp(created_at / 1000, tz=timezone.utc)
                started_at = dt.isoformat()
            except (OSError, OverflowError, ValueError):
                pass

        # Session-level model fallback.
        session_model: str | None = None
        last_used = data.get("lastUsedModel")
        if isinstance(last_used, dict):
            session_model = last_used.get("identifier")

        messages: list[ParsedMessage] = []
        for idx, msg_entry in enumerate(messages_raw):
            if not isinstance(msg_entry, dict):
                continue

            # Pick the active version.
            versions = msg_entry.get("versions")
            if not isinstance(versions, list) or not versions:
                continue
            selected = msg_entry.get("currentlySelected", 0)
            if not isinstance(selected, int) or selected < 0 or selected >= len(versions):
                selected = 0
            version = versions[selected]
            if not isinstance(version, dict):
                continue

            role = version.get("role")
            if not isinstance(role, str) or role not in ("user", "assistant", "system"):
                continue

            # Extract text based on message type.
            step_type = version.get("type", "")
            if step_type == "multiStep":
                content = _extract_multi_step(version)
            else:
                # singleStep and any future type with a content array.
                content = _extract_single_step(version)

            if not content:
                continue

            # Model from assistant's senderInfo, falling back to session-level.
            model: str | None = None
            if role == "assistant":
                sender = version.get("senderInfo")
                if isinstance(sender, dict):
                    model = sender.get("senderName")
            if not model:
                model = session_model

            messages.append(ParsedMessage(
                id=f"{session_id}-{idx}",
                session_id=session_id,
                role=role,
                content=content,
                model=model,
            ))

        if not messages:
            return None

        return ParsedSession(
            id=session_id,
            source=self.source_type,
            file_path=path,
            file_mtime=mtime,
            title=data.get("name") if isinstance(data.get("name"), str) else None,
            model=session_model,
            started_at=started_at,
            messages=messages,
        )
