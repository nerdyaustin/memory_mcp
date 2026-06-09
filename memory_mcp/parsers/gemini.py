"""Parser for Google Gemini CLI chat session JSON files.

Gemini writes each session to ~/.gemini/tmp/<project>/chats/session-<iso>-<uuid>.json.
The file is a single JSON object, not JSONL:

    {
      "sessionId": "...",
      "projectHash": "...",
      "startTime": "...",
      "lastUpdated": "...",
      "messages": [
        {"id": "...", "timestamp": "...", "type": "user",
         "content": [{"text": "..."}]},
        {"id": "...", "timestamp": "...", "type": "gemini",
         "content": "final answer text",
         "thoughts": [{"subject": "...", "description": "...", "timestamp": "..."}],
         "tokens": {...}, "model": "gemini-3-flash-preview"}
      ],
      "kind": "main"
    }

User messages carry content as a list of {text} blocks; assistant ("gemini")
messages carry content as a plain string plus optional thoughts which we
collapse into the ``thinking`` field. Sibling logs.json files (which live in
the same tree but are not session transcripts) are rejected via the
``session-`` filename prefix.
"""

from __future__ import annotations

import json
import logging
import os

from .base import ParsedMessage, ParsedSession

log = logging.getLogger(__name__)


class GeminiParser:
    """Parses Gemini CLI session-*.json files into ParsedSession objects."""

    source_type = "gemini"
    file_extensions = (".json",)

    def parse_file(self, path: str) -> ParsedSession | None:
        basename = os.path.basename(path)
        if not basename.startswith("session-"):
            return None

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
            return None

        session_id = data.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            session_id = os.path.splitext(basename)[0]

        started_at = data.get("startTime") if isinstance(data.get("startTime"), str) else None

        messages: list[ParsedMessage] = []
        title: str | None = None
        session_model: str | None = None

        for idx, m in enumerate(messages_raw):
            if not isinstance(m, dict):
                continue

            mtype = m.get("type")
            if mtype == "user":
                role = "user"
                content = _extract_user_content(m.get("content"))
                thinking = None
                model = None
                if content and title is None:
                    title = content[:80]
            elif mtype == "gemini":
                role = "assistant"
                raw = m.get("content")
                content = raw.strip() if isinstance(raw, str) and raw.strip() else None
                thinking = _extract_thoughts(m.get("thoughts"))
                model = m.get("model") if isinstance(m.get("model"), str) else None
                if session_model is None and model:
                    session_model = model
            else:
                continue

            if not content and not thinking:
                continue

            msg_id = m.get("id") if isinstance(m.get("id"), str) else f"{session_id}-{idx}"
            timestamp = m.get("timestamp") if isinstance(m.get("timestamp"), str) else None

            messages.append(ParsedMessage(
                id=msg_id,
                session_id=session_id,
                role=role,
                timestamp=timestamp,
                content=content,
                thinking=thinking,
                model=model,
            ))

        if not messages:
            return None

        return ParsedSession(
            id=session_id,
            source=self.source_type,
            file_path=path,
            file_mtime=mtime,
            title=title,
            model=session_model,
            started_at=started_at,
            messages=messages,
        )


def _extract_user_content(content_raw) -> str | None:
    if isinstance(content_raw, str):
        return content_raw.strip() or None
    if not isinstance(content_raw, list):
        return None
    parts: list[str] = []
    for block in content_raw:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n\n".join(parts) if parts else None


def _extract_thoughts(thoughts_raw) -> str | None:
    if not isinstance(thoughts_raw, list):
        return None
    parts: list[str] = []
    for thought in thoughts_raw:
        if not isinstance(thought, dict):
            continue
        subject = thought.get("subject") or ""
        description = thought.get("description") or ""
        if not isinstance(subject, str):
            subject = ""
        if not isinstance(description, str):
            description = ""
        subject = subject.strip()
        description = description.strip()
        if subject and description:
            parts.append(f"{subject}: {description}")
        elif subject or description:
            parts.append(subject or description)
    return "\n\n".join(parts) if parts else None
