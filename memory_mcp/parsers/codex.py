"""Parser for OpenAI Codex CLI rollout JSONL files.

Codex writes each session to ~/.codex/sessions/YYYY/MM/DD/rollout-<iso>-<uuid>.jsonl.
One JSON event per line. Event types we care about:

    session_meta   - first line, carries id / cwd / startTime / cli_version
    turn_context   - carries the active model (and cwd) for subsequent events
    event_msg      - wraps user_message / agent_message / task_started / etc.
    response_item  - low-level messages, reasoning traces (encrypted), ignored
                     because event_msg already has clean summaries.

For each user turn we emit one user ParsedMessage. For each assistant response
we merge all agent_message events between user turns into a single assistant
ParsedMessage, splitting by phase: "commentary" goes into ``thinking`` and
"final_answer" goes into ``content`` (mirroring the Claude Code parser's
text/thinking split).
"""

from __future__ import annotations

import json
import logging
import os

from .base import ParsedMessage, ParsedSession

log = logging.getLogger(__name__)


class CodexParser:
    """Parses Codex CLI rollout-*.jsonl files into ParsedSession objects."""

    source_type = "codex"
    file_extensions = (".jsonl",)

    def parse_file(self, path: str) -> ParsedSession | None:
        basename = os.path.basename(path)
        if not basename.startswith("rollout-"):
            return None

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            log.warning("Cannot stat file: %s", path)
            return None

        events = _read_events(path)
        if not events:
            return None

        session_id: str | None = None
        cwd: str | None = None
        started_at: str | None = None
        for e in events:
            if e.get("type") == "session_meta":
                p = e.get("payload") or {}
                session_id = p.get("id") if isinstance(p.get("id"), str) else None
                cwd = p.get("cwd") if isinstance(p.get("cwd"), str) else None
                ts = p.get("timestamp")
                started_at = ts if isinstance(ts, str) else None
                break

        if not session_id:
            # Fallback: rollout-<iso>-<uuid5-parts>.jsonl → take trailing uuid
            stem = os.path.splitext(basename)[0]
            parts = stem.split("-")
            session_id = "-".join(parts[-5:]) if len(parts) >= 5 else stem

        messages: list[ParsedMessage] = []
        title: str | None = None
        current_model: str | None = None
        current_turn: dict | None = None
        turn_counter = 0
        user_counter = 0

        def flush() -> None:
            nonlocal current_turn
            if current_turn is None:
                return
            content = "\n\n".join(current_turn["content"]) or None
            thinking = "\n\n".join(current_turn["thinking"]) or None
            if content or thinking:
                messages.append(ParsedMessage(
                    id=current_turn["id"],
                    session_id=session_id,
                    role="assistant",
                    timestamp=current_turn["timestamp"],
                    content=content,
                    thinking=thinking,
                    model=current_turn["model"],
                ))
            current_turn = None

        for e in events:
            etype = e.get("type")
            payload = e.get("payload") if isinstance(e.get("payload"), dict) else {}
            ts = e.get("timestamp") if isinstance(e.get("timestamp"), str) else None

            if etype == "turn_context":
                m = payload.get("model")
                if isinstance(m, str) and m:
                    current_model = m
                continue

            if etype != "event_msg":
                continue

            inner = payload.get("type")
            if inner == "user_message":
                flush()
                msg = payload.get("message")
                if not isinstance(msg, str) or not msg.strip():
                    continue
                user_counter += 1
                text = msg.strip()
                if title is None:
                    title = text[:80]
                messages.append(ParsedMessage(
                    id=f"{session_id}-u{user_counter}",
                    session_id=session_id,
                    role="user",
                    timestamp=ts,
                    content=text,
                ))
            elif inner == "agent_message":
                msg = payload.get("message")
                if not isinstance(msg, str) or not msg.strip():
                    continue
                if current_turn is None:
                    turn_counter += 1
                    current_turn = {
                        "id": f"{session_id}-a{turn_counter}",
                        "timestamp": ts,
                        "model": current_model,
                        "thinking": [],
                        "content": [],
                    }
                phase = payload.get("phase")
                if phase == "commentary":
                    current_turn["thinking"].append(msg.strip())
                else:
                    current_turn["content"].append(msg.strip())

        flush()

        if not messages:
            return None

        return ParsedSession(
            id=session_id,
            source=self.source_type,
            file_path=path,
            file_mtime=mtime,
            title=title,
            cwd=cwd,
            model=current_model,
            started_at=started_at,
            messages=messages,
        )


def _read_events(path: str) -> list[dict]:
    events: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("%s:%d malformed JSON, skipping", path, lineno)
                    continue
                if isinstance(obj, dict):
                    events.append(obj)
    except OSError as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return []
    return events
