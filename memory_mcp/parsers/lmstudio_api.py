"""Parser for LM Studio API log files captured by lms-log-capture.

Each daily log file contains sequential JSON lines from `lms log stream`:

    {"timestamp": ms, "data": {"type": "llm.prediction.input",  "input": "...", "modelIdentifier": "..."}}
    {"timestamp": ms, "data": {"type": "llm.prediction.output", "output": "...", "modelIdentifier": "...", "stats": {...}}}

Input lines contain the full chat-template-formatted prompt.  The template
uses ``<|turn>ROLE\\n...CONTENT...<turn|>`` markers.  We parse these to
extract individual messages.

Exchanges are grouped into sessions by time proximity (gap > 10 minutes
= new session).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone

from .base import ParsedMessage, ParsedSession

log = logging.getLogger(__name__)

# Max gap (seconds) between exchanges before starting a new session.
_SESSION_GAP = 600  # 10 minutes

# Pattern to split the chat-template-formatted input into turns.
# Matches: <|turn>role\ncontent<turn|>
_TURN_RE = re.compile(
    r"<\|turn>(\w+)\n(.*?)<turn\|>",
    re.DOTALL,
)


def _parse_turns(raw_input: str) -> list[tuple[str, str]]:
    """Extract (role, content) pairs from a chat-template-formatted prompt.

    Strips thinking markers (<|channel>...<channel|>) and the trailing
    empty model turn (the generation prompt).
    """
    turns: list[tuple[str, str]] = []
    for match in _TURN_RE.finditer(raw_input):
        role = match.group(1)
        content = match.group(2).strip()

        # Map template roles to standard roles.
        if role == "model":
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            role = "user"

        # Strip thinking channel markers.
        content = re.sub(r"<\|channel>.*?<channel\|>", "", content, flags=re.DOTALL).strip()

        if content:
            turns.append((role, content))
    return turns


def _ts_to_iso(ms: int | float) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


class LmStudioApiParser:
    """Parses daily API log files from lms-log-capture."""

    source_type = "lmstudio_api"

    def parse_file(self, path: str) -> ParsedSession | None:
        """Parse a daily log file into one or more sessions.

        Returns the first session found.  If multiple sessions exist
        (separated by time gaps), they are returned as a single session
        for simplicity — the FTS index makes them all searchable regardless.
        """
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            log.warning("Cannot stat file: %s", path)
            return None

        # Read and pair input/output entries.
        exchanges: list[dict] = []
        pending_input: dict | None = None

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

                    data = entry.get("data", {})
                    entry_type = data.get("type", "")
                    timestamp = entry.get("timestamp", 0)

                    if entry_type == "llm.prediction.input":
                        pending_input = {
                            "timestamp": timestamp,
                            "input": data.get("input", ""),
                            "model": data.get("modelIdentifier"),
                        }

                    elif entry_type == "llm.prediction.output" and pending_input is not None:
                        exchanges.append({
                            "timestamp": pending_input["timestamp"],
                            "model": data.get("modelIdentifier") or pending_input["model"],
                            "input": pending_input["input"],
                            "output": data.get("output", ""),
                            "stats": data.get("stats"),
                        })
                        pending_input = None

        except OSError:
            log.warning("Cannot read file: %s", path)
            return None

        if not exchanges:
            return None

        # Session ID from filename: "2026-04-11.jsonl" -> "lmsapi-2026-04-11"
        basename = os.path.basename(path)
        session_id = "lmsapi-" + basename.replace(".jsonl", "")

        # Build messages from exchanges.
        messages: list[ParsedMessage] = []
        first_model: str | None = None
        msg_idx = 0

        for ex in exchanges:
            model = ex["model"]
            if first_model is None:
                first_model = model

            # Extract user messages from the input template.
            # Only take the LAST user message — earlier ones are context
            # from prior exchanges that the caller re-sent.
            turns = _parse_turns(ex["input"])
            last_user: str | None = None
            for role, content in turns:
                if role == "user":
                    last_user = content

            if last_user:
                messages.append(ParsedMessage(
                    id=f"{session_id}-{msg_idx}",
                    session_id=session_id,
                    role="user",
                    content=last_user,
                    model=model,
                    timestamp=_ts_to_iso(ex["timestamp"]),
                ))
                msg_idx += 1

            if ex["output"]:
                messages.append(ParsedMessage(
                    id=f"{session_id}-{msg_idx}",
                    session_id=session_id,
                    role="assistant",
                    content=ex["output"],
                    model=model,
                    timestamp=_ts_to_iso(ex["timestamp"]),
                ))
                msg_idx += 1

        if not messages:
            return None

        # Title from first user message.
        first_user_msg = next((m for m in messages if m.role == "user"), None)
        title = first_user_msg.content[:80] if first_user_msg else None

        return ParsedSession(
            id=session_id,
            source=self.source_type,
            file_path=path,
            file_mtime=mtime,
            title=title,
            model=first_model,
            started_at=_ts_to_iso(exchanges[0]["timestamp"]),
            messages=messages,
        )
