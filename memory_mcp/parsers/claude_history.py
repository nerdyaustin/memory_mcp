"""Parser for Claude Code history.jsonl — the user-prompt-only log.

Claude Code keeps ~/.claude/history.jsonl as a running log of every user
prompt, even after the full session JSONL files are purged.  Each line is:

    {"display": "...", "pastedContents": {}, "timestamp": <epoch_ms>,
     "project": "C:\\...", "sessionId": "uuid"}

We group entries by sessionId and emit one ParsedSession per session.
These sessions contain only user messages (assistant responses are lost),
but they're invaluable for search when the full session files no longer
exist.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

from .base import ParsedMessage, ParsedSession

log = logging.getLogger(__name__)


def _epoch_ms_to_iso(ms: int | float) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


class ClaudeHistoryParser:
    """Parses ~/.claude/history.jsonl into per-session ParsedSessions."""

    source_type = "claude_code"

    def parse_file(self, path: str) -> list[ParsedSession]:
        """Return one ParsedSession per sessionId found in the file.

        Unlike other parsers that return a single session, history.jsonl
        contains many sessions interleaved. Returns a list instead.
        """
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            log.warning("Cannot stat file: %s", path)
            return []

        # Group entries by sessionId.
        sessions: dict[str, list[dict]] = defaultdict(list)
        try:
            with open(path, encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    sid = entry.get("sessionId")
                    if not sid:
                        continue
                    sessions[sid].append(entry)
        except OSError:
            log.warning("Cannot read file: %s", path)
            return []

        results: list[ParsedSession] = []
        for sid, entries in sessions.items():
            entries.sort(key=lambda e: e.get("timestamp", 0))
            messages: list[ParsedMessage] = []
            first_ts = None
            cwd = None
            for i, entry in enumerate(entries):
                ts_ms = entry.get("timestamp")
                ts_iso = _epoch_ms_to_iso(ts_ms) if ts_ms else None
                if first_ts is None and ts_iso:
                    first_ts = ts_iso
                if cwd is None:
                    cwd = entry.get("project")

                display = entry.get("display", "")
                # Include pasted content if present.
                pasted = entry.get("pastedContents")
                if isinstance(pasted, dict):
                    for name, text in pasted.items():
                        if text:
                            display += f"\n[pasted: {name}]\n{text}"

                messages.append(ParsedMessage(
                    id=f"hist-{sid}-{i}",
                    session_id=f"hist-{sid}",
                    role="user",
                    timestamp=ts_iso,
                    content=display or None,
                ))

            # Derive a title from the first prompt.
            first_msg = entries[0].get("display", "")
            title = first_msg[:80] if first_msg else None

            results.append(ParsedSession(
                id=f"hist-{sid}",
                source=self.source_type,
                file_path=path,
                file_mtime=mtime,
                title=title,
                cwd=cwd,
                started_at=first_ts,
                messages=messages,
            ))

        return results
