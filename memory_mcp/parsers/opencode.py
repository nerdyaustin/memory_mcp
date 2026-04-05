"""Parser for OpenCode sessions stored in a SQLite database.

OpenCode keeps all session data in a single SQLite DB at
$XDG_DATA_HOME/opencode/opencode.db (or ~/.local/share/opencode/opencode.db).
Unlike other parsers that read JSONL files, this one queries the DB directly
and returns multiple ParsedSessions — one per session row.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

from .base import ParsedMessage, ParsedSession

log = logging.getLogger(__name__)

# Part types we extract content from; everything else is skipped.
_TEXT_TYPES = frozenset({"text", "reasoning", "tool"})


def _epoch_ms_to_iso(ms: int | float) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


class OpenCodeParser:
    """Reads OpenCode's SQLite DB and yields ParsedSession objects."""

    source_type = "opencode"

    def parse_db(self, path: str) -> list[ParsedSession]:
        """Open the OpenCode DB read-only and return all sessions.

        Returns an empty list on any database-level error (missing file,
        locked DB, corrupt schema, etc.).
        """
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            log.warning("Cannot stat OpenCode DB: %s", path)
            return []

        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error:
            log.warning("Cannot open OpenCode DB (read-only): %s", path)
            return []

        conn.row_factory = sqlite3.Row
        try:
            return self._read_sessions(conn, path, mtime)
        except sqlite3.Error:
            log.warning("Error querying OpenCode DB: %s", path, exc_info=True)
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_sessions(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        mtime: float,
    ) -> list[ParsedSession]:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, directory, time_created "
            "FROM session ORDER BY time_created DESC"
        )
        session_rows = cur.fetchall()

        results: list[ParsedSession] = []
        for srow in session_rows:
            sid = srow["id"]
            started_at = _epoch_ms_to_iso(srow["time_created"]) if srow["time_created"] else None

            messages, total_cost = self._parse_session_messages(conn, sid)

            results.append(ParsedSession(
                id=sid,
                source=self.source_type,
                file_path=db_path,
                file_mtime=mtime,
                title=srow["title"] or None,
                cwd=srow["directory"] or None,
                started_at=started_at,
                total_cost_usd=total_cost,
                messages=messages,
            ))

        return results

    def _parse_session_messages(
        self,
        conn: sqlite3.Connection,
        session_id: str,
    ) -> tuple[list[ParsedMessage], float]:
        """Return (messages, total_cost) for one session."""
        cur = conn.cursor()
        cur.execute(
            "SELECT id, session_id, time_created, data "
            "FROM message WHERE session_id = ? ORDER BY time_created",
            (session_id,),
        )
        msg_rows = cur.fetchall()

        messages: list[ParsedMessage] = []
        total_cost = 0.0

        for mrow in msg_rows:
            try:
                mdata = json.loads(mrow["data"]) if mrow["data"] else {}
            except (json.JSONDecodeError, TypeError):
                log.warning("Malformed message data for message %s — skipping", mrow["id"])
                continue

            if not isinstance(mdata, dict):
                continue

            role = mdata.get("role", "")
            if role not in ("user", "assistant"):
                continue

            msg_id: str = mrow["id"]
            ts = _epoch_ms_to_iso(mrow["time_created"]) if mrow["time_created"] else None
            parent_id = mdata.get("parentID")
            model = mdata.get("modelID")

            cost: float | None = None
            if role == "assistant":
                raw_cost = mdata.get("cost")
                if raw_cost is not None:
                    try:
                        cost = float(raw_cost)
                        total_cost += cost
                    except (ValueError, TypeError):
                        pass

            # Gather parts for this message.
            parts = self._fetch_parts(conn, msg_id)
            parsed = self._assemble_message(
                parts,
                msg_id=msg_id,
                session_id=session_id,
                role=role,
                timestamp=ts,
                parent_id=parent_id,
                model=model,
                cost=cost,
            )
            messages.extend(parsed)

        return messages, total_cost

    def _fetch_parts(
        self, conn: sqlite3.Connection, message_id: str
    ) -> list[dict]:
        """Return parsed part data dicts for a message, in order."""
        cur = conn.cursor()
        cur.execute(
            "SELECT id, message_id, data "
            "FROM part WHERE message_id = ? ORDER BY time_created",
            (message_id,),
        )
        out: list[dict] = []
        for row in cur:
            try:
                pdata = json.loads(row["data"]) if row["data"] else {}
            except (json.JSONDecodeError, TypeError):
                log.warning("Malformed part data for part %s — skipping", row["id"])
                continue
            if isinstance(pdata, dict):
                out.append(pdata)
        return out

    def _assemble_message(
        self,
        parts: list[dict],
        *,
        msg_id: str,
        session_id: str,
        role: str,
        timestamp: str | None,
        parent_id: str | None,
        model: str | None,
        cost: float | None,
    ) -> list[ParsedMessage]:
        """Build one or more ParsedMessage from a message's parts.

        Text and reasoning parts are folded into a single message.
        Each completed/errored tool part becomes its own ParsedMessage
        (role="tool_use") emitted after the main message.
        """
        text_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        tool_messages: list[ParsedMessage] = []

        for p in parts:
            ptype = p.get("type")
            if ptype not in _TEXT_TYPES:
                continue

            if ptype == "text":
                text = p.get("text")
                if text:
                    text_chunks.append(text)

            elif ptype == "reasoning":
                text = p.get("text")
                if text:
                    reasoning_chunks.append(text)

            elif ptype == "tool":
                state = p.get("state") or {}
                status = state.get("status", "")
                if status not in ("completed", "error"):
                    # Skip pending/running tool invocations.
                    continue

                tool_output = state.get("output", "")
                if status == "error":
                    tool_output = state.get("error", tool_output or "")

                raw_input = state.get("input")
                try:
                    tool_input = json.dumps(raw_input) if raw_input is not None else None
                except (TypeError, ValueError):
                    tool_input = None

                tool_messages.append(ParsedMessage(
                    id=f"{msg_id}-tool-{p.get('id', '')}",
                    session_id=session_id,
                    role="tool_use",
                    timestamp=timestamp,
                    parent_id=msg_id,
                    tool_name=p.get("tool"),
                    tool_input=tool_input,
                    tool_output=tool_output or None,
                    model=model,
                ))

        result: list[ParsedMessage] = []

        content = "\n".join(text_chunks) if text_chunks else None
        thinking = "\n".join(reasoning_chunks) if reasoning_chunks else None

        # Always emit the primary message (even if content is empty for an
        # assistant turn that only used tools).
        result.append(ParsedMessage(
            id=msg_id,
            session_id=session_id,
            role=role,
            timestamp=timestamp,
            parent_id=parent_id,
            content=content,
            thinking=thinking,
            model=model,
            cost_usd=cost,
        ))

        result.extend(tool_messages)
        return result
