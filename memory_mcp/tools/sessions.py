"""MCP tools for browsing and searching indexed AI sessions."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP, Context

from memory_mcp import db
from memory_mcp.readiness import ensure_semantic_ready
from memory_mcp.scanner import scan_sessions

# Per-field content caps for display.
MAX_LINE = 4096


def _get_db(ctx: Context):
    return ctx.request_context.lifespan_context["db"]

def _clip(text: str | None, limit: int = MAX_LINE) -> str:
    """Truncate a string to *limit* chars, appending '...' when clipped."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _ts(raw: str | None) -> str:
    """Format an ISO timestamp for display.  Returns '' if absent."""
    if not raw:
        return ""
    # Show date + time to the minute: '2026-03-29 02:17'
    return raw[:16].replace("T", " ")


def _session_line(s: dict) -> str:
    """One-line summary for a session, with its ID for drill-down."""
    sid = s.get("id", "?")
    title = s.get("title") or "Untitled"
    ts = _ts(s.get("started_at"))
    source = s.get("source", "?")
    cwd = s.get("cwd") or ""
    count = s.get("message_count") or 0
    return f"[{source}] {title}  ({ts}, {count} msgs, {cwd})  id={sid}"


def _page_footer(offset: int, page_len: int, total: int, noun: str) -> str | None:
    """Return a '... N more' hint if there are more items, else None."""
    end = offset + page_len
    if end < total:
        return f"\n... {total - end} more {noun} (use offset={end})"
    return None


def register_session_tools(mcp: FastMCP) -> None:
    @mcp.tool(
        description=(
            "List recent AI coding sessions. Filter by source "
            "('claude_code' or 'omp') or by project path substring."
        ),
    )
    def list_sessions(
        source: str = "",
        project: str = "",
        limit: int = 20,
        offset: int = 0,
        ctx: Context = None,
    ) -> str:
        conn = _get_db(ctx)
        total = db.count_sessions(conn, source or None, project or None)
        if total == 0:
            return (
                "No sessions found. Try running refresh_sessions to index "
                "new session files, or adjust your filters."
            )
        rows = db.list_sessions(
            conn, source or None, project or None, limit, offset,
        )
        end = offset + len(rows)
        lines = [f"Showing {offset + 1}\u2013{end} of {total} session(s):\n"]
        for s in rows:
            lines.append(_session_line(s))
        footer = _page_footer(offset, len(rows), total, "sessions")
        if footer:
            lines.append(footer)
        return "\n".join(lines)

    @mcp.tool(
        description=(
            "Retrieve the conversation from a specific session. Shows the "
            "full message flow including user messages, assistant responses, "
            "and tool usage."
        ),
    )
    def get_session(
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        ctx: Context = None,
    ) -> str:
        conn = _get_db(ctx)
        header, messages, total = db.get_session_messages(
            conn, session_id, limit, offset,
        )
        if header is None:
            return f"Session '{session_id}' not found."

        title = header.get("title") or "Untitled"
        ts = _ts(header.get("started_at"))
        cwd = header.get("cwd") or ""
        model = header.get("model") or ""
        end = offset + len(messages)
        parts = [
            f"Session: {title}  (id={header['id']})",
            f"Date: {ts}  Model: {model}  CWD: {cwd}",
            f"Messages {offset + 1}\u2013{end} of {total}",
            "---",
        ]

        for m in messages:
            mts = _ts(m.get("timestamp"))
            prefix = f"[{mts}] " if mts else ""
            role = (m.get("role") or "").lower()

            if role == "user":
                parts.append(f"{prefix}USER: {_clip(m.get('content'), 4096)}")
            elif role == "assistant":
                thinking = m.get("thinking")
                if thinking:
                    parts.append(f"{prefix}THINKING: {_clip(thinking, 2000)}")
                parts.append(f"{prefix}ASSISTANT: {_clip(m.get('content'), 8000)}")
            elif m.get("tool_name"):
                name = m["tool_name"]
                inp = _clip(m.get("tool_input"), 2000)
                out = _clip(m.get("tool_output"), 4096)
                parts.append(f"{prefix}{name}: {inp}")
                if out:
                    parts.append(f"  \u2192 {out}")
            else:
                parts.append(
                    f"{prefix}{role.upper()}: {_clip(m.get('content'), 4096)}"
                )

        footer = _page_footer(offset, len(messages), total, "messages")
        if footer:
            parts.append(footer)
        return "\n".join(parts)

    @mcp.tool(
        description=(
            "Get tool calls and results for a specific session. "
            "Filter by tool name for targeted queries. "
            "Each entry shows role, tool name, input, and output."
        ),
    )
    def get_tool_calls(
        session_id: str,
        tool_name: str = "",
        limit: int = 50,
        offset: int = 0,
        ctx: Context = None,
    ) -> str:
        conn = _get_db(ctx)
        name_filter = tool_name or None
        calls, total = db.get_tool_calls(conn, session_id, name_filter, limit, offset)
        if not calls:
            return f"No tool calls found for session '{session_id}'."

        end = offset + len(calls)
        lines = [
            f"Tool calls for session {session_id} ({offset + 1}–{end} of {total}):\n"
        ]
        for c in calls:
            role = c.get("role", "")
            name = c.get("tool_name") or "(no name)"
            ts = _ts(c.get("timestamp"))
            prefix = f"[{ts}] " if ts else ""
            input_snippet = _clip(c.get("tool_input"), 2000)
            output_snippet = _clip(c.get("tool_output"), 4096)

            if role == "tool_use":
                lines.append(f"{prefix}CALL {name}: {input_snippet}")
            elif role == "tool_result":
                lines.append(f"{prefix}RESULT {name}: {output_snippet}")
            else:
                lines.append(f"{prefix}{role.upper()} {name}")

        footer = _page_footer(offset, len(calls), total, "tool calls")
        if footer:
            lines.append(footer)
        return "\n".join(lines)

    @mcp.tool(
        description=(
            "Search across all session messages. Use short keywords, not"
            " natural language phrases \u2014 each word is matched independently"
            " and ranked by relevance. Set semantic=true for meaning-based"
            " search (finds related concepts even when exact words differ)."
        ),
    )
    async def search_sessions(
        query: str,
        limit: int = 10,
        offset: int = 0,
        semantic: bool = False,
        ctx: Context = None,
    ) -> str:
        conn = _get_db(ctx)

        if semantic:
            try:
                embedder = await ensure_semantic_ready()
            except Exception as exc:
                return f"Semantic search unavailable: {exc}"
            query_emb = embedder.embed(query)
            results = db.semantic_search_messages(conn, query_emb, limit, offset)
            mode = "semantic"
        else:
            results = db.search_messages(conn, query, limit, offset)
            mode = "keyword"

        if not results:
            if offset > 0:
                return f"No more results for '{query}' at offset {offset} ({mode})."
            return f"No session messages found matching '{query}' ({mode} search)."

        start = offset + 1
        end = offset + len(results)
        lines = [f"Results {start}\u2013{end} for '{query}' ({mode}):\n"]
        for r in results:
            sid = r.get("session_id", "?")
            title = r.get("session_title") or "Untitled"
            source = r.get("session_source") or "?"
            role = (r.get("role") or "").upper()
            mts = _ts(r.get("timestamp"))
            session_ts = _ts(r.get("session_date"))
            ts_display = mts or session_ts

            # Pick the most meaningful content for the snippet.
            content = r.get("content") or r.get("thinking") or r.get("tool_output") or ""
            tool = r.get("tool_name")
            prefix = f"[{tool}] " if tool else ""
            dist = r.get("distance")
            dist_str = f" [dist={dist:.3f}]" if dist is not None else ""
            snippet = _clip(prefix + content, MAX_LINE)

            lines.append(f"[{source}] {title} ({ts_display}) {role}{dist_str}  session_id={sid}")
            lines.append(f"  {snippet}")
            lines.append("")

        # If we got a full page, hint there may be more.
        if len(results) == limit:
            lines.append(f"... may have more (use offset={end})")
        return "\n".join(lines)

    @mcp.tool(
        description=(
            "Scan for new or updated session files and index them. "
            "Run this if recent sessions aren't showing up in search results."
        ),
    )
    def refresh_sessions(ctx: Context = None) -> str:
        conn = _get_db(ctx)
        stats = scan_sessions(conn)
        return (
            f"Scanned {stats['sources_scanned']} sources, "
            f"found {stats['files_found']} files, "
            f"indexed {stats['files_indexed']} new, "
            f"skipped {stats['files_skipped']} unchanged, "
            f"{stats['errors']} errors."
        )
