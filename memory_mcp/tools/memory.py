import json

from mcp.server.fastmcp import FastMCP, Context

from memory_mcp import db


def _get_db(ctx: Context):
    return ctx.request_context.lifespan_context["db"]


def _parse_tags(tags: str) -> list[str] | None:
    """Split comma-separated tags string, strip whitespace, drop empties. None if no tags."""
    parsed = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    return parsed or None


def _format_tags(tags_raw) -> str:
    """Format a raw tags value (JSON string or list) for display."""
    if not tags_raw:
        return ""
    try:
        tag_list = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
    except (json.JSONDecodeError, TypeError):
        return ""
    return f" [{', '.join(tag_list)}]" if tag_list else ""

def register_memory_tools(mcp: FastMCP):
    @mcp.tool(
        description=(
            "Save a note to persistent memory. Use this to remember important"
            " context, decisions, patterns, or user preferences across sessions."
        )
    )
    def save_memory(
        content: str, tags: str = "", context: str = "", ctx: Context = None
    ) -> str:
        conn = _get_db(ctx)
        tags_list = _parse_tags(tags)
        context_str = context or None
        memory_id = db.save_memory(conn, content, tags_list, context_str)
        return f"Saved memory #{memory_id}."

    @mcp.tool(
        description=(
            "Search saved memories. Use short keywords, not natural language"
            " phrases — each word is matched independently and ranked by relevance."
        )
    )
    def search_memory(
        query: str, tags: str = "", limit: int = 10, ctx: Context = None
    ) -> str:
        conn = _get_db(ctx)
        tags_list = _parse_tags(tags)
        results = db.search_memories(conn, query, tags_list, limit)
        if not results:
            return f"No memories found matching '{query}'."
        lines = []
        for i, mem in enumerate(results, 1):
            tag_str = _format_tags(mem.get("tags"))
            lines.append(
                f"{i}. [#{mem['id']}]{tag_str} ({mem['created_at']})\n"
                f"   {mem['content']}"
            )
        return "\n\n".join(lines)

    @mcp.tool(
        description="List recent saved memories, optionally filtered by tag."
    )
    def list_memories(
        tag: str = "", limit: int = 20, ctx: Context = None
    ) -> str:
        conn = _get_db(ctx)
        results = db.list_memories(conn, tag or None, limit)
        if not results:
            return "No memories saved yet."
        lines = []
        for i, mem in enumerate(results, 1):
            content_snippet = mem["content"][:200]
            if len(mem["content"]) > 200:
                content_snippet += "..."
            tag_str = _format_tags(mem.get("tags"))
            lines.append(
                f"{i}. [#{mem['id']}]{tag_str} ({mem['created_at']})\n"
                f"   {content_snippet}"
            )
        return "\n\n".join(lines)

    @mcp.tool(
        description="Delete a specific memory by its ID."
    )
    def delete_memory(memory_id: int, ctx: Context = None) -> str:
        conn = _get_db(ctx)
        if db.delete_memory(conn, memory_id):
            return f"Deleted memory #{memory_id}."
        return f"Memory #{memory_id} not found."
