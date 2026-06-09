import json

from mcp.server.fastmcp import FastMCP, Context

from memory_mcp import db
from memory_mcp.readiness import ensure_semantic_ready, get_embedder_if_ready


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
        # Opportunistic embedding: only embed if the model is already loaded.
        # Never trigger a cold load from save_memory — periodic backfill
        # handles unembedded rows once a real semantic call warms the model.
        embedder = get_embedder_if_ready()
        tags_list = _parse_tags(tags)
        context_str = context or None

        embedding = None
        if embedder is not None:
            try:
                embedding = embedder.embed(content)
            except Exception:
                pass  # Non-fatal; memory is saved without a vector.

        memory_id = db.save_memory(
            conn, content, tags_list, context_str, embedding=embedding,
        )
        return f"Saved memory #{memory_id}."

    @mcp.tool(
        description=(
            "Search saved memories. Use short keywords, not natural language"
            " phrases \u2014 each word is matched independently and ranked by"
            " relevance. Set semantic=true for meaning-based search (finds"
            " related concepts even when exact words differ)."
        )
    )
    async def search_memory(
        query: str,
        tags: str = "",
        limit: int = 10,
        semantic: bool = False,
        ctx: Context = None,
    ) -> str:
        conn = _get_db(ctx)
        tags_list = _parse_tags(tags)

        if semantic:
            try:
                embedder = await ensure_semantic_ready()
            except Exception as exc:
                return f"Semantic search unavailable: {exc}"
            query_emb = embedder.embed(query)
            results = db.semantic_search_memories(conn, query_emb, tags_list, limit)
            mode = "semantic"
        else:
            results = db.search_memories(conn, query, tags_list, limit)
            mode = "keyword"

        if not results:
            return f"No memories found matching '{query}' ({mode} search)."

        lines = []
        for i, mem in enumerate(results, 1):
            tag_str = _format_tags(mem.get("tags"))
            dist = mem.get("distance")
            dist_str = f" [dist={dist:.3f}]" if dist is not None else ""
            lines.append(
                f"{i}. [#{mem['id']}]{tag_str}{dist_str} ({mem['created_at']})\n"
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
