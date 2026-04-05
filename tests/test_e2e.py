"""End-to-end test: starts the MCP server as a subprocess and exercises every tool."""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure we test against a throwaway DB
_tmp = tempfile.mkdtemp()
os.environ["MEMORY_MCP_DB"] = str(Path(_tmp) / "test.db")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "memory_mcp"],
        env={**os.environ, "MEMORY_MCP_DB": os.environ["MEMORY_MCP_DB"]},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("=== initialized ===")

            # --- tools/list ---
            tools = await session.list_tools()
            tool_names = sorted(t.name for t in tools.tools)
            print(f"Tools ({len(tool_names)}): {tool_names}")
            assert len(tool_names) == 8, f"Expected 8 tools, got {len(tool_names)}"

            # --- save_memory ---
            r = await session.call_tool("save_memory", {
                "content": "Austin prefers SQLite over Postgres for local tools",
                "tags": "preferences, architecture",
                "context": "memory_mcp design discussion",
            })
            text = r.content[0].text
            print(f"save_memory: {text}")
            assert "Saved memory #" in text

            r = await session.call_tool("save_memory", {
                "content": "OMP session files are JSONL with type headers",
                "tags": "omp, formats",
            })
            print(f"save_memory: {r.content[0].text}")

            # --- search_memory ---
            r = await session.call_tool("search_memory", {"query": "SQLite"})
            text = r.content[0].text
            print(f"search_memory 'SQLite': {text[:120]}...")
            assert "SQLite" in text
            assert "preferences" in text

            # --- list_memories ---
            r = await session.call_tool("list_memories", {})
            text = r.content[0].text
            print(f"list_memories: {text[:120]}...")
            assert "#1" in text or "#2" in text

            # --- list_memories with tag filter ---
            r = await session.call_tool("list_memories", {"tag": "omp"})
            text = r.content[0].text
            print(f"list_memories tag=omp: {text[:120]}...")
            assert "JSONL" in text

            # --- delete_memory ---
            r = await session.call_tool("delete_memory", {"memory_id": 2})
            text = r.content[0].text
            print(f"delete_memory: {text}")
            assert "Deleted" in text

            r = await session.call_tool("delete_memory", {"memory_id": 999})
            text = r.content[0].text
            print(f"delete_memory (missing): {text}")
            assert "not found" in text

            # --- list_sessions ---
            r = await session.call_tool("list_sessions", {"limit": 3})
            text = r.content[0].text
            print(f"list_sessions: {text[:200]}...")
            assert "session" in text.lower()

            # --- search_sessions ---
            r = await session.call_tool("search_sessions", {"query": "proxmox", "limit": 2})
            text = r.content[0].text
            print(f"search_sessions 'proxmox': {text[:200]}...")
            # May or may not find results depending on session data, just ensure no crash

            # --- get_session (grab first session id from list) ---
            r = await session.call_tool("list_sessions", {"limit": 1, "source": "omp"})
            list_text = r.content[0].text
            # Not easy to extract ID from formatted text, so test with a known-bad ID
            r = await session.call_tool("get_session", {"session_id": "nonexistent-id"})
            text = r.content[0].text
            print(f"get_session (missing): {text}")
            assert "not found" in text.lower()

            # --- refresh_sessions ---
            r = await session.call_tool("refresh_sessions", {})
            text = r.content[0].text
            print(f"refresh_sessions: {text}")
            assert "Scanned" in text
            assert "sources" in text

            print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
