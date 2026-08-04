"""Startup-speed contract test for the v0.3.0 lazy-startup refactor.

Measures wall time from subprocess.Popen to the server's tools/list response
arriving on stdout. Spans Python import cost and lifespan setup, so an
import-time regression (e.g. someone re-adds an eager fastembed import) shows
up immediately as a test failure.

The threshold targets the contract documented in memory_mcp/CLAUDE.md: the
server must answer tools/list within a small constant time on every cold
boot, regardless of whether the embedding model has ever been loaded.
"""

import json
import os
import queue
import subprocess
import threading
import sys
import tempfile
from pathlib import Path
from time import monotonic


STARTUP_THRESHOLD_SECONDS = 1.5
# Hard cap on any single readline — well above the threshold so a legitimate
# slow boot is still measured, but finite so a deadlocked server fails loudly
# instead of hanging CI or an interactive test run.
READ_TIMEOUT_SECONDS = 10.0


def _write_message(proc: subprocess.Popen, payload: dict) -> None:
    line = json.dumps(payload) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()


def _read_response(proc: subprocess.Popen, expected_id: int) -> dict:
    """Read JSON-RPC lines from stdout until we see the response with
    matching id. Notifications (no id) are ignored. Raises RuntimeError
    if no matching response arrives within READ_TIMEOUT_SECONDS — catches
    a deadlocked server without hanging the caller.

    Uses a daemon reader thread + queue instead of select(): on Windows,
    select() only works on sockets, not pipes."""
    q: queue.Queue = getattr(proc, "_line_queue", None)
    if q is None:
        q = queue.Queue()
        proc._line_queue = q

        def _pump() -> None:
            for raw in proc.stdout:
                q.put(raw)
            q.put(None)  # EOF sentinel

        threading.Thread(target=_pump, daemon=True).start()

    deadline = monotonic() + READ_TIMEOUT_SECONDS
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"No response for id={expected_id} within "
                f"{READ_TIMEOUT_SECONDS}s — server likely deadlocked."
            )
        try:
            raw = q.get(timeout=remaining)
        except queue.Empty:
            raise RuntimeError(
                f"No response for id={expected_id} within "
                f"{READ_TIMEOUT_SECONDS}s — server likely deadlocked."
            )
        if raw is None:
            stderr_tail = proc.stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"Server closed stdout before responding. stderr:\n{stderr_tail}"
            )
        try:
            msg = json.loads(raw.decode())
        except json.JSONDecodeError:
            continue
        if msg.get("id") == expected_id:
            return msg


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="memory_mcp_startup_")
    db_path = Path(tmp) / "startup.db"
    env = {**os.environ, "MEMORY_MCP_DB": str(db_path)}

    repo_root = Path(__file__).resolve().parent.parent

    t0 = monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-m", "memory_mcp"],
        cwd=str(repo_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        _write_message(proc, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "startup-test", "version": "0"},
            },
        })
        init_resp = _read_response(proc, 1)
        assert "result" in init_resp, f"initialize failed: {init_resp}"

        _write_message(proc, {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

        _write_message(proc, {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })
        tools_resp = _read_response(proc, 2)
        t1 = monotonic()

        elapsed = t1 - t0
        tools = tools_resp.get("result", {}).get("tools", [])
        names = sorted(t["name"] for t in tools)
        print(f"Popen -> tools/list: {elapsed*1000:.1f} ms")
        print(f"Tools ({len(names)}): {names}")

        assert len(names) == 10, f"Expected 10 tools, got {len(names)}: {names}"
        assert elapsed < STARTUP_THRESHOLD_SECONDS, (
            f"Startup too slow: {elapsed*1000:.1f} ms exceeds "
            f"{STARTUP_THRESHOLD_SECONDS*1000:.0f} ms threshold. "
            f"Likely an eager import or pre-yield blocking call has been added."
        )
        print("=== STARTUP TEST PASSED ===")
        return 0
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
