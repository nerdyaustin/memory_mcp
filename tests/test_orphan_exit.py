"""The server must not outlive its MCP client, even when stdin never EOFs.

Real reproduction of the failure that took memory search down: an orphaned
server process kept ~/.memory_mcp/memory.db locked, so every later client got
"database is locked" from init_db() and lost all memory tools.

Topology (all real processes, no fakes):

    test  ->  shim   ->  server   (stdin=PIPE, owned by shim)
                    \\->  keeper   (spawned with close_fds=False, so it
                                    inherits the shim's copy of the pipe's
                                    write handle)

Killing the shim leaves the keeper holding the write end open, so the server
never sees EOF on stdin. Without the parent watchdog it hangs around forever.
With it, it exits within seconds.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXIT_DEADLINE_SECONDS = 20.0
REPO_ROOT = Path(__file__).resolve().parent.parent

SHIM = r"""
import os, subprocess, sys, time

# Build the pipe by hand so the write end can be marked inheritable: a plain
# stdin=PIPE handle is private to this process and closes when we die, which
# would hand the server an EOF and hide the bug under test.
read_fd, write_fd = os.pipe()
os.set_inheritable(write_fd, True)

server = subprocess.Popen(
    [sys.executable, "-m", "memory_mcp"],
    stdin=read_fd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
os.close(read_fd)

# close_fds=False + an inheritable write end: the keeper holds stdin open
# after we are killed, so the server never observes EOF.
keeper = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(120)"], close_fds=False,
)
print(f"{server.pid} {keeper.pid}", flush=True)
time.sleep(120)
"""


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True, text=True,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True)
    else:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="memory_mcp_orphan_")
    env = {**os.environ, "MEMORY_MCP_DB": str(Path(tmp) / "orphan.db")}
    # Sync would just add network noise to a lifecycle test.
    env.pop("MEMORY_MCP_SYNC_URL", None)

    shim = subprocess.Popen(
        [sys.executable, "-c", SHIM],
        cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    server_pid = keeper_pid = None
    try:
        server_pid, keeper_pid = (
            int(x) for x in shim.stdout.readline().decode().split()
        )
        # Let the server finish booting so it is holding a live DB connection.
        time.sleep(3)
        assert _alive(server_pid), "server died before the test began"

        _kill(shim.pid)  # no /T: the server must survive its parent's death
        shim.wait(timeout=10)
        assert _alive(keeper_pid), "keeper died; stdin would EOF and invalidate the test"

        deadline = time.monotonic() + EXIT_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            if not _alive(server_pid):
                elapsed = EXIT_DEADLINE_SECONDS - (deadline - time.monotonic())
                print(f"Server exited {elapsed:.1f}s after its parent died")
                print("=== ORPHAN EXIT TEST PASSED ===")
                return 0
            time.sleep(0.25)

        raise AssertionError(
            f"Server pid {server_pid} still running {EXIT_DEADLINE_SECONDS}s after "
            "its parent died — it will hold the SQLite write lock and break the "
            "next client's init_db()."
        )
    finally:
        for pid in (keeper_pid, server_pid):
            if pid:
                _kill(pid)
        _kill(shim.pid)


if __name__ == "__main__":
    raise SystemExit(main())
