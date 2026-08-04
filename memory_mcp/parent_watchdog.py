"""Exit when the MCP client that spawned us goes away.

Why this exists: a stdio MCP server is supposed to notice its client dying
through EOF on stdin. That is not reliable on Windows. If any other process
inherited the write end of our stdin pipe (a shell shim, a broker, a sibling
spawned by the same client), the handle stays open after the client dies and
we block on a read that will never return EOF.

An orphan that survives that way is not harmless: it keeps an open SQLite
connection to ~/.memory_mcp/memory.db, so the next client's ``init_db()``
fails its schema DDL with "database is locked" and every memory tool in the
new session is dead on arrival. Under ``pythonw.exe`` the orphan has no
console or window, so nothing on screen says why.

So we watch the parent directly and hard-exit when it terminates. Uses
``os._exit`` on purpose: interpreter shutdown can block on the same wedged
state we are escaping, and there is nothing worth flushing.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

log = logging.getLogger(__name__)

# POSIX fallback poll interval. The Windows path blocks on a real handle.
_POLL_SECONDS = 5.0


def _exit_orphaned(ppid: int) -> None:
    log.warning("Parent process %s exited; shutting down to release the database", ppid)
    sys.stderr.flush()
    os._exit(0)


def _watch_windows(ppid: int) -> None:
    import ctypes
    from ctypes import wintypes

    SYNCHRONIZE = 0x00100000
    WAIT_OBJECT_0 = 0x0

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD

    handle = kernel32.OpenProcess(SYNCHRONIZE, False, ppid)
    if not handle:
        # Parent is already gone, or we cannot observe it. Either way there is
        # no client to serve.
        log.warning(
            "Cannot open parent process %s (error %s); assuming orphaned",
            ppid, ctypes.get_last_error(),
        )
        _exit_orphaned(ppid)
        return

    try:
        if kernel32.WaitForSingleObject(handle, 0xFFFFFFFF) == WAIT_OBJECT_0:
            _exit_orphaned(ppid)
    finally:
        kernel32.CloseHandle(handle)


def _watch_posix(ppid: int) -> None:
    while True:
        # Reparented to init/reaper, or the pid is simply gone.
        if os.getppid() != ppid:
            _exit_orphaned(ppid)
            return
        try:
            os.kill(ppid, 0)
        except ProcessLookupError:
            _exit_orphaned(ppid)
            return
        except PermissionError:
            pass  # Alive, just not ours to signal.
        threading.Event().wait(_POLL_SECONDS)


def start_parent_watchdog() -> None:
    """Start the watchdog thread. Safe to call once at process start."""
    ppid = os.getppid()
    if ppid <= 0:
        log.warning("No parent pid available; parent watchdog disabled")
        return

    target = _watch_windows if sys.platform == "win32" else _watch_posix
    threading.Thread(
        target=target,
        args=(ppid,),
        name="parent-watchdog",
        daemon=True,
    ).start()
    log.info("Parent watchdog armed on pid %s", ppid)
