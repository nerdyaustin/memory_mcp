"""Walk configured session directories and index JSONL files into the database."""

from __future__ import annotations

import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from memory_mcp.config import get_session_sources
from memory_mcp.db import get_session_mtime, record_indexed_file, upsert_session
from memory_mcp.parsers import PARSERS
from memory_mcp.parsers.claude_history import ClaudeHistoryParser
from memory_mcp.parsers.opencode import OpenCodeParser

log = logging.getLogger(__name__)

# Workers for parallel file parsing.  File I/O on Windows benefits from
# higher concurrency than CPU count; 8 is a safe default.
_PARSE_WORKERS = 8


def _empty_stats() -> dict:
    return {
        "sources_scanned": 0,
        "files_found": 0,
        "files_indexed": 0,
        "files_skipped": 0,
        "errors": 0,
    }


def _merge_stats(total: dict, part: dict) -> None:
    for key in total:
        total[key] += part[key]


def _find_session_files(root: str, extensions: tuple[str, ...] = (".jsonl",)) -> list[str]:
    """Recursively find session files under *root* matching *extensions*."""
    paths: list[str] = []
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            if any(name.endswith(ext) for ext in extensions):
                paths.append(os.path.join(dirpath, name))
    return paths


def _index_session(
    db: sqlite3.Connection, session_dict: dict, stats: dict, machine_id: str,
) -> bool:
    """Upsert one parsed session and report whether it succeeded."""
    try:
        upsert_session(db, session_dict, machine_id)
        stats["files_indexed"] += 1
        return True
    except Exception:
        db.rollback()
        log.exception("Failed to index session %s", session_dict.get("id", "?"))
        stats["errors"] += 1
        return False


# ---------------------------------------------------------------------------
# History file (one file -> many sessions, not parallelizable)
def _scan_history_file(
    db: sqlite3.Connection, path: str, machine_id: str,
) -> dict:
    """Index ~/.claude/history.jsonl -- a single file containing many sessions."""
    stats = _empty_stats()
    stats["sources_scanned"] = 1
    stats["files_found"] = 1

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        log.warning("Cannot stat history file: %s", path)
        stats["errors"] += 1
        return stats

    stored_mtime = get_session_mtime(db, path)
    if stored_mtime is not None and stored_mtime == mtime:
        stats["files_skipped"] += 1
        return stats

    parser = ClaudeHistoryParser()
    try:
        sessions = parser.parse_file(path)
    except Exception:
        log.exception("Failed to parse history file %s", path)
        stats["errors"] += 1
        return stats

    succeeded = True
    for session in sessions:
        succeeded = _index_session(
            db, asdict(session), stats, machine_id,
        ) and succeeded
    if succeeded:
        record_indexed_file(db, path, mtime, "claude_history")

    return stats


# ---------------------------------------------------------------------------
# OpenCode DB (one SQLite DB -> many sessions, not parallelizable)
# ---------------------------------------------------------------------------
def _scan_opencode_db(
    db: sqlite3.Connection, path: str, machine_id: str,
) -> dict:
    """Index OpenCode's SQLite database -- one DB, many sessions."""
    stats = _empty_stats()
    stats["sources_scanned"] = 1
    stats["files_found"] = 1

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        log.warning("Cannot stat OpenCode DB: %s", path)
        stats["errors"] += 1
        return stats

    stored_mtime = get_session_mtime(db, path)
    if stored_mtime is not None and stored_mtime == mtime:
        stats["files_skipped"] += 1
        return stats

    parser = OpenCodeParser()
    try:
        sessions = parser.parse_db(path)
    except Exception:
        log.exception("Failed to parse OpenCode DB %s", path)
        stats["errors"] += 1
        return stats

    succeeded = True
    for session in sessions:
        succeeded = _index_session(
            db, asdict(session), stats, machine_id,
        ) and succeeded
    if succeeded:
        record_indexed_file(db, path, mtime, "opencode")

    return stats


# ---------------------------------------------------------------------------
# Directory source (threaded: parse in parallel, write in serial)
# ---------------------------------------------------------------------------

def _parse_one(parser, file_path: str) -> dict | None:
    """Parse a single file.  Runs in a worker thread -- no DB access."""
    try:
        session = parser.parse_file(file_path)
    except Exception:
        log.exception("Failed to parse %s", file_path)
        return None
    if session is None:
        return None
    return asdict(session)


def scan_source(
    db: sqlite3.Connection, source_type: str, source_path: str, machine_id: str,
) -> dict:
    """Scan a single source directory and index new/changed session files."""
    stats = _empty_stats()
    stats["sources_scanned"] = 1

    if source_type not in PARSERS:
        log.warning("Unknown source type %r -- skipping", source_type)
        return stats

    if not os.path.isdir(source_path):
        log.warning("Source path does not exist: %s -- skipping", source_path)
        return stats

    parser = PARSERS[source_type]()
    extensions = getattr(parser, "file_extensions", (".jsonl",))
    session_files = _find_session_files(source_path, extensions)
    stats["files_found"] = len(session_files)

    # Filter to files that actually need re-indexing (mtime changed).
    to_parse: list[tuple[str, float]] = []
    for file_path in session_files:
        try:
            mtime = os.path.getmtime(file_path)
        except OSError as exc:
            log.warning("Cannot stat %s: %s", file_path, exc)
            stats["errors"] += 1
            continue

        stored_mtime = get_session_mtime(db, file_path)
        if stored_mtime is not None and stored_mtime == mtime:
            stats["files_skipped"] += 1
            continue
        to_parse.append((file_path, mtime))

    if not to_parse:
        return stats

    # Parse files in parallel, write to DB on main thread.
    with ThreadPoolExecutor(max_workers=_PARSE_WORKERS) as pool:
        futures = {
            pool.submit(_parse_one, parser, fp): (fp, mtime)
            for fp, mtime in to_parse
        }
        for future in as_completed(futures):
            file_path, file_mtime = futures[future]
            try:
                session_dict = future.result()
            except Exception:
                log.exception("Worker crashed on %s", file_path)
                stats["errors"] += 1
                continue

            if session_dict is None:
                record_indexed_file(
                    db, file_path, file_mtime, source_type,
                )
                stats["files_skipped"] += 1
                continue

            if _index_session(db, session_dict, stats, machine_id):
                record_indexed_file(
                    db, file_path, file_mtime, source_type,
                )

    return stats


# ---------------------------------------------------------------------------
# Top-level scan
# ---------------------------------------------------------------------------

def scan_sessions(db: sqlite3.Connection, machine_id: str = "") -> dict:
    """Walk all configured session sources and index new/changed files."""
    total = _empty_stats()

    for source in get_session_sources():
        source_type = source["type"]
        source_path = source["path"]
        log.info("Scanning %s source: %s", source_type, source_path)

        if source_type == "claude_history":
            part = _scan_history_file(db, source_path, machine_id)
        elif source_type == "opencode":
            part = _scan_opencode_db(db, source_path, machine_id)
        else:
            part = scan_source(db, source_type, source_path, machine_id)
        _merge_stats(total, part)

    log.info(
        "Scan complete -- indexed %d, skipped %d, errors %d",
        total["files_indexed"],
        total["files_skipped"],
        total["errors"],
    )
    return total
