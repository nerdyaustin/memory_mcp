"""Minimal HTTP client for the memory_mcp sync server.

Uses stdlib ``urllib`` so sync remains a zero-dependency feature
when not configured.  All I/O runs via ``asyncio.to_thread`` so the
event loop never blocks.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


class SyncClient:
    """HTTP client for the memory_mcp sync server."""

    def __init__(self, api_url: str, api_key: str) -> None:
        self._base = api_url.rstrip("/")
        self._key = api_key

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push(
        self,
        machine_id: str,
        sessions: list[dict],
        memories: list[dict],
    ) -> dict:
        """Push local sessions and memories to the server.

        Returns the server response dict, which includes
        ``server_ts`` and counts of accepted items.
        """
        data = {
            "machine_id": machine_id,
            "sessions": sessions,
            "memories": memories,
        }
        return self._post("/sync/push", data)

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------

    def pull(self, machine_id: str, since: str | None = None) -> dict:
        """Pull sessions and memories from other machines.

        *since* is an ISO-8601 timestamp; the server returns items
        updated after that point.
        """
        params = f"?machine_id={machine_id}"
        if since:
            params += f"&since={since}"
        return self._get("/sync/pull" + params)

    # ------------------------------------------------------------------
    # Machine registration
    # ------------------------------------------------------------------

    def register_machine(self, machine_id: str, hostname: str) -> dict:
        """Register this machine with the sync server."""
        return self._post("/machines/register", {
            "machine_id": machine_id,
            "hostname": hostname,
        })

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Check if the sync server is reachable."""
        return self._get("/health")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, data: dict) -> dict:
        return self._request("POST", path, data)

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _request(
        self, method: str, path: str, data: dict | None = None,
    ) -> dict:
        url = self._base + path
        body = json.dumps(data).encode("utf-8") if data else None
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        req = urllib.request.Request(
            url, data=body, headers=headers, method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            log.warning(
                "Sync server returned %d %s: %s",
                exc.code, exc.reason, detail[:200],
            )
            raise
        except urllib.error.URLError as exc:
            log.warning("Sync server unreachable: %s", exc.reason)
            raise
        except OSError as exc:
            log.warning("Sync request failed: %s", exc)
            raise
