"""API key authentication for the sync server.

Expects ``Authorization: Bearer <key>`` on every request except
``/health``.  Validates against a SHA-256 hash stored in the
``users`` table.
"""

from __future__ import annotations

import hashlib
import logging
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


# Public paths that skip auth.
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


async def get_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = None,
) -> str:
    """Return the SHA-256 hash of the request's Bearer token."""
    if request.url.path in _PUBLIC_PATHS:
        return ""  # no auth needed

    token = credentials.credentials if credentials is not None else None
    if not token:
        authorization = request.headers.get("authorization", "")
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value:
            token = value.strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    return _hash_key(token)


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def verify_api_key(
    db: AsyncSession, token_hash: str,
) -> str:
    """Check *token_hash* against stored hashes.  Returns the matching
    user id string on success, raises 401 on failure."""
    if not token_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    from hosted.models import User

    result = await db.execute(
        select(User.id).where(User.api_key_hash == token_hash),
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return str(user)
