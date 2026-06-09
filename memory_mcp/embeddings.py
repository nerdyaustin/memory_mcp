"""Lightweight embedding wrapper using fastembed (ONNX-based).

Loads BAAI/bge-small-en-v1.5 (384-dim) once and exposes embed/embed_batch.
Degrades gracefully: if fastembed is not installed, AVAILABLE is False and
all embed calls return None.  Callers must check AVAILABLE before relying
on embeddings.
"""

from __future__ import annotations

import importlib.util
import logging
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Availability probe without importing fastembed. Importing fastembed eagerly
# triggers ONNX runtime initialisation (hundreds of ms to seconds), which would
# block MCP server startup even before the model itself is loaded. The real
# import happens lazily inside Embedder.__init__ on first instantiation.
AVAILABLE = importlib.util.find_spec("fastembed") is not None
if not AVAILABLE:
    log.info("fastembed not installed — semantic search disabled")


# ---------------------------------------------------------------------------
# Serialization helpers (sqlite-vec expects little-endian float32 blobs)
# ---------------------------------------------------------------------------


def serialize_f32(vector: list[float] | np.ndarray) -> bytes:
    """Pack a float vector into a little-endian float32 blob for sqlite-vec."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return struct.pack(f"<{len(vector)}f", *vector)


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------


class Embedder:
    """Thin wrapper around fastembed.TextEmbedding.

    Instantiation loads the model into memory (~130 MB ONNX, first run
    downloads to ``~/.cache/fastembed/``).  After that, ``embed`` is ~2 ms
    per text on Apple Silicon.
    """

    def __init__(self) -> None:
        if not AVAILABLE:
            raise RuntimeError(
                "Cannot create Embedder: fastembed is not installed. "
                "Install with: uv add fastembed"
            )
        # Lazy import: pulls in fastembed + ONNX runtime only when an Embedder
        # is actually constructed, not at module import time. Kept inside
        # __init__ so server.py can import this module cheaply during startup.
        from fastembed import TextEmbedding  # type: ignore[import-untyped]

        log.info("Loading embedding model %s …", MODEL_NAME)
        self._model = TextEmbedding(model_name=MODEL_NAME)
        log.info("Embedding model ready (dim=%d)", EMBEDDING_DIM)

    def embed(self, text: str) -> bytes:
        """Embed a single text and return serialized float32 bytes."""
        # fastembed.embed() returns a generator of numpy arrays.
        vec = next(iter(self._model.embed([text])))
        return serialize_f32(vec)

    def embed_batch(self, texts: list[str]) -> list[bytes]:
        """Embed multiple texts and return list of serialized float32 bytes."""
        return [serialize_f32(v) for v in self._model.embed(texts)]
