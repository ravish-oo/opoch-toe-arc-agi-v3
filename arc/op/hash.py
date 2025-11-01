# arc/op/hash.py
# WO-00: BLAKE3 hashing helpers
# Implements 02_determinism_addendum.md §0 global invariants (BLAKE3 only)

from __future__ import annotations
from blake3 import blake3
import numpy as np
from .bytes import to_bytes_grid


def hash_bytes(b: bytes) -> str:
    """
    Hash bytes with BLAKE3, return hex digest.

    Contract (02_determinism_addendum.md line 68):
    "Hashes: BLAKE3 only"

    Args:
        b: bytes to hash

    Returns:
        str: hex digest (64 hex chars = 256 bits)
    """
    return blake3(b).hexdigest()


def hash_grid(G: np.ndarray) -> str:
    """
    Hash grid using uint32_le row-major serialization.

    Contract (01_engineering_spec.md line 68):
    "Grid hashing uses uint32_le row-major"

    Contract (02_determinism_addendum.md line 15):
    "Cell encoding for hashes: uint32 little-endian row-major order"

    Args:
        G: numpy array of integer colors

    Returns:
        str: BLAKE3 hex digest of serialized grid
    """
    return hash_bytes(to_bytes_grid(G))
