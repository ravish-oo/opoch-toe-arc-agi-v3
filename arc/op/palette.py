# arc/op/palette.py
# WO-01: Palette canonicalization (inputs only)
# Implements 02_determinism_addendum.md §1.3

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .hash import hash_bytes
from .bytes import varu


@dataclass
class PaletteRc:
    """
    Palette canonicalization receipt.

    Contract (02_determinism_addendum.md §1.3 lines 59-63):
    - Build over inputs only (train inputs + test input)
    - Sort by descending frequency, ties by ascending color value
    - Record (color, freq) list and mapping

    Contract (docs/common_mistakes.md F1):
    - scope must be "inputs_only" to prevent palette misuse
    """
    palette_hash: str
    palette_freqs: list[tuple[int, int]]  # [(color, freq), ...] sorted
    mapping: list[tuple[int, int]]        # [(orig -> code), ...] sorted by code
    scope: str  # Must be "inputs_only"


def build_palette_canon(
    inputs: list[np.ndarray],
    *,
    forbid_outputs: bool = True
) -> tuple[dict[int, int], dict[int, int], PaletteRc]:
    """
    Build inputs-only palette canonicalization.

    Algorithm (02_determinism_addendum.md §1.3):
    1. Count color frequencies over ALL input grids
    2. Sort by descending frequency; ties by ascending color value
    3. Assign codes 0..k-1 in that order

    Contract (docs/common_mistakes.md F1):
    Palette must be built over inputs-only (forbid_outputs=True).
    This guard prevents accidental inclusion of training outputs.

    Args:
        inputs: list of input grids (train inputs + test input)
        forbid_outputs: must be True (enforces inputs-only scope)

    Returns:
        (map, inv_map, receipt)
        - map: dict[original_color -> canonical_code]
        - inv_map: dict[canonical_code -> original_color]
        - receipt: PaletteRc with hash/freqs/mapping/scope

    Raises:
        ValueError: if forbid_outputs != True
    """
    # F1 guard: prevent palette misuse
    if forbid_outputs is not True:
        raise ValueError(
            "Palette canon must be built over inputs-only (forbid_outputs=True). "
            "This prevents accidental inclusion of training outputs (F1 mistake)."
        )

    # Count frequencies across all inputs
    freqs: dict[int, int] = {}
    for G in inputs:
        # Use int64 to avoid overflow on large grids
        vals, counts = np.unique(G.astype(np.int64, copy=False), return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            freqs[v] = freqs.get(v, 0) + c

    # Sort: descending frequency, then ascending color value (ties)
    # Contract: freq↓, value↑
    items = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))

    # Assign canonical codes 0..k-1
    mapping = {color: idx for idx, (color, _) in enumerate(items)}
    inv_map = {idx: color for idx, (color, _) in enumerate(items)}

    # Build deterministic hash payload
    # Serialize as varints: <color1><freq1><color2><freq2>...
    payload = bytearray()
    for color, freq in items:
        payload += varu(color)
        payload += varu(freq)

    # Create receipt
    rc = PaletteRc(
        palette_hash=hash_bytes(bytes(payload)),
        palette_freqs=items,
        mapping=sorted(mapping.items(), key=lambda kv: kv[1]),  # sorted by code
        scope="inputs_only",  # F1 guard: document scope in receipt
    )

    return mapping, inv_map, rc


def apply_palette_map(G: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """
    Apply palette mapping to grid.

    Args:
        G: input grid
        mapping: dict[original_color -> canonical_code]

    Returns:
        Grid with colors mapped to canonical codes
    """
    # Vectorized lookup
    uniq = np.unique(G)
    lut = np.zeros(int(uniq.max()) + 1, dtype=np.int64)
    for orig, code in mapping.items():
        if orig in uniq:
            lut[orig] = code

    # Fast indexing
    return lut[G.astype(np.int64, copy=False)]


def invert_palette_map(G: np.ndarray, inv_map: dict[int, int]) -> np.ndarray:
    """
    Invert palette mapping (for unpresentation).

    Args:
        G: canonical grid
        inv_map: dict[canonical_code -> original_color]

    Returns:
        Grid with original colors restored
    """
    # Vectorized lookup
    uniq = np.unique(G)
    lut = np.zeros(int(uniq.max()) + 1, dtype=np.int64)
    for code, orig in inv_map.items():
        if code in uniq:
            lut[code] = orig

    # Identity fallback: if code not in inv_map, keep as-is
    # (handles unseen output colors per spec)
    result = G.astype(np.int64, copy=False).copy()
    mask = np.isin(result, list(inv_map.keys()))
    result[mask] = lut[result[mask]]

    return result
