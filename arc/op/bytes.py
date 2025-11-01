# arc/op/bytes.py
# WO-00: Canonical encodings (uint32_le, ZigZag LEB128, varints)
# Implements 02_determinism_addendum.md §0 global invariants

from __future__ import annotations
import numpy as np


def to_bytes_grid(G: np.ndarray) -> bytes:
    """
    Encode H×W grid as uint32 little-endian row-major bytes.

    Contract (02_determinism_addendum.md line 15):
    "encode each color as uint32 little-endian (uint32_le) in row-major order"

    Args:
        G: numpy array of integer colors

    Returns:
        bytes: uint32_le serialization

    Raises:
        TypeError: if G is not integer dtype
    """
    if G.dtype.kind not in "iu":
        raise TypeError("Grid must be integer dtype")

    # Cast to uint32, keep row-major order (C-contiguous)
    g32 = G.astype(np.uint32, copy=False)

    # Ensure little-endian regardless of platform
    if g32.dtype.byteorder == '>':
        g32 = g32.byteswap().newbyteorder('<')
    elif g32.dtype.byteorder not in ('<', '|', '='):
        g32 = g32.astype(np.dtype('<u4'))

    return g32.tobytes(order="C")


def _zigzag(i: int) -> int:
    """
    ZigZag encoding: map signed int to unsigned.

    Contract (02_determinism_addendum.md line 16):
    "signed integers are ZigZag-encoded LEB128 varints"

    ZigZag mapping:
      0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, ...
    """
    return (i << 1) ^ (i >> 63) if i < 0 else (i << 1)


def zigzag_encode(i: int) -> bytes:
    """
    Encode signed integer as ZigZag LEB128 varint.

    Args:
        i: signed integer

    Returns:
        bytes: ZigZag-encoded LEB128 varint
    """
    return varu(_zigzag(i))


def varu(n: int) -> bytes:
    """
    Encode unsigned integer as LEB128 varint.

    Contract (02_determinism_addendum.md line 16):
    "Unsigned integers use plain LEB128 varints"

    Args:
        n: unsigned integer (must be >= 0)

    Returns:
        bytes: LEB128 varint

    Raises:
        ValueError: if n < 0
        OverflowError: if n too large for LEB128
    """
    if n < 0:
        raise ValueError("varu expects unsigned (n >= 0)")

    # Safety check for overflow (LEB128 can handle arbitrarily large, but bound it)
    if n >= (1 << 63):
        raise OverflowError(f"Integer {n} too large for safe LEB128 encoding")

    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def frame_params(*ints: int, signed: bool = False) -> bytes:
    """
    Frame parameter list as <count><p1>...<pk>.

    Contract (02_determinism_addendum.md lines 31-42):
    "Param serialization: big-endian varint per integer"
    (Note: "big-endian" here means byte order within multi-byte varints,
     which LEB128 handles naturally; we use LEB128 as specified)

    Args:
        *ints: parameters to encode
        signed: if True, use ZigZag encoding; else plain LEB128

    Returns:
        bytes: framed parameter list
    """
    out = bytearray()
    out += varu(len(ints))
    for v in ints:
        out += zigzag_encode(v) if signed else varu(v)
    return bytes(out)


def from_bytes_grid(b: bytes, shape: tuple[int, int]) -> np.ndarray:
    """
    Decode uint32_le row-major bytes to H×W grid.

    Used only in property tests; not required by operator.

    Args:
        b: uint32_le serialized bytes
        shape: (H, W) dimensions

    Returns:
        np.ndarray: reconstructed grid
    """
    H, W = shape
    expected_len = H * W * 4  # 4 bytes per uint32
    if len(b) != expected_len:
        raise ValueError(f"Expected {expected_len} bytes for {shape}, got {len(b)}")

    # Reconstruct as little-endian uint32
    g32 = np.frombuffer(b, dtype='<u4').reshape(H, W)
    return g32.astype(np.int64)
