#!/usr/bin/env python3
# arc/op/admit.py
# WO-11A: Admit & Propagate layer (Addendum v1.3)
# Implements admissible-set calculus with least fixed point (Knaster-Tarski)

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from .hash import hash_bytes


@dataclass
class AdmitLayerRc:
    """
    Admit layer receipt.

    Contract (WO-11A + Scope S):
    - name: layer identifier (e.g., "witness", "engine:column_dict", "unanimity")
    - bitmap_hash: BLAKE3 over bitset tensor bytes
    - scope_hash: BLAKE3 over scope tensor bytes
    - support_colors: color universe this layer touched
    - stats: scope_bits, nontrivial_bits, removed_bottom_bits, etc.
    """
    name: str
    bitmap_hash: str
    scope_hash: str
    support_colors: List[int]
    stats: Dict[str, int]


@dataclass
class PropagateRc:
    """
    Fixed-point propagation receipt.

    Contract (WO-11A):
    - passes: number of iterations until convergence
    - shrink_events: total bit changes across all passes
    - shrunk_pixels: approximate pixel count affected
    - per_pass_shrink: list of changes per pass
    - domains_hash: BLAKE3 over final D* tensor
    """
    passes: int
    shrink_events: int
    shrunk_pixels: int
    per_pass_shrink: List[int]
    domains_hash: str


# ============================================================================
# Bitset primitives (frozen by WO-11A Addendum v1.3)
# ============================================================================

def _kwords(num_colors: int) -> int:
    """
    Compute number of uint64 words needed for bitset.

    Contract (WO-11A freeze):
    K = ceil(|C| / 64) = (len(C) + 63) // 64
    """
    return (num_colors + 63) // 64


def _set_bit(word_array: np.ndarray, color_idx: int) -> None:
    """
    Set bit for color_idx in bitset.

    Contract (WO-11A freeze):
    - Bit i represents color C[i]
    - Word: w = i >> 6
    - Bit within word: b = i & 63
    - Little-endian, LSB-first

    Args:
        word_array: 1D array of K uint64 words
        color_idx: index into color universe C
    """
    w = color_idx >> 6
    b = color_idx & 63
    word_array[w] |= np.uint64(1) << np.uint64(b)


def _test_bit(word_array: np.ndarray, color_idx: int) -> bool:
    """
    Test if bit for color_idx is set.

    Args:
        word_array: 1D array of K uint64 words
        color_idx: index into color universe C

    Returns:
        True if bit is set
    """
    w = color_idx >> 6
    b = color_idx & 63
    return ((word_array[w] >> np.uint64(b)) & np.uint64(1)) != 0


def _popcount_bitset(word_array: np.ndarray) -> int:
    """
    Count number of set bits in bitset.

    Args:
        word_array: 1D array of K uint64 words

    Returns:
        Number of set bits
    """
    # Use numpy's binary_repr to count bits (there's no vectorized popcount)
    count = 0
    for word in word_array:
        # Count bits in this word
        count += bin(int(word)).count('1')
    return count


def _bitmap_hash(bitset: np.ndarray) -> str:
    """
    Compute BLAKE3 hash of bitset tensor.

    Contract (WO-11A freeze):
    - Row-major order (r,c)
    - Each uint64 contributes 8 bytes little-endian

    Args:
        bitset: (H, W, K) array of uint64

    Returns:
        BLAKE3 hex digest
    """
    # View as bytes (numpy handles little-endian)
    bytes_view = bitset.view(np.uint8)
    return hash_bytes(bytes(bytes_view))


def _normalize_scope(A: np.ndarray, S: np.ndarray, C: List[int]) -> None:
    """
    Normalize scope: if A[p] admits all colors, set S[p] = 0 (silent).

    Contract (Scope S):
    "Silence ≠ admit-all: if popcount(A[p]) == |C|, set S[p] = 0"

    This ensures:
    - Silent layers (S=0) don't win precedence
    - Admit-all is equivalent to "no constraint" for selection

    Args:
        A: (H, W, K) admits bitset (modified in-place)
        S: (H, W) scope (modified in-place)
        C: color universe

    Modifies:
        S: sets S[p] = 0 where A[p] admits all colors
    """
    H, W, K = A.shape
    num_colors = len(C)

    for r in range(H):
        for c in range(W):
            if _popcount_bitset(A[r, c]) == num_colors:
                S[r, c] = 0  # Force to silent


# ============================================================================
# Color universe
# ============================================================================

def color_universe(
    Xt: np.ndarray,
    train_Xt: List[np.ndarray],
    bottom_color: int = 0
) -> List[int]:
    """
    Return frozen color universe C = sorted unique(int) over Π(inputs only).

    Contract (WO-11A + Scope S):
    - Build from inputs only (train inputs + test input)
    - ALWAYS include bottom_color (default: 0)
    - Sorted ascending
    - Original color IDs (after palette mapping)

    Why include bottom_color:
    Guarantees containment invariant: when all layers are silent at pixel p,
    D*[p] remains full and includes bottom_color, so bottom path never violates
    containment (selected ∈ D*[p]).

    Args:
        Xt: test input in Π frame
        train_Xt: list of training inputs in Π frame
        bottom_color: frozen bottom color (default 0)

    Returns:
        Sorted list of unique color values, always including bottom_color
    """
    colors = set([int(bottom_color)])  # Force bottom into C

    # Collect from test input
    colors.update(int(c) for c in np.unique(Xt))

    # Collect from training inputs
    for X in train_Xt:
        colors.update(int(c) for c in np.unique(X))

    return sorted(colors)


# ============================================================================
# Domain initialization
# ============================================================================

def empty_domains(H: int, W: int, C: List[int]) -> np.ndarray:
    """
    Return domains D0[p] = full bitmask over |C| (uint64 blocks).

    Contract (WO-11A freeze):
    - Shape: (H, W, K) where K = ceil(|C| / 64)
    - All bits set to 1 initially (all colors allowed)
    - Unused high bits in last word masked to 0

    Args:
        H: grid height
        W: grid width
        C: color universe (sorted list)

    Returns:
        (H, W, K) array of uint64, all colors admitted initially
    """
    K = _kwords(len(C))

    # Initialize all bits to 1
    D0 = np.empty((H, W, K), dtype=np.uint64)
    D0.fill(np.uint64(0xFFFFFFFFFFFFFFFF))

    # Mask unused high bits in last word
    num_colors = len(C)
    unused = (K * 64 - num_colors)
    if unused > 0:
        # Create mask: set lower (64 - unused) bits to 1
        mask = (np.uint64(1) << np.uint64(64 - unused)) - np.uint64(1)
        D0[..., K - 1] &= mask

    return D0


# ============================================================================
# Admit from witness
# ============================================================================

def admit_from_witness(
    Xt: np.ndarray,
    witness_rc: dict,
    C: List[int]
) -> Tuple[np.ndarray, np.ndarray, AdmitLayerRc]:
    """
    Build A^w[p] bitsets and S^w[p] scope from geometric witness.

    Contract (WO-11A + Scope S):
    - Copy admits: for each φ piece, admit source color at target pixel
    - Recolor admits: if σ present, map through permutation
    - Summary/failed witness: emit all-ones (no constraints), S=0 (silent)
    - Normalization: if A[p] admits all colors, S[p]=0

    Args:
        Xt: test input in Π frame (H, W)
        witness_rc: witness receipt with per_train[], intersection
        C: color universe

    Returns:
        (A_w, S_w, receipt) where:
        - A_w is (H, W, K) bitset admits
        - S_w is (H, W) uint8 scope (1=scoped, 0=silent)
    """
    H, W = Xt.shape
    K = _kwords(len(C))

    # Build color index lookup
    color_to_idx = {c: i for i, c in enumerate(C)}

    # Initialize: all-ones (no constraints by default), scope=0 (silent)
    A_w = np.empty((H, W, K), dtype=np.uint64)
    A_w.fill(np.uint64(0xFFFFFFFFFFFFFFFF))
    S_w = np.zeros((H, W), dtype=np.uint8)  # Silent by default

    # Mask unused bits
    unused = (K * 64 - len(C))
    if unused > 0:
        mask = (np.uint64(1) << np.uint64(64 - unused)) - np.uint64(1)
        A_w[..., K - 1] &= mask

    # Check if witness produced a singleton law
    # Contract (WO-04 updated): intersection.status can be:
    # - "singleton": unique geometric law found → emit copy+recolor admits
    # - "contradictory": no geometric law found → emit all-ones (no constraints), S=0
    # - "underdetermined": multiple laws fit → emit all-ones (no constraints), S=0
    intersection = witness_rc.get("intersection", {})
    if intersection.get("status") != "singleton":
        # No constraint from witness - silent (S=0)
        # This covers: contradictory, underdetermined, or failed witness
        stats = {
            "set_bits": H * W * len(C),
            "scope_bits": 0,
            "nontrivial_bits": 0,
            "removed_bottom_bits": 0,
            "pieces": 0,
            "sigma_moved": 0
        }
        return A_w, S_w, AdmitLayerRc(
            name="witness",
            bitmap_hash=_bitmap_hash(A_w),
            scope_hash=hash_bytes(S_w.tobytes()),
            support_colors=C.copy(),
            stats=stats
        )

    # Singleton geometric witness: emit copy admits + recolor admits (WO-10Z)
    per_train = witness_rc.get("per_train", [])
    # Start with empty admits (no colors allowed), scope=0 (will set to 1 where constrained)
    A_w = np.zeros((H, W, K), dtype=np.uint64)
    S_w = np.zeros((H, W), dtype=np.uint8)

    # Extract phi pieces and sigma from first training (all should be same after intersection)
    first_train = per_train[0]
    phi = first_train.get("phi", {})
    pieces = phi.get("pieces", [])
    sigma = first_train.get("sigma", {})
    domain_colors = sigma.get("domain_colors", [])
    lehmer = sigma.get("lehmer", [])

    # Import helpers
    from .d4 import apply_pose_to_pixel
    from .witness import _lehmer_to_perm

    # Build permutation from lehmer code
    if lehmer and domain_colors:
        perm = _lehmer_to_perm(lehmer)
        # Build color mapping: domain_colors[i] → domain_colors[perm[i]]
        sigma_map = {domain_colors[i]: domain_colors[perm[i]] for i in range(len(domain_colors))}
    else:
        sigma_map = {}

    set_bits = 0
    pieces_count = len(pieces)

    # For each phi piece, emit copy admits
    for piece in pieces:
        # Extract piece parameters
        pose_id = piece.get("pose_id", 0)
        dr = piece.get("dr", 0)
        dc = piece.get("dc", 0)
        bbox_h = piece.get("bbox_h", 0)
        bbox_w = piece.get("bbox_w", 0)
        src_r0 = piece.get("src_r0", 0)
        src_c0 = piece.get("src_c0", 0)
        target_r0 = piece.get("target_r0", 0)
        target_c0 = piece.get("target_c0", 0)

        # Iterate source bbox
        for sr in range(bbox_h):
            for sc in range(bbox_w):
                # Source pixel in Xt
                src_r = src_r0 + sr
                src_c = src_c0 + sc

                # Bounds check
                if not (0 <= src_r < H and 0 <= src_c < W):
                    continue

                # Read source color
                color = int(Xt[src_r, src_c])

                # Skip if color not in universe
                if color not in color_to_idx:
                    continue

                # Apply D4 transformation to get target offset
                tr_offset, tc_offset = apply_pose_to_pixel(sr, sc, bbox_h, bbox_w, pose_id)

                # Target pixel in Xt
                target_r = target_r0 + tr_offset
                target_c = target_c0 + tc_offset

                # Bounds check
                if not (0 <= target_r < H and 0 <= target_c < W):
                    continue

                # Apply σ recolor if present
                if color in sigma_map:
                    admitted_color = sigma_map[color]
                else:
                    admitted_color = color

                # Skip if recolored color not in universe
                if admitted_color not in color_to_idx:
                    continue

                # Admit the color at target pixel
                color_idx = color_to_idx[admitted_color]
                _set_bit(A_w[target_r, target_c], color_idx)
                S_w[target_r, target_c] = 1  # Mark as scoped
                set_bits += 1

    # Normalize: if A[p] admits all colors, set S[p] = 0 (silent)
    _normalize_scope(A_w, S_w, C)

    # Compute stats
    bottom_idx = color_to_idx.get(0, -1)
    removed_bottom = 0
    nontrivial = 0
    scope_bits = int(S_w.sum())

    for r in range(H):
        for c in range(W):
            if S_w[r, c]:
                # Check if removed bottom bit
                if bottom_idx >= 0 and not _test_bit(A_w[r, c], bottom_idx):
                    removed_bottom += 1
                # Check if nontrivial (scoped and not admit-all)
                if _popcount_bitset(A_w[r, c]) < len(C):
                    nontrivial += 1

    stats = {
        "set_bits": int(set_bits),
        "scope_bits": scope_bits,
        "nontrivial_bits": nontrivial,
        "removed_bottom_bits": removed_bottom,
        "pieces": pieces_count,
        "sigma_moved": len(sigma_map)
    }

    return A_w, S_w, AdmitLayerRc(
        name="witness",
        bitmap_hash=_bitmap_hash(A_w),
        scope_hash=hash_bytes(S_w.tobytes()),
        support_colors=C.copy(),
        stats=stats
    )


# ============================================================================
# Admit from engine
# ============================================================================

def admit_from_engine(
    Xt: np.ndarray,
    engine_apply_rc: Optional[dict],
    C: List[int]
) -> Tuple[np.ndarray, np.ndarray, AdmitLayerRc]:
    """
    Build A^e[p] bitsets and S^e[p] scope from engine.

    Contract (WO-11A + WO-10Z + Scope S):
    - Engine emits singleton admits per pixel where it painted
    - Extract Yt from engine_apply_rc and build singleton admits
    - If engine didn't run or failed: return all-ones (no constraint), S=0 (silent)

    Args:
        Xt: test input in Π frame (H, W)
        engine_apply_rc: engine application receipt with 'Yt' field, or None
        C: color universe

    Returns:
        (A_e, S_e, receipt) where:
        - A_e is (H, W, K) bitset admits
        - S_e is (H, W) uint8 scope (1=scoped, 0=silent)
    """
    H, W = Xt.shape
    K = _kwords(len(C))

    # Build color→index lookup
    color_to_idx = {c: i for i, c in enumerate(C)}

    # Check if engine ran successfully
    engine_name = "none"
    Yt = None

    if engine_apply_rc is not None:
        engine_name = engine_apply_rc.get("engine", "unknown")

        # Try to extract painted grid (different engines use different field names)
        if "Yt" in engine_apply_rc:
            Yt = engine_apply_rc["Yt"]
        elif hasattr(engine_apply_rc, "Yt"):
            # Handle dataclass receipts
            Yt = engine_apply_rc.Yt
        elif "ok" in engine_apply_rc and engine_apply_rc.get("ok"):
            # Engine succeeded but no Yt field (shouldn't happen)
            pass

    if Yt is None:
        # Engine didn't run or failed: return all-ones (no constraints), S=0 (silent)
        A_e = np.empty((H, W, K), dtype=np.uint64)
        A_e.fill(np.uint64(0xFFFFFFFFFFFFFFFF))
        S_e = np.zeros((H, W), dtype=np.uint8)  # Silent

        # Mask unused bits
        unused = (K * 64 - len(C))
        if unused > 0:
            mask = (np.uint64(1) << np.uint64(64 - unused)) - np.uint64(1)
            A_e[..., K - 1] &= mask

        stats = {
            "set_bits": H * W * len(C),
            "scope_bits": 0,
            "nontrivial_bits": 0,
            "removed_bottom_bits": 0,
            "covered_pixels": 0,
            "singleton_admits": 0
        }
    else:
        # Engine succeeded: build singleton admits from Yt
        A_e = np.zeros((H, W, K), dtype=np.uint64)
        S_e = np.zeros((H, W), dtype=np.uint8)
        singleton_count = 0
        covered = 0

        for r in range(H):
            for c in range(W):
                color = int(Yt[r, c])

                if color in color_to_idx:
                    # Admit singleton {color} at this pixel
                    color_idx = color_to_idx[color]
                    _set_bit(A_e[r, c], color_idx)
                    S_e[r, c] = 1  # Mark as scoped
                    singleton_count += 1
                    covered += 1
                else:
                    # Color not in universe: leave empty (no constraint)
                    # This shouldn't happen if engine is correct
                    pass

        # Normalize: if A[p] admits all colors, set S[p] = 0 (silent)
        _normalize_scope(A_e, S_e, C)

        # Compute stats
        bottom_idx = color_to_idx.get(0, -1)
        removed_bottom = 0
        nontrivial = 0
        scope_bits = int(S_e.sum())

        for r in range(H):
            for c in range(W):
                if S_e[r, c]:
                    # Check if removed bottom bit
                    if bottom_idx >= 0 and not _test_bit(A_e[r, c], bottom_idx):
                        removed_bottom += 1
                    # Check if nontrivial (scoped and not admit-all)
                    if _popcount_bitset(A_e[r, c]) < len(C):
                        nontrivial += 1

        stats = {
            "set_bits": singleton_count,
            "scope_bits": scope_bits,
            "nontrivial_bits": nontrivial,
            "removed_bottom_bits": removed_bottom,
            "covered_pixels": covered,
            "singleton_admits": singleton_count
        }

    return A_e, S_e, AdmitLayerRc(
        name=f"engine:{engine_name}",
        bitmap_hash=_bitmap_hash(A_e),
        scope_hash=hash_bytes(S_e.tobytes()),
        support_colors=C.copy(),
        stats=stats
    )


# ============================================================================
# Admit from unanimity
# ============================================================================

def admit_from_unanimity(
    truth_blocks: np.ndarray,
    uni_rc: dict,
    C: List[int]
) -> Tuple[np.ndarray, np.ndarray, AdmitLayerRc]:
    """
    Build A^u[p] bitsets and S^u[p] scope from unanimity.

    Contract (WO-11A + Scope S):
    - For unanimous block B with color u: admit {u} for all p ∈ B, S=1
    - For non-unanimous blocks: admit all colors (no restriction), S=0

    Args:
        truth_blocks: (H, W) array of block IDs
        uni_rc: unanimity receipt with blocks[]
        C: color universe

    Returns:
        (A_u, S_u, receipt) where:
        - A_u is (H, W, K) bitset admits
        - S_u is (H, W) uint8 scope (1=scoped, 0=silent)
    """
    H, W = truth_blocks.shape
    K = _kwords(len(C))

    # Build color index lookup
    color_to_idx = {c: i for i, c in enumerate(C)}

    # Initialize: all-ones (no constraints), S=0 (silent)
    A_u = np.empty((H, W, K), dtype=np.uint64)
    A_u.fill(np.uint64(0xFFFFFFFFFFFFFFFF))
    S_u = np.zeros((H, W), dtype=np.uint8)

    # Mask unused bits
    unused = (K * 64 - len(C))
    if unused > 0:
        mask = (np.uint64(1) << np.uint64(64 - unused)) - np.uint64(1)
        A_u[..., K - 1] &= mask

    # Apply unanimity constraints
    unanimous_count = 0
    blocks = uni_rc.get("blocks", [])

    for block_info in blocks:
        block_id = block_info.get("block_id")
        color = block_info.get("color")

        if color is None:
            # Non-unanimous block: no constraint
            continue

        if color not in color_to_idx:
            # Color not in universe (shouldn't happen)
            continue

        # Find all pixels in this block
        block_mask = (truth_blocks == block_id)

        # For these pixels, admit only the unanimous color
        # Set all bits to 0, then set only the color bit
        A_u[block_mask] = np.uint64(0)

        color_idx = color_to_idx[color]
        w = color_idx >> 6
        b = color_idx & 63
        A_u[block_mask, w] = np.uint64(1) << np.uint64(b)

        # Mark these pixels as scoped
        S_u[block_mask] = 1

        unanimous_count += 1

    # Normalize: if A[p] admits all colors, set S[p] = 0 (silent)
    _normalize_scope(A_u, S_u, C)

    # Compute stats
    bottom_idx = color_to_idx.get(0, -1)
    removed_bottom = 0
    nontrivial = 0
    scope_bits = int(S_u.sum())

    for r in range(H):
        for c in range(W):
            if S_u[r, c]:
                # Check if removed bottom bit
                if bottom_idx >= 0 and not _test_bit(A_u[r, c], bottom_idx):
                    removed_bottom += 1
                # Check if nontrivial (scoped and not admit-all)
                if _popcount_bitset(A_u[r, c]) < len(C):
                    nontrivial += 1

    stats = {
        "unanimous_blocks": unanimous_count,
        "set_bits": int(np.sum([_popcount_bitset(A_u[r, c]) for r in range(H) for c in range(W)])),
        "scope_bits": scope_bits,
        "nontrivial_bits": nontrivial,
        "removed_bottom_bits": removed_bottom
    }

    return A_u, S_u, AdmitLayerRc(
        name="unanimity",
        bitmap_hash=_bitmap_hash(A_u),
        scope_hash=hash_bytes(S_u.tobytes()),
        support_colors=C.copy(),
        stats=stats
    )


# ============================================================================
# Fixed-point propagation
# ============================================================================

def propagate_fixed_point(
    domains: np.ndarray,
    layers: List[Tuple[np.ndarray, np.ndarray, str]],
    C: List[int]
) -> Tuple[np.ndarray, PropagateRc]:
    """
    Monotone lfp with scope-gated intersection: D ← D ∩ A only where S[p]=1.

    Contract (WO-11A + Scope S):
    - Intersection order: frozen by caller (witness → engines → unanimity)
    - Scope-gated: only intersect D[p] &= A[p] where S[p]=1
    - Invariant: S[p]=0 ⇒ popcount(A[p]) = |C| (silent implies admit-all)
    - Terminates when no changes (monotone on finite lattice)
    - Returns (D*, receipt) with convergence stats

    Args:
        domains: initial D0, shape (H, W, K) uint64
        layers: list of (A, S, name) tuples where:
                - A: admit bitmap (H, W, K) uint64
                - S: scope mask (H, W) uint8
                - name: layer name (str)
        C: color universe (for invariant checking)

    Returns:
        (D*, PropagateRc) where D* is fixed point
    """
    D = domains.copy()
    H, W, K = D.shape
    num_colors = len(C)
    passes = 0
    total_shrunk = 0
    per_pass = []

    while True:
        passes += 1
        before = D.copy()

        # Scope-gated intersection over all layers
        for (A, S, name) in layers:
            if A is None or S is None:
                # Layer disabled/not present
                continue

            # Apply scope-gated intersection pixel by pixel
            for r in range(H):
                for c in range(W):
                    if S[r, c] == 1:
                        # Scoped: intersect this pixel's bitset
                        D[r, c] &= A[r, c]
                    # else: silent (S=0), skip intersection (preserves D[r,c])

        # Count changes (byte-wise comparison)
        changes = np.count_nonzero(before != D)
        total_shrunk += changes
        per_pass.append(int(changes))

        # Converged?
        if changes == 0:
            break

        # Safety: prevent infinite loop (shouldn't happen with monotone intersection)
        if passes > 100:
            raise RuntimeError(f"propagate_fixed_point: exceeded 100 passes (non-termination)")

    # Approximate shrunk pixels (changes / K)
    shrunk_pixels = total_shrunk // K if K > 0 else 0

    return D, PropagateRc(
        passes=passes,
        shrink_events=total_shrunk,
        shrunk_pixels=shrunk_pixels,
        per_pass_shrink=per_pass,
        domains_hash=_bitmap_hash(D)
    )
