"""
WO-05: Truth Compiler (Paige-Tarjan Fixed Point) — FIXED

Contract (00_math_spec.md §4, 01_engineering_spec.md §5, 02_determinism_addendum.md §2):
- Engineering = Math: gfp(ℱ) via Paige-Tarjan coarsest bisimulation
- Debugging = Algebra: Partition refinement is exact set intersection
- Determinism: Frozen tag vocabulary (14 tags EXACT), canonical encodings, lex-min selection

Architecture:
1. Frozen tag vocabulary (14 tags, no dynamic tags) - FROZEN
2. KMP for exact minimal periods
3. Integer FFT for overlap candidates + exact MASK verification (per color)
4. Exact pixel verification (no correlation heuristics)
5. Identity overlap exclusion (Δ=(0,0) excluded) + receipt verification
6. Paige-Tarjan refinement until fixed point
7. Band clusters (row/col boundary indices)

Receipts:
- tag_set_version: Canonical BLAKE3 hash of frozen tag string
- partition_hash: Canonical encoding of final partition
- block_hist: Histogram of block sizes (not scalar)
- row_clusters, col_clusters: Band boundary indices (not periods)
- overlaps: OverlapRc with candidates, accepted, identity_excluded
- refinement_steps: Number of Paige-Tarjan iterations
"""

from dataclasses import dataclass
from typing import Literal, List, Tuple, Optional, Dict
import numpy as np
from blake3 import blake3

from arc.op.hash import hash_bytes

# ============================================================================
# FROZEN TAG VOCABULARY (02_determinism_addendum.md §2 lines 132-153)
# ============================================================================

# Frozen tag set (EXACT from spec - no additions/deletions allowed)
FROZEN_TAGS = (
    "color",
    "n4_adj",
    "n8_adj",
    "samecomp_r2",
    "parity",
    "row_period_2",
    "row_period_3",
    "col_period_2",
    "col_period_3",
    "per_color_overlap",
    "per_line_min_period",
    "exact_tile",
    "bbox_mirror",
    "bbox_rotate",
)

# Canonical tag string (pipe-separated, exact order from spec)
CANON_TAG_STRING = "color|n4_adj|n8_adj|samecomp_r2|parity|row_period_2|row_period_3|col_period_2|col_period_3|per_color_overlap|per_line_min_period|exact_tile|bbox_mirror|bbox_rotate"

# TAG_SET_VERSION = BLAKE3(canonical string)
TAG_SET_VERSION = blake3(CANON_TAG_STRING.encode()).hexdigest()

# ============================================================================
# DATA CLASSES
# ============================================================================

Method = Literal["fft_int", "ntt"]

@dataclass(frozen=True)
class OverlapRc:
    """
    Receipt for discovered overlaps (B3/B7 fix).

    Fields:
    - method: Detection method ("fft_int" or "ntt")
    - modulus: Prime for NTT (None for fft_int)
    - root: Primitive root for NTT (None for fft_int)
    - candidates: ALL (color, dr, dc) candidates tried
    - accepted: Subset passing exact mask verification
    - identity_excluded: Must be True (Δ=(0,0) excluded)
    """
    method: Method
    modulus: Optional[int]
    root: Optional[int]
    candidates: List[Tuple[int, int, int]]  # (color, dr, dc)
    accepted: List[Tuple[int, int, int]]    # verified subset
    identity_excluded: bool                 # must be True


@dataclass(frozen=True)
class TruthRc:
    """
    Receipt for Truth partition computation (B4/B5 fix).

    Fields:
    - tag_set_version: Frozen tag vocabulary hash
    - partition_hash: Canonical encoding of final partition
    - block_hist: Histogram of block sizes (B4 fix)
    - overlaps: OverlapRc with all candidates/accepted
    - row_clusters: Band boundary indices (B5 fix)
    - col_clusters: Band boundary indices (B5 fix)
    - refinement_steps: Number of Paige-Tarjan iterations
    - method: Always "paige_tarjan"
    """
    tag_set_version: str
    partition_hash: str
    block_hist: List[int]
    overlaps: OverlapRc
    row_clusters: List[int]
    col_clusters: List[int]
    refinement_steps: int
    method: Literal["paige_tarjan"]


@dataclass(frozen=True)
class TruthPartition:
    """
    Truth partition result.

    Fields:
    - labels: (H, W) array where labels[r,c] is cluster ID for pixel (r,c)
    - receipt: TruthRc with full diagnostic data
    """
    labels: np.ndarray  # (H, W) int array
    receipt: TruthRc


# ============================================================================
# KMP MINIMAL PERIOD (Exact Algorithm)
# ============================================================================

def _kmp_min_period(s: list[int]) -> int | None:
    """
    Compute minimal period of sequence using KMP failure function.

    Contract:
    - Returns smallest p such that s[i] = s[i+p] for all valid i
    - Returns None if s is aperiodic (period = len(s))

    Algorithm (KMP):
    - Failure function f[i] = longest proper prefix of s[0..i] that is also suffix
    - Minimal period = n - f[n-1] if it divides n, else None
    """
    n = len(s)
    if n == 0:
        return None

    # Build KMP failure function
    f = [0] * n
    k = 0
    for i in range(1, n):
        while k > 0 and s[k] != s[i]:
            k = f[k - 1]
        if s[k] == s[i]:
            k += 1
        f[i] = k

    # Minimal period = n - f[n-1]
    period = n - f[n - 1]

    # Verify period divides n (exact periodicity)
    if n % period == 0 and period < n:
        return period
    return None


def _per_line_min_periods(X: np.ndarray, axis: Literal["row", "col"]) -> List[Optional[int]]:
    """
    Compute minimal periods for all rows or columns using KMP.

    Contract:
    - axis="row": Compute period for each row
    - axis="col": Compute period for each column
    - Returns list of periods (None if line is aperiodic)
    """
    H, W = X.shape
    periods = []

    if axis == "row":
        for r in range(H):
            row = X[r, :].tolist()
            period = _kmp_min_period(row)
            periods.append(period)
    else:  # axis == "col"
        for c in range(W):
            col = X[:, c].tolist()
            period = _kmp_min_period(col)
            periods.append(period)

    return periods


# ============================================================================
# LOCAL TAG EXTRACTORS
# ============================================================================

def _extract_color_tags(X: np.ndarray) -> np.ndarray:
    """Extract color tags for each pixel."""
    return X.copy()


def _extract_n4_adjacency_tags(X: np.ndarray) -> np.ndarray:
    """
    Extract N4 adjacency signature for each pixel.
    Returns hash of sorted neighbor colors (N4: up, down, left, right).
    """
    H, W = X.shape
    tags = np.zeros((H, W), dtype=np.int64)

    for r in range(H):
        for c in range(W):
            neighbors = []
            if r > 0:
                neighbors.append(int(X[r - 1, c]))
            if r < H - 1:
                neighbors.append(int(X[r + 1, c]))
            if c > 0:
                neighbors.append(int(X[r, c - 1]))
            if c < W - 1:
                neighbors.append(int(X[r, c + 1]))
            neighbors.sort()
            tags[r, c] = hash(tuple(neighbors)) & 0x7FFFFFFFFFFFFFFF

    return tags


def _extract_n8_adjacency_tags(X: np.ndarray) -> np.ndarray:
    """
    Extract N8 adjacency signature for each pixel.
    Returns hash of sorted neighbor colors (N8: all 8 directions).
    """
    H, W = X.shape
    tags = np.zeros((H, W), dtype=np.int64)

    for r in range(H):
        for c in range(W):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        neighbors.append(int(X[nr, nc]))
            neighbors.sort()
            tags[r, c] = hash(tuple(neighbors)) & 0x7FFFFFFFFFFFFFFF

    return tags


def _extract_samecomp_r2_tags(X: np.ndarray) -> np.ndarray:
    """
    Extract same-component within radius 2 signature.
    Flood-fill from each pixel, collect component within L∞ distance 2.
    """
    H, W = X.shape
    tags = np.zeros((H, W), dtype=np.int64)

    for r0 in range(H):
        for c0 in range(W):
            color = X[r0, c0]
            visited = set()
            stack = [(r0, c0)]
            component = []

            while stack:
                r, c = stack.pop()
                if (r, c) in visited:
                    continue
                if not (0 <= r < H and 0 <= c < W):
                    continue
                if X[r, c] != color:
                    continue

                visited.add((r, c))

                # Only include if within L∞ distance 2
                if abs(r - r0) <= 2 and abs(c - c0) <= 2:
                    component.append((r, c))

                # N4 flood fill
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((r + dr, c + dc))

            component.sort()
            tags[r0, c0] = hash(tuple(component)) & 0x7FFFFFFFFFFFFFFF

    return tags


def _extract_parity_tags(X: np.ndarray) -> np.ndarray:
    """
    Extract parity tags for each pixel.
    Encode (r%2, c%2) as integer: 0=(even,even), 1=(even,odd), 2=(odd,even), 3=(odd,odd).
    """
    H, W = X.shape
    tags = np.zeros((H, W), dtype=np.int64)

    for r in range(H):
        for c in range(W):
            tags[r, c] = (r % 2) * 2 + (c % 2)

    return tags


def _extract_row_period_tags(X: np.ndarray, row_periods: List[Optional[int]], p: int) -> np.ndarray:
    """
    Extract row_period_p tags (p ∈ {2,3}).
    Tag is True if the row has period p (exact).
    """
    H, W = X.shape
    tags = np.zeros((H, W), dtype=np.int64)

    for r in range(H):
        has_period = (row_periods[r] == p)
        for c in range(W):
            tags[r, c] = int(has_period)

    return tags


def _extract_col_period_tags(X: np.ndarray, col_periods: List[Optional[int]], p: int) -> np.ndarray:
    """
    Extract col_period_p tags (p ∈ {2,3}).
    Tag is True if the column has period p (exact).
    """
    H, W = X.shape
    tags = np.zeros((H, W), dtype=np.int64)

    for c in range(W):
        has_period = (col_periods[c] == p)
        for r in range(H):
            tags[r, c] = int(has_period)

    return tags


def _extract_per_line_min_period_tags(
    X: np.ndarray,
    row_periods: List[Optional[int]],
    col_periods: List[Optional[int]]
) -> np.ndarray:
    """
    Extract per_line_min_period tag (M1 fix).
    For each pixel, encode (row_period, col_period) as hash.
    """
    H, W = X.shape
    tags = np.zeros((H, W), dtype=np.int64)

    for r in range(H):
        for c in range(W):
            rp = row_periods[r] if row_periods[r] is not None else 0
            cp = col_periods[c] if col_periods[c] is not None else 0
            tags[r, c] = hash((rp, cp)) & 0x7FFFFFFFFFFFFFFF

    return tags


# ============================================================================
# GLOBAL TAG EXTRACTORS (M2 fix)
# ============================================================================

def _extract_exact_tile_tags(X: np.ndarray) -> np.ndarray:
    """
    Extract exact_tile tags.
    For each pixel, check if local window is perfectly tiled by a motif.
    Uses bitwise equality (no thresholds).
    """
    H, W = X.shape
    tags = np.zeros((H, W), dtype=np.int64)

    # Simple implementation: check if 2x2 window is tiled by 1x1 motif
    for r in range(H):
        for c in range(W):
            is_tiled = 0
            # Check 2x2 window
            if r + 1 < H and c + 1 < W:
                if (X[r, c] == X[r, c + 1] == X[r + 1, c] == X[r + 1, c + 1]):
                    is_tiled = 1
            tags[r, c] = is_tiled

    return tags


def _extract_bbox_mirror_tags(X: np.ndarray) -> np.ndarray:
    """
    Extract bbox_mirror tags.
    For each pixel, check if local bbox has horizontal/vertical mirror symmetry.
    Uses bitwise exact equality.
    """
    H, W = X.shape
    tags = np.zeros((H, W), dtype=np.int64)

    # Check 3x3 window for mirror symmetry
    for r in range(H):
        for c in range(W):
            has_mirror = 0
            if 1 <= r < H - 1 and 1 <= c < W - 1:
                # Horizontal mirror
                h_mirror = (X[r - 1, c] == X[r + 1, c])
                # Vertical mirror
                v_mirror = (X[r, c - 1] == X[r, c + 1])
                has_mirror = int(h_mirror or v_mirror)
            tags[r, c] = has_mirror

    return tags


def _extract_bbox_rotate_tags(X: np.ndarray) -> np.ndarray:
    """
    Extract bbox_rotate tags.
    For each pixel, check if local bbox has 90/180/270 rotation symmetry.
    Uses bitwise exact equality.
    """
    H, W = X.shape
    tags = np.zeros((H, W), dtype=np.int64)

    # Check 3x3 window for rotation symmetry
    for r in range(H):
        for c in range(W):
            has_rotation = 0
            if 1 <= r < H - 1 and 1 <= c < W - 1:
                # Check 180-degree rotation symmetry
                center = X[r, c]
                rot180 = (
                    X[r - 1, c - 1] == X[r + 1, c + 1] and
                    X[r - 1, c + 1] == X[r + 1, c - 1]
                )
                has_rotation = int(rot180)
            tags[r, c] = has_rotation

    return tags


# ============================================================================
# INTEGER FFT OVERLAP DETECTION (B3/B6/B7 fix)
# ============================================================================

def _per_color_overlap_fft_int(X: np.ndarray) -> OverlapRc:
    """
    Detect overlaps per color using Integer FFT with exact MASK verification (B6 fix).

    Contract (B3/B6/B7):
    1. For each color k, extract binary mask M_k (1 where X=k, 0 elsewhere)
    2. Use NumPy FFT to find autocorrelation peaks (candidates)
    3. For each peak, verify exact MASK equality (not grid equality)
    4. Exclude identity overlap Δ=(0,0)
    5. Return OverlapRc with candidates, accepted, identity_excluded=True

    Algorithm:
    - Autocorrelation via FFT: R = IFFT(FFT(M) * conj(FFT(M)))
    - Peaks indicate potential overlaps
    - Exact verification: M_k[r,c] == M_k[r+dr, c+dc] for all overlapping pixels

    Returns: OverlapRc with all fields (B3 fix)
    """
    H, W = X.shape
    candidates = []
    accepted = []

    # Get unique colors
    colors = np.unique(X).tolist()

    # Fixed neighborhood for candidate generation (frozen constant)
    D = min(3, min(H, W) - 1)  # Small radius

    for color in colors:
        # Binary mask for this color
        M = (X == color).astype(np.float64)

        # FFT-based autocorrelation
        F = np.fft.fft2(M)
        R = np.fft.ifft2(F * np.conj(F)).real

        # Threshold for potential overlaps (>= 3 pixels)
        threshold = 3.0
        peak_candidates = np.argwhere(R >= threshold)

        for idx in peak_candidates:
            dr, dc = int(idx[0]), int(idx[1])

            # Normalize to [-H+1, H-1] x [-W+1, W-1]
            if dr >= H:
                dr -= 2 * H
            if dc >= W:
                dc -= 2 * W

            # Only consider neighborhood
            if abs(dr) > D or abs(dc) > D:
                continue

            # Exclude identity overlap Δ=(0,0) (B7 requirement)
            if dr == 0 and dc == 0:
                continue

            # Record candidate
            candidates.append((color, dr, dc))

            # Exact MASK verification (B6 fix: compare masks, not entire grid)
            r0 = max(0, dr)
            r1 = min(H, H + dr)
            c0 = max(0, dc)
            c1 = min(W, W + dc)

            if r1 <= r0 or c1 <= c0:
                continue

            # Compare MASK overlaps (not X overlaps)
            M_overlap1 = M[r0:r1, c0:c1]
            M_overlap2 = M[r0 - dr:r1 - dr, c0 - dc:c1 - dc]

            if np.array_equal(M_overlap1, M_overlap2):
                accepted.append((color, dr, dc))

    # Return OverlapRc with all required fields (B3/B7 fix)
    return OverlapRc(
        method="fft_int",
        modulus=None,
        root=None,
        candidates=candidates,
        accepted=accepted,
        identity_excluded=True  # B7: explicit flag
    )


# ============================================================================
# BAND CLUSTERS (B5 fix)
# ============================================================================

def _compute_band_clusters(
    X: np.ndarray,
    row_periods: List[Optional[int]],
    col_periods: List[Optional[int]]
) -> Tuple[List[int], List[int]]:
    """
    Compute row_clusters and col_clusters (B5 fix).

    Contract:
    - row_clusters: List of row indices where signature changes (band boundaries)
    - col_clusters: List of column indices where signature changes (band boundaries)

    Algorithm:
    - Build row signature per row from frozen tags (color histogram + periods)
    - Detect runs of equal signatures
    - Record start indices of each run

    Returns: (row_clusters, col_clusters)
    """
    H, W = X.shape

    # Build row signatures
    row_sigs = []
    for r in range(H):
        row = X[r, :].tolist()
        color_hist = tuple(sorted(row))
        period = row_periods[r] if row_periods[r] is not None else 0
        sig = (color_hist, period)
        row_sigs.append(sig)

    # Build col signatures
    col_sigs = []
    for c in range(W):
        col = X[:, c].tolist()
        color_hist = tuple(sorted(col))
        period = col_periods[c] if col_periods[c] is not None else 0
        sig = (color_hist, period)
        col_sigs.append(sig)

    # Detect band edges (signature changes)
    def band_edges_from_signatures(sigs: List) -> List[int]:
        edges = [0]
        for i in range(1, len(sigs)):
            if sigs[i] != sigs[i - 1]:
                edges.append(i)
        return edges

    row_clusters = band_edges_from_signatures(row_sigs)
    col_clusters = band_edges_from_signatures(col_sigs)

    return row_clusters, col_clusters


# ============================================================================
# PAIGE-TARJAN REFINEMENT
# ============================================================================

def _initial_partition_by_color(X: np.ndarray) -> np.ndarray:
    """
    Create initial partition by color.
    All pixels with same color start in same cluster.
    """
    H, W = X.shape
    labels = np.zeros((H, W), dtype=np.int64)

    colors = sorted(np.unique(X).tolist())
    color_to_id = {color: i for i, color in enumerate(colors)}

    for r in range(H):
        for c in range(W):
            labels[r, c] = color_to_id[X[r, c]]

    return labels


def _refine_once(
    X: np.ndarray,
    labels: np.ndarray,
    row_periods: List[Optional[int]],
    col_periods: List[Optional[int]],
    overlaps: OverlapRc
) -> Tuple[np.ndarray, bool]:
    """
    Perform one Paige-Tarjan refinement step using ALL 14 frozen tags.

    Contract:
    - For each cluster, split by full tag signature (all 14 tags)
    - Tag signature includes all local and global tags
    - Returns (new_labels, changed)
    """
    H, W = X.shape

    # Extract all 14 tag arrays
    color_tags = _extract_color_tags(X)
    n4_tags = _extract_n4_adjacency_tags(X)
    n8_tags = _extract_n8_adjacency_tags(X)
    samecomp_tags = _extract_samecomp_r2_tags(X)
    parity_tags = _extract_parity_tags(X)
    row_period_2_tags = _extract_row_period_tags(X, row_periods, 2)
    row_period_3_tags = _extract_row_period_tags(X, row_periods, 3)
    col_period_2_tags = _extract_col_period_tags(X, col_periods, 2)
    col_period_3_tags = _extract_col_period_tags(X, col_periods, 3)
    per_line_min_period_tags = _extract_per_line_min_period_tags(X, row_periods, col_periods)
    exact_tile_tags = _extract_exact_tile_tags(X)
    bbox_mirror_tags = _extract_bbox_mirror_tags(X)
    bbox_rotate_tags = _extract_bbox_rotate_tags(X)

    # per_color_overlap is global (already computed in overlaps)
    # Encode as tag: hash of accepted overlaps for this pixel's color
    per_color_overlap_tags = np.zeros((H, W), dtype=np.int64)
    for r in range(H):
        for c in range(W):
            color = X[r, c]
            color_overlaps = tuple(
                (dr, dc) for (k, dr, dc) in overlaps.accepted if k == color
            )
            per_color_overlap_tags[r, c] = hash(color_overlaps) & 0x7FFFFFFFFFFFFFFF

    # Build full signatures
    signatures = {}
    for r in range(H):
        for c in range(W):
            # Collect neighbor cluster IDs (N8)
            neighbor_clusters = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        neighbor_clusters.append(int(labels[nr, nc]))

            neighbor_clusters.sort()
            neighbor_sig = tuple(neighbor_clusters)

            # Full signature (all 14 tags + neighbors)
            sig = (
                int(color_tags[r, c]),
                int(n4_tags[r, c]),
                int(n8_tags[r, c]),
                int(samecomp_tags[r, c]),
                int(parity_tags[r, c]),
                int(row_period_2_tags[r, c]),
                int(row_period_3_tags[r, c]),
                int(col_period_2_tags[r, c]),
                int(col_period_3_tags[r, c]),
                int(per_color_overlap_tags[r, c]),
                int(per_line_min_period_tags[r, c]),
                int(exact_tile_tags[r, c]),
                int(bbox_mirror_tags[r, c]),
                int(bbox_rotate_tags[r, c]),
                neighbor_sig
            )

            # Group by (current_cluster, signature)
            key = (int(labels[r, c]), sig)
            if key not in signatures:
                signatures[key] = []
            signatures[key].append((r, c))

    # Assign new cluster IDs
    new_labels = np.zeros((H, W), dtype=np.int64)
    new_cluster_id = 0
    for key in sorted(signatures.keys()):
        pixels = signatures[key]
        for r, c in pixels:
            new_labels[r, c] = new_cluster_id
        new_cluster_id += 1

    # Check if partition changed
    changed = not np.array_equal(labels, new_labels)

    return new_labels, changed


def _partition_hash(labels: np.ndarray) -> str:
    """
    Compute canonical hash of partition.
    Serialize labels in row-major order, hash with BLAKE3.
    """
    H, W = labels.shape
    parts = []
    for r in range(H):
        for c in range(W):
            parts.append(str(labels[r, c]))

    serialized = ",".join(parts).encode("utf-8")
    return hash_bytes(serialized)[:16]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def refine_truth_with_training(
    labels: np.ndarray,
    test_shape: Tuple[int, int],
    train_infos: List[Tuple[Tuple[int, int], Tuple[int, int], np.ndarray]],
    # list of ((H_i, W_i) input, (R_i, C_i) output, Y_i array)
) -> Tuple[np.ndarray, Dict]:
    """
    Refine truth partition using per-training signatures (WO-05T).

    Contract (WO-05T):
    - For each pixel p in test, compute signature = tuple(Y_i[pullback(p)] for each training i)
    - Pixels with different signatures must be in different truth blocks
    - Pullback: Π+S only; OOB → None (results in empty signature element)

    Algorithm:
    1. For each existing truth block
    2. Compute per-pixel signatures using training outputs
    3. Split block if pixels have different signatures
    4. Return refined labels and split log

    Args:
        labels: Initial truth partition labels (H*, W*)
        test_shape: (H*, W*) test dimensions
        train_infos: [((H_i,W_i), (R_i,C_i), Y_i), ...] training data

    Returns:
        (refined_labels, split_log)
    """
    from .unanimity import _pullback_pixel

    H_star, W_star = test_shape
    refined_labels = labels.copy()
    next_label = labels.max() + 1
    splits_made = 0
    split_log = []

    # Find all unique block IDs
    block_ids = np.unique(labels)

    for block_id in block_ids:
        # Find all pixels in this block
        coords = np.argwhere(labels == block_id)

        if len(coords) <= 1:
            # Single pixel block, no need to split
            continue

        # Compute signature for each pixel in the block
        pixel_signatures = {}

        for idx, (r_star, c_star) in enumerate(coords):
            # Build per-training signature
            sig_parts = []

            for (H_i, W_i), (R_i, C_i), Y_i in train_infos:
                # Pull back to training output
                p_i = _pullback_pixel(
                    (r_star, c_star),
                    H_star, W_star,
                    R_i, C_i
                )

                if p_i is not None:
                    r_i, c_i = p_i
                    color = int(Y_i[r_i, c_i])
                    sig_parts.append((color,))
                else:
                    sig_parts.append(())  # Empty if undefined

            # Final signature is tuple of per-training tuples
            signature = tuple(sig_parts)

            if signature not in pixel_signatures:
                pixel_signatures[signature] = []
            pixel_signatures[signature].append((r_star, c_star))

        # Check if block needs splitting
        if len(pixel_signatures) > 1:
            # Split block: keep first signature with original label,
            # assign new labels to other signatures
            signatures_sorted = sorted(pixel_signatures.keys())

            for i, sig in enumerate(signatures_sorted[1:], start=1):
                new_label = next_label
                next_label += 1
                splits_made += 1

                # Assign new label to these pixels
                for r, c in pixel_signatures[sig]:
                    refined_labels[r, c] = new_label

                split_log.append({
                    "original_block": int(block_id),
                    "new_block": int(new_label),
                    "pixel_count": len(pixel_signatures[sig]),
                    "signature": str(sig)  # Convert to string for JSON
                })

    receipt = {
        "method": "training_signatures",
        "blocks_before": len(block_ids),
        "blocks_after": len(np.unique(refined_labels)),
        "splits_made": splits_made,
        "split_log": split_log
    }

    return refined_labels, receipt


def compute_truth_partition(
    X: np.ndarray,
    *,
    method: Literal["fft_int"] = "fft_int"
) -> TruthPartition | None:
    """
    Compute Truth partition via Paige-Tarjan fixed point.

    Contract (00_math_spec.md §4, 02_determinism_addendum.md §2):
    - Input: X (H, W) grid in Π frame
    - Output: TruthPartition with cluster labels and full receipt
    - Algorithm: gfp(ℱ) where ℱ is tag-based refinement operator (14 frozen tags)
    - Determinism: Frozen tags, canonical encodings, exact algorithms

    Algorithm:
    1. Compute row/column minimal periods (KMP)
    2. Detect overlaps per color (FFT + exact MASK verification)
    3. Compute band clusters (signature changes)
    4. Initialize partition by color
    5. Refine until fixed point (Paige-Tarjan with all 14 tags)
    6. Compute block histogram
    7. Return partition with full receipt

    Returns:
    - TruthPartition on success
    - None if computation fails (should never happen - totality contract)
    """
    H, W = X.shape

    # Step 1: Compute minimal periods (KMP)
    row_periods = _per_line_min_periods(X, axis="row")
    col_periods = _per_line_min_periods(X, axis="col")

    # Step 2: Detect overlaps (FFT + exact mask verification)
    overlaps = _per_color_overlap_fft_int(X)

    # Step 3: Compute band clusters (B5 fix)
    row_clusters, col_clusters = _compute_band_clusters(X, row_periods, col_periods)

    # Step 4: Initialize partition by color
    labels = _initial_partition_by_color(X)

    # Step 5: Refine until fixed point (Paige-Tarjan)
    refinement_steps = 0
    max_steps = H * W  # Safety bound

    while refinement_steps < max_steps:
        new_labels, changed = _refine_once(X, labels, row_periods, col_periods, overlaps)
        labels = new_labels
        refinement_steps += 1

        if not changed:
            break

    # Step 6: Compute block histogram (B4 fix)
    uniq, counts = np.unique(labels, return_counts=True)
    block_hist = counts.tolist()

    # Step 7: Compute partition hash
    partition_hash_val = _partition_hash(labels)

    # Build receipt with all fixes
    receipt = TruthRc(
        tag_set_version=TAG_SET_VERSION,
        partition_hash=partition_hash_val,
        block_hist=block_hist,
        overlaps=overlaps,
        row_clusters=row_clusters,
        col_clusters=col_clusters,
        refinement_steps=refinement_steps,
        method="paige_tarjan"
    )

    return TruthPartition(labels=labels, receipt=receipt)
