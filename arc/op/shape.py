# arc/op/shape.py
# WO-02: Shape S synthesizer (exact + least)
# Implements 00_math_spec.md §2, 01_engineering_spec.md §3, 02_determinism_addendum.md §1.1

# Registered COUNT qualifiers (frozen):
# - q_rows: R=α1·H+β1, C=α2·H+β2 (param serialization: <4><α1><β1><α2><β2>)
# - q_hw_bilinear: R=a1·H+b1, C=c2·W+d2·H+e2 (param serialization: <5><a1><b1><c2><d2><e2>)
#
# Activation rule: COUNT candidate considered ONLY if equality proof exists
# using a registered qualifier on ALL trainings.
#
# Receipts must include: qual_id, qual_hash (BLAKE3 of ASCII qualifier name)

from __future__ import annotations
import numpy as np
from typing import Callable
from dataclasses import dataclass
from blake3 import blake3
from .bytes import frame_params
from .receipts import ShapeRc


# Type alias for shape function S(H,W) -> (R,C)
SFn = Callable[[int, int], tuple[int, int]]


def _hash_ascii(s: str) -> str:
    """Hash ASCII string with BLAKE3."""
    return blake3(s.encode()).hexdigest()


def _lcm(a: int, b: int) -> int:
    """Least common multiple."""
    import math
    if a == 0 or b == 0:
        return max(a, b)
    return abs(a * b) // math.gcd(a, b)


def _min_period_kmp(seq: list[int]) -> int:
    """
    Compute minimal period of sequence using KMP prefix table.

    Contract (02_determinism_addendum.md §2):
    "per-line minimal periods (KMP)"

    Returns:
        Minimal period p such that seq[i] = seq[i % p] for all i
        If no period < len(seq), returns len(seq)
    """
    n = len(seq)
    if n == 0:
        return 0

    # Build KMP prefix table
    lps = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and seq[i] != seq[j]:
            j = lps[j - 1]
        if seq[i] == seq[j]:
            j += 1
            lps[i] = j

    # Minimal period is n - lps[n-1] if it divides n
    p = n - lps[-1]
    if n % p == 0:
        return p
    return n


def _min_periods_grid(G: np.ndarray) -> tuple[int, int]:
    """
    Compute minimal row and column periods for grid.

    Returns:
        (pr_min, pc_min): LCM of per-row and per-col minimal periods
    """
    H, W = G.shape

    # Row periods
    pr = 1
    for i in range(H):
        p = _min_period_kmp(G[i].tolist())
        if p > 1:
            pr = _lcm(pr, p)

    # Column periods
    pc = 1
    for j in range(W):
        p = _min_period_kmp(G[:, j].tolist())
        if p > 1:
            pc = _lcm(pc, p)

    return pr, pc


def _solve_linear_2var(samples: list[tuple[str, int, int]]) -> tuple[int, int] | None:
    """
    Solve y = a*x + b from samples.

    Contract: exact equality (no regression, no thresholds)

    Args:
        samples: [(id, x, y), ...]

    Returns:
        (a, b) if exact solution exists, None otherwise
    """
    # Collect unique x values
    vals: dict[int, set[int]] = {}
    for _, x, y in samples:
        vals.setdefault(x, set()).add(y)

    xs = sorted(vals.keys())

    # Single x value: underdetermined, use constant (a=0)
    if len(xs) == 1:
        x = xs[0]
        ys = vals[x]
        if len(ys) != 1:
            return None  # contradiction: same x, different y
        y = next(iter(ys))
        # a=0, b=y (constant function)
        a, b = 0, y
    else:
        # At least 2 distinct x values: solve 2-point line
        x1, x2 = xs[0], xs[1]
        y1_set = vals[x1]
        y2_set = vals[x2]

        if len(y1_set) != 1 or len(y2_set) != 1:
            return None  # contradiction

        y1 = next(iter(y1_set))
        y2 = next(iter(y2_set))

        # Solve: y = a*x + b
        if x2 - x1 == 0:
            return None  # shouldn't happen (x1 != x2)

        a = (y2 - y1) // (x2 - x1)
        b = y1 - a * x1

        # Verify it's exact (not just best-fit)
        if a * x1 + b != y1 or a * x2 + b != y2:
            return None  # integer division doesn't give exact solution

    # Verify all samples
    for _, x, y in samples:
        if a * x + b != y:
            return None

    return a, b


def _fit_affine(pairs: list[tuple[str, tuple[int, int], tuple[int, int]]]) -> tuple[SFn, bytes, dict] | None:
    """
    Fit AFFINE family: R = aH + b, C = cW + d.

    Contract (00_math_spec.md §2):
    "Affine: S(H,W)=(aH+b, cW+d), a,b,c,d ∈ Z≥0"

    Returns:
        (S_fn, params_bytes, extras) if exact fit, None otherwise
    """
    # Solve R = aH + b
    H_to_R = [(tid, H, R) for tid, (H, W), (R, C) in pairs]
    ab = _solve_linear_2var(H_to_R)
    if ab is None:
        return None
    a, b = ab

    # Solve C = cW + d
    W_to_C = [(tid, W, C) for tid, (H, W), (R, C) in pairs]
    cd = _solve_linear_2var(W_to_C)
    if cd is None:
        return None
    c, d = cd

    # Verify all pairs (redundant but explicit)
    def S(H: int, W: int) -> tuple[int, int]:
        return (a * H + b, c * W + d)

    for _, (H, W), (R, C) in pairs:
        if S(H, W) != (R, C):
            return None

    # Serialize params: <4><a><b><c><d>
    params = frame_params(a, b, c, d, signed=False)
    extras: dict = {}

    return S, params, extras


def _fit_period(
    pairs: list[tuple[str, tuple[int, int], tuple[int, int]]],
    presented_inputs: list[tuple[str, np.ndarray]] | None
) -> tuple[SFn, bytes, dict] | None:
    """
    Fit PERIOD_MULTIPLE family: (kr·pr^min, kc·pc^min).

    Contract (00_math_spec.md §2):
    "Period-multiple: compute minimal per-line periods p_r(i), p_c(j) (exact, KMP)"

    Args:
        pairs: [(id, (H,W), (R,C))]
        presented_inputs: [(id, G)] - needed to compute periods

    Returns:
        (S_fn, params_bytes, extras) if exact fit, None otherwise
    """
    if presented_inputs is None:
        return None  # cannot compute periods without grids

    # Compute LCM of minimal periods across all inputs
    pr_lcm = 1
    pc_lcm = 1
    for _, G in presented_inputs:
        pr, pc = _min_periods_grid(G)
        pr_lcm = _lcm(pr_lcm, pr)
        pc_lcm = _lcm(pc_lcm, pc)

    # If both degenerate (1), skip PERIOD family
    if pr_lcm == 1 and pc_lcm == 1:
        return None

    # Find kr, kc such that (R', C') = (kr·pr_lcm, kc·pc_lcm) for all trainings
    kr_set: set[int] = set()
    kc_set: set[int] = set()

    for _, (H, W), (R, C) in pairs:
        if pr_lcm > 1:
            if R % pr_lcm != 0:
                return None  # R not a multiple
            kr_set.add(R // pr_lcm)
        else:
            kr_set.add(R)  # degenerate case

        if pc_lcm > 1:
            if C % pc_lcm != 0:
                return None  # C not a multiple
            kc_set.add(C // pc_lcm)
        else:
            kc_set.add(C)

    # All trainings must agree on kr and kc
    if len(kr_set) != 1 or len(kc_set) != 1:
        return None

    kr = next(iter(kr_set))
    kc = next(iter(kc_set))

    def S(H: int, W: int) -> tuple[int, int]:
        return (kr * pr_lcm, kc * pc_lcm)

    # Verify
    for _, (H, W), (R, C) in pairs:
        if S(H, W) != (R, C):
            return None

    # Serialize params: <2><kr><kc>
    params = frame_params(kr, kc, signed=False)

    extras = {
        "row_periods_lcm": pr_lcm,
        "col_periods_lcm": pc_lcm,
        "axis_code": 0,  # rows (default; tie logic not yet implemented)
    }

    return S, params, extras


def _fit_count_rows(pairs: list[tuple[str, tuple[int, int], tuple[int, int]]]) -> tuple[SFn, bytes, dict] | None:
    """
    Fit COUNT_BASED family with q_rows qualifier: q(H,W) = H.

    Contract (00_math_spec.md §2):
    "Count-based: (H'i,W'i)=(α1·#qual(Xi)+β1, α2·#qual(Xi)+β2)"

    Qualifier q_rows: q(H,W) = H
    Thus: R = α1·H + β1, C = α2·H + β2

    Returns:
        (S_fn, params_bytes, extras) if exact fit, None otherwise
    """
    # Solve R = α1·H + β1
    H_to_R = [(tid, H, R) for tid, (H, W), (R, C) in pairs]
    a1b1 = _solve_linear_2var(H_to_R)
    if a1b1 is None:
        return None
    a1, b1 = a1b1

    # Solve C = α2·H + β2
    H_to_C = [(tid, H, C) for tid, (H, W), (R, C) in pairs]
    a2b2 = _solve_linear_2var(H_to_C)
    if a2b2 is None:
        return None
    a2, b2 = a2b2

    def S(H: int, W: int) -> tuple[int, int]:
        return (a1 * H + b1, a2 * H + b2)

    # Verify
    for _, (H, W), (R, C) in pairs:
        if S(H, W) != (R, C):
            return None

    # Serialize params: <4><α1><β1><α2><β2> (signed, may have negatives)
    params = frame_params(a1, b1, a2, b2, signed=True)

    extras = {
        "qual_id": "q_rows",
        "qual_hash": _hash_ascii("q_rows"),
    }

    return S, params, extras


def _fit_count_hw_bilinear(pairs: list[tuple[str, tuple[int, int], tuple[int, int]]]) -> tuple[SFn, bytes, dict] | None:
    """
    Fit COUNT_BASED family with registered q_hw_bilinear qualifier.

    Contract (WO-02 patch):
    Registered qualifier: q_hw_bilinear
    Semantics (frozen): R = a1·H + b1, C = c2·W + d2·H + e2
    Parameter serialization: <5><a1><b1><c2><d2><e2>

    This handles patterns like C = W + (H-1) where C depends on both W and H.

    Activation rule: Only consider if equality proof exists for ALL trainings.

    Returns:
        (S_fn, params_bytes, extras) if exact fit, None otherwise
    """
    # Solve R = a1·H + b1 (same as q_rows)
    H_to_R = [(tid, H, R) for tid, (H, W), (R, C) in pairs]
    a1b1 = _solve_linear_2var(H_to_R)
    if a1b1 is None:
        return None
    # Convert to Python int (not numpy int)
    a1, b1 = int(a1b1[0]), int(a1b1[1])

    # Solve C = c2·W + d2·H + e2 (bilinear in W and H)
    # Need at least 3 training examples with varying W and H
    if len(pairs) < 3:
        return None

    # Try to solve: C = c2·W + d2·H + e2
    # Use first 3 samples to solve for c2, d2, e2
    samples = [(H, W, C) for _, (H, W), (R, C) in pairs[:3]]
    (H1, W1, C1), (H2, W2, C2), (H3, W3, C3) = samples

    # System of equations:
    # C1 = c2·W1 + d2·H1 + e2
    # C2 = c2·W2 + d2·H2 + e2
    # C3 = c2·W3 + d2·H3 + e2

    # Solve using matrix approach (simplified for 3 equations)
    # Try integer solutions
    import numpy as np

    try:
        A = np.array([
            [W1, H1, 1],
            [W2, H2, 1],
            [W3, H3, 1]
        ], dtype=np.float64)
        b = np.array([C1, C2, C3], dtype=np.float64)

        # Check if matrix is singular
        if np.linalg.det(A) == 0:
            return None

        solution = np.linalg.solve(A, b)
        c2, d2, e2 = solution

        # Round to nearest integer and check if it's exact
        # Convert to Python int (not numpy int) for zigzag encoding
        c2_int = int(round(float(c2)))
        d2_int = int(round(float(d2)))
        e2_int = int(round(float(e2)))

        # Verify the rounded solution works for all samples
        def S(H: int, W: int) -> tuple[int, int]:
            return (a1 * H + b1, c2_int * W + d2_int * H + e2_int)

        for _, (H, W), (R, C) in pairs:
            if S(H, W) != (R, C):
                return None

        # Serialize params: <5><a1><b1><c2><d2><e2> (patch specification)
        params = frame_params(a1, b1, c2_int, d2_int, e2_int, signed=True)

        extras = {
            "qual_id": "q_hw_bilinear",
            "qual_hash": _hash_ascii("q_hw_bilinear"),
        }

        return S, params, extras

    except (np.linalg.LinAlgError, ValueError):
        return None


def _fit_frame(pairs: list[tuple[str, tuple[int, int], tuple[int, int]]]) -> tuple[SFn, bytes, dict] | None:
    """
    Fit FRAME family (stubbed).

    Contract (00_math_spec.md §2):
    "BBox/frame and pad-to-multiple(k) if exactly implied"

    Returns:
        None (not yet implemented)
    """
    # TODO: implement when needed
    return None


def synthesize_shape(
    train_pairs: list[tuple[str, tuple[int, int], tuple[int, int]]],
    presented_inputs: list[tuple[str, np.ndarray]] | None = None
) -> tuple[SFn, ShapeRc]:
    """
    Synthesize shape function S from training examples.

    Contract (01_engineering_spec.md §3):
    "Synthesize constraints; choose the least branch/integers that fit all trainings"

    Contract (02_determinism_addendum.md §1.1):
    Ordering key: (branch_byte, params_bytes, R, C)

    Contract (WO-02 patch):
    Registered COUNT qualifiers: {"q_rows", "q_hw_bilinear"}
    Only registered qualifiers may be used.

    Args:
        train_pairs: [(train_id, (H,W), (R',C')), ...]
        presented_inputs: [(train_id, G), ...] - optional, for PERIOD

    Returns:
        (S_fn, ShapeRc): shape function and receipt

    Raises:
        ValueError: if no family fits all trainings exactly
    """
    # Registered qualifiers (frozen)
    REGISTERED_COUNT_QUALIFIERS = {"q_rows", "q_hw_bilinear"}

    candidates: list[tuple[str, SFn, bytes, dict]] = []

    # Try AFFINE
    aff = _fit_affine(train_pairs)
    if aff is not None:
        candidates.append(("A",) + aff)

    # Try PERIOD (requires grids)
    period = _fit_period(train_pairs, presented_inputs)
    if period is not None:
        candidates.append(("P",) + period)

    # Try COUNT (q_rows - registered qualifier)
    count = _fit_count_rows(train_pairs)
    if count is not None:
        candidates.append(("C",) + count)

    # Try COUNT (q_hw_bilinear - registered qualifier, patch-approved)
    count_hw_bilinear = _fit_count_hw_bilinear(train_pairs)
    if count_hw_bilinear is not None:
        candidates.append(("C",) + count_hw_bilinear)

    # Try FRAME (stubbed)
    frame = _fit_frame(train_pairs)
    if frame is not None:
        candidates.append(("F",) + frame)

    # Validate COUNT qualifiers (fail-closed on unregistered qualifiers)
    for branch, _, _, extras in candidates:
        if branch == "C":
            qual_id = extras.get("qual_id")
            if qual_id not in REGISTERED_COUNT_QUALIFIERS:
                raise ValueError(
                    f"CONTRACT_VIOLATION: Unregistered COUNT qualifier '{qual_id}'. "
                    f"Registered qualifiers: {REGISTERED_COUNT_QUALIFIERS}"
                )

    # Fail-closed: no family fits
    if not candidates:
        raise ValueError(
            "SHAPE_CONTRADICTION: no family (AFFINE, PERIOD, COUNT, FRAME) fits all trainings exactly"
        )

    # Select by lex-min on (branch_byte, params_bytes)
    # Note: R, C will be filled by caller after applying to test
    def key(c: tuple[str, SFn, bytes, dict]) -> tuple[str, bytes]:
        branch, _, params, _ = c
        return (branch, params)

    branch, S, params_bytes, extras = min(candidates, key=key)

    # Create receipt (R, C, verified_train_ids will be filled by caller)
    rc = ShapeRc(
        branch_byte=branch,
        params_bytes_hex=params_bytes.hex(),
        R=-1,  # placeholder
        C=-1,  # placeholder
        verified_train_ids=[],  # placeholder
        extras=extras,
    )

    return S, rc


def apply_shape(S: SFn, H: int, W: int) -> tuple[int, int]:
    """
    Apply shape function to input dimensions.

    Args:
        S: shape function from synthesize_shape
        H: input height
        W: input width

    Returns:
        (R, C): output dimensions
    """
    return S(H, W)
