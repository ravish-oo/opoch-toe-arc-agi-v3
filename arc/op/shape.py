# arc/op/shape.py
# WO-02: Shape S synthesizer (exact + least)
# Implements 00_math_spec.md §2, 01_engineering_spec.md §3, 02_determinism_addendum.md §1.1

# Registered COUNT qualifiers (frozen per WO-02 patch):
# - q_rows: R=α1·H+β1, C=α2·H+β2 (param serialization: <4><α1><β1><α2><β2>)
# - q_hw_bilinear: R=a1·H+b1, C=c2·W+d2·H+e2 (param serialization: <5><a1><b1><c2><d2><e2>)
# - q_wh_bilinear: R=a1·W+b1·H+c1, C=a2·W+b2 (param serialization: <5><a1><b1><c1><a2><b2>)
# - q_components: R=a1·q+b1, C=a2·q+b2 where q=#components (param serialization: <4><a1><b1><a2><b2>)
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

# Rational AFFINE max denominator (FROZEN per 02_determinism_addendum.md §1.1.1)
MAX_DENOMINATOR_AFFINE_RATIONAL = 10


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

    Contract (WO-02 integer AFFINE fix):
    Uses ALL pair differences to prove unique integer a exists.
    Not just first two points - verifies all (i,j) pairs have same slope.

    Args:
        samples: [(id, x, y), ...]

    Returns:
        (a, b) if exact integer solution exists, None otherwise
    """
    # Extract x and y values
    xs = [x for _, x, _ in samples]
    ys = [y for _, _, y in samples]

    # Check for duplicate x with different y (contradiction)
    x_to_ys: dict[int, set[int]] = {}
    for x, y in zip(xs, ys):
        x_to_ys.setdefault(x, set()).add(y)

    for x, y_set in x_to_ys.items():
        if len(y_set) > 1:
            return None  # Same x, different y → contradiction

    # If only one unique x: constant function (a=0, b=y)
    unique_xs = set(xs)
    if len(unique_xs) == 1:
        # All x same → a=0, b=y (if all y same)
        unique_ys = set(ys)
        if len(unique_ys) != 1:
            return None  # Contradiction
        return 0, ys[0]

    # Find integer a from ALL pair differences
    # For all i,j: (y_i - y_j) must equal a·(x_i - x_j)
    # Thus a must be same for all pairs with distinct x
    a_val = None
    n = len(samples)

    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]

            if dx == 0:
                # Same x: already checked for contradiction above
                if dy != 0:
                    return None  # Safety check
                continue

            # Check if dy/dx is integer
            if dy % dx != 0:
                return None  # Not integer slope

            candidate_a = dy // dx

            if a_val is None:
                a_val = candidate_a
            elif a_val != candidate_a:
                return None  # Different pairs give different a → no solution

    # If all pairs had dx=0 (shouldn't happen given len(unique_xs)>1), fallback
    if a_val is None:
        a_val = 0

    # Compute b from each sample: b = y - a·x (must be constant)
    b_vals = set(y - a_val * x for x, y in zip(xs, ys))
    if len(b_vals) != 1:
        return None  # Different samples give different b

    b_val = next(iter(b_vals))

    # Final verification: all samples must satisfy y = a·x + b
    for x, y in zip(xs, ys):
        if a_val * x + b_val != y:
            return None  # Safety check

    return a_val, b_val


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

    # Serialize params: <4><a><b><c><d> (ZigZag for signed coefficients)
    # Contract (02_determinism_addendum.md line 16): signed integers use ZigZag LEB128
    params = frame_params(a, b, c, d, signed=True)
    extras: dict = {}

    return S, params, extras


def _fit_affine_rational_floor(
    pairs: list[tuple[str, tuple[int, int], tuple[int, int]]],
    max_d: int = 10
) -> tuple[SFn, bytes, dict] | None:
    """
    Fit rational AFFINE with floor division:
      R = floor((a·H + b) / d1)
      C = floor((c·W + e) / d2)

    Contract (02_determinism_addendum.md §1.1.1):
    - Denominators: d1, d2 ∈ {2, ..., 10} FROZEN
    - Exact interval intersection over integer inequalities
    - Lex-min selection by (d1, a, b, d2, c, e)

    Args:
        pairs: [(id, (H,W), (R,C))]
        max_d: Maximum denominator (FROZEN=10)

    Returns:
        (S_fn, params_bytes, extras) if solution exists, None otherwise
    """
    import math

    # Extract H→R and W→C relationships
    Hs = [H for _, (H, W), _ in pairs]
    Ws = [W for _, (H, W), _ in pairs]
    Rs = [R for _, _, (R, _) in pairs]
    Cs = [C for _, _, (_, C) in pairs]

    def fit_axis(X: list[int], Y: list[int]) -> tuple[int, int, int] | None:
        """
        Fit Y = floor((a·X + b) / d) for smallest valid d.

        Returns: (d, a, b) or None
        """
        best = None
        n = len(X)

        # Try denominators d ∈ {2, ..., max_d}
        for d in range(2, max_d + 1):
            # Step 1: Bound 'a' from ALL pair differences
            # For any i,j: d·(Y_i - Y_j) - (d-1) ≤ a·(X_i - X_j) ≤ d·(Y_i - Y_j) + (d-1)
            a_lo, a_hi = -(10**9), 10**9
            ok = True

            for i in range(n):
                for j in range(i + 1, n):
                    dx = X[i] - X[j]
                    dy = Y[i] - Y[j]

                    if dx == 0:
                        # Same X: Y must be same (floor constraint)
                        if Y[i] != Y[j]:
                            ok = False
                            break
                        continue

                    # Bounds for a·dx from floor constraint
                    base = d * dy
                    lo = base - (d - 1)
                    hi = base + (d - 1)

                    if dx > 0:
                        # a ∈ [ceil(lo/dx), floor(hi/dx)]
                        a_lo = max(a_lo, math.ceil(lo / dx))
                        a_hi = min(a_hi, math.floor(hi / dx))
                    else:  # dx < 0
                        # Reverse bounds due to negative divisor
                        a_lo = max(a_lo, math.ceil(hi / dx))
                        a_hi = min(a_hi, math.floor(lo / dx))

                    if a_lo > a_hi:
                        ok = False
                        break

                if not ok:
                    break

            if not ok or a_lo > a_hi:
                continue  # No valid 'a' for this d

            # Pick lex-min 'a'
            a = a_lo

            # Step 2: Bound 'b' from per-sample constraints
            # For each i: d·Y_i ≤ a·X_i + b ≤ d·Y_i + (d-1)
            b_lo, b_hi = -(10**9), 10**9

            for i in range(n):
                lo = d * Y[i] - a * X[i]
                hi = d * Y[i] + (d - 1) - a * X[i]
                b_lo = max(b_lo, lo)
                b_hi = min(b_hi, hi)

                if b_lo > b_hi:
                    ok = False
                    break

            if not ok:
                continue

            # Pick lex-min 'b'
            b = b_lo

            # Verify: floor((a·X_i + b) / d) == Y_i for ALL i
            if all((a * X[i] + b) // d == Y[i] for i in range(n)):
                cand = (d, a, b)
                if best is None or cand < best:  # Lex-min
                    best = cand

        return best  # (d, a, b) or None

    # Fit R = floor((a·H + b) / d1)
    RH = fit_axis(Hs, Rs)
    if RH is None:
        return None

    # Fit C = floor((c·W + e) / d2)
    CW = fit_axis(Ws, Cs)
    if CW is None:
        return None

    d1, a, b = RH
    d2, c, e = CW

    # S function
    def S(H: int, W: int) -> tuple[int, int]:
        R = (a * H + b) // d1
        C = (c * W + e) // d2
        return (R, C)

    # Verify (safety)
    for _, (H, W), (R, C) in pairs:
        if S(H, W) != (R, C):
            return None

    # Serialize params: <6><d1><a><b><d2><c><e>
    # ZigZag for signed integers (a, b, c, e can be negative)
    params = frame_params(d1, a, b, d2, c, e, signed=True)

    extras = {
        "rational_floor": True,
        "d1": d1,
        "d2": d2,
    }

    return S, params, extras


def _fit_period(
    pairs: list[tuple[str, tuple[int, int], tuple[int, int]]],
    presented_inputs: list[tuple[str, np.ndarray]]
) -> tuple[SFn, bytes, dict] | None:
    """
    Fit PERIOD_MULTIPLE family: (kr·pr^min, kc·pc^min).

    Contract (00_math_spec.md §2):
    "Period-multiple: compute minimal per-line periods p_r(i), p_c(j) (exact, KMP)"

    Contract (WO-02 reconciled patch):
    Actually use presented grids to compute periods (was bug: always returned None)

    Args:
        pairs: [(id, (H,W), (R,C))]
        presented_inputs: [(id, G)] - REQUIRED to compute periods

    Returns:
        (S_fn, params_bytes, extras) if exact fit, None otherwise
    """
    # Compute LCM of minimal periods across all presented inputs
    pr_lcm = 1
    pc_lcm = 1

    for _, G in presented_inputs:
        pr, pc = _min_periods_grid(G)
        if pr > 1:
            pr_lcm = _lcm(pr_lcm, pr)
        if pc > 1:
            pc_lcm = _lcm(pc_lcm, pc)

    # If both degenerate (1), no periodic pattern
    if pr_lcm == 1 and pc_lcm == 1:
        return None

    # Find kr, kc such that (R', C') = (kr·pr_lcm, kc·pc_lcm) for all trainings
    kr = None
    kc = None

    for _, (H, W), (R, C) in pairs:
        # Check row multiple
        if pr_lcm > 1:
            if R % pr_lcm != 0:
                return None  # R not a multiple of pr_lcm
            k = R // pr_lcm
            if kr is None:
                kr = k
            elif kr != k:
                return None  # Inconsistent kr across trainings
        else:
            # No row period, use R directly
            if kr is None:
                kr = R
            elif kr != R:
                return None

        # Check col multiple
        if pc_lcm > 1:
            if C % pc_lcm != 0:
                return None  # C not a multiple of pc_lcm
            k = C // pc_lcm
            if kc is None:
                kc = k
            elif kc != k:
                return None  # Inconsistent kc across trainings
        else:
            # No col period, use C directly
            if kc is None:
                kc = C
            elif kc != C:
                return None

    # Sanity check
    if kr is None or kc is None:
        return None

    def S(H: int, W: int) -> tuple[int, int]:
        return (kr * max(1, pr_lcm), kc * max(1, pc_lcm))

    # Verify all pairs
    for _, (H, W), (R, C) in pairs:
        if S(H, W) != (R, C):
            return None

    # Serialize params: <2><kr><kc>
    params = frame_params(kr, kc, signed=False)

    extras = {
        "row_periods_lcm": pr_lcm,
        "col_periods_lcm": pc_lcm,
        "axis_code": 0,  # rows (default; axis-tie rule deferred)
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


def _fit_count_cols(pairs: list[tuple[str, tuple[int, int], tuple[int, int]]]) -> tuple[SFn, bytes, dict] | None:
    """
    Fit COUNT_BASED family with q_cols qualifier: q(H,W) = W.

    Contract (WO-02 reconciled patch):
    Registered qualifier: q_cols (symmetric to q_rows)
    Semantics (frozen): R = α1·W + β1, C = α2·W + β2

    This handles patterns where output depends on width only.

    Returns:
        (S_fn, params_bytes, extras) if exact fit, None otherwise
    """
    # Solve R = α1·W + β1
    W_to_R = [(tid, W, R) for tid, (H, W), (R, C) in pairs]
    a1b1 = _solve_linear_2var(W_to_R)
    if a1b1 is None:
        return None
    a1, b1 = a1b1

    # Solve C = α2·W + β2
    W_to_C = [(tid, W, C) for tid, (H, W), (R, C) in pairs]
    a2b2 = _solve_linear_2var(W_to_C)
    if a2b2 is None:
        return None
    a2, b2 = a2b2

    def S(H: int, W: int) -> tuple[int, int]:
        return (a1 * W + b1, a2 * W + b2)

    # Verify
    for _, (H, W), (R, C) in pairs:
        if S(H, W) != (R, C):
            return None

    # Serialize params: <4><α1><β1><α2><β2> (signed, may have negatives)
    params = frame_params(a1, b1, a2, b2, signed=True)

    extras = {
        "qual_id": "q_cols",
        "qual_hash": _hash_ascii("q_cols"),
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


def _fit_count_wh_bilinear(pairs: list[tuple[str, tuple[int, int], tuple[int, int]]]) -> tuple[SFn, bytes, dict] | None:
    """
    Fit COUNT_BASED family with registered q_wh_bilinear qualifier.

    Contract (WO-02 patch):
    Registered qualifier: q_wh_bilinear
    Semantics (frozen): R = a1·W + b1·H + c1, C = a2·W + b2
    Parameter serialization: <5><a1><b1><c1><a2><b2>

    This handles patterns where R depends on both W and H (bilinear in R).

    Activation rule: Only consider if equality proof exists for ALL trainings.

    Returns:
        (S_fn, params_bytes, extras) if exact fit, None otherwise
    """
    # Need at least 3 trainings for R (3 unknowns), 2 for C (2 unknowns)
    if len(pairs) < 3:
        return None

    import numpy as np

    # Collect data
    Hs = np.array([H for _, (H, W), (R, C) in pairs], dtype=np.int64)
    Ws = np.array([W for _, (H, W), (R, C) in pairs], dtype=np.int64)
    Rs = np.array([R for _, (H, W), (R, C) in pairs], dtype=np.int64)
    Cs = np.array([C for _, (H, W), (R, C) in pairs], dtype=np.int64)

    try:
        # Solve R = a1·W + b1·H + c1
        A_R = np.stack([Ws, Hs, np.ones_like(Hs)], axis=1)
        if np.linalg.matrix_rank(A_R) < 3:
            return None  # Underdetermined or degenerate

        sol_R = np.linalg.lstsq(A_R.astype(np.float64), Rs.astype(np.float64), rcond=None)[0]
        a1, b1, c1 = sol_R

        # Round to integers and verify exact
        a1_int = int(round(float(a1)))
        b1_int = int(round(float(b1)))
        c1_int = int(round(float(c1)))

        if not np.all(a1_int * Ws + b1_int * Hs + c1_int == Rs):
            return None

        # Solve C = a2·W + b2
        A_C = np.stack([Ws, np.ones_like(Ws)], axis=1)
        if np.linalg.matrix_rank(A_C) < 2:
            return None

        sol_C = np.linalg.lstsq(A_C.astype(np.float64), Cs.astype(np.float64), rcond=None)[0]
        a2, b2 = sol_C

        # Round to integers and verify exact
        a2_int = int(round(float(a2)))
        b2_int = int(round(float(b2)))

        if not np.all(a2_int * Ws + b2_int == Cs):
            return None

        # Define shape function
        def S(H: int, W: int) -> tuple[int, int]:
            return (a1_int * W + b1_int * H + c1_int, a2_int * W + b2_int)

        # Verify all pairs
        for _, (H, W), (R, C) in pairs:
            if S(H, W) != (R, C):
                return None

        # Serialize params: <5><a1><b1><c1><a2><b2>
        params = frame_params(a1_int, b1_int, c1_int, a2_int, b2_int, signed=True)

        extras = {
            "qual_id": "q_wh_bilinear",
            "qual_hash": _hash_ascii("q_wh_bilinear"),
        }

        return S, params, extras

    except (np.linalg.LinAlgError, ValueError):
        return None


def _fit_count_components(
    pairs: list[tuple[str, tuple[int, int], tuple[int, int]]],
    presented_inputs: list[tuple[str, np.ndarray]] | None
) -> tuple[SFn | None, bytes, dict, list[int]] | None:
    """
    Fit COUNT_BASED family with registered q_components qualifier.

    Contract (WO-02 patch):
    Registered qualifier: q_components
    Semantics (frozen): R = a1·q + b1, C = a2·q + b2 where q = #components
    Parameter serialization: <4><a1><b1><a2><b2>

    Contract (00_math_spec.md §2):
    "Count-based: ... where #qual is an exact content count (e.g., ... components)"

    This uses WO-03 CC4 (4-connected components) to compute q.

    Args:
        pairs: [(train_id, (H,W), (R,C)), ...]
        presented_inputs: [(train_id, G), ...] - required for component extraction

    Returns:
        (None, params_bytes, extras, qs) if exact fit (S_fn=None signals need for q at test time)
        None if no fit

    Note: S_fn is None because we can't evaluate S(H,W) without knowing q for test.
          The runner must compute q(test) and apply (R,C) = (a1·q+b1, a2·q+b2) directly.
    """
    if presented_inputs is None:
        return None  # Cannot compute q without grids

    if len(pairs) < 2:
        return None  # Need at least 2 trainings to solve 2-variable systems

    # Import here to avoid circular dependency
    from .components import cc4_by_color

    import numpy as np

    # Compute q for each training
    qs = []
    for (tid, (H, W), (R, C)), (tid2, G) in zip(pairs, presented_inputs):
        if tid != tid2:
            raise ValueError(f"Mismatched training IDs: {tid} != {tid2}")
        _, rc = cc4_by_color(G)
        q = len(rc.invariants)
        qs.append(q)

    qs_arr = np.array(qs, dtype=np.int64)
    Rs = np.array([R for _, _, (R, C) in pairs], dtype=np.int64)
    Cs = np.array([C for _, _, (_, C) in pairs], dtype=np.int64)

    try:
        # Solve R = a1·q + b1
        A = np.stack([qs_arr, np.ones_like(qs_arr)], axis=1)
        if np.linalg.matrix_rank(A) < 2:
            return None

        sol_R = np.linalg.lstsq(A.astype(np.float64), Rs.astype(np.float64), rcond=None)[0]
        a1, b1 = sol_R

        # Round to integers and verify exact
        a1_int = int(round(float(a1)))
        b1_int = int(round(float(b1)))

        if not np.all(a1_int * qs_arr + b1_int == Rs):
            return None

        # Solve C = a2·q + b2
        sol_C = np.linalg.lstsq(A.astype(np.float64), Cs.astype(np.float64), rcond=None)[0]
        a2, b2 = sol_C

        # Round to integers and verify exact
        a2_int = int(round(float(a2)))
        b2_int = int(round(float(b2)))

        if not np.all(a2_int * qs_arr + b2_int == Cs):
            return None

        # Serialize params: <4><a1><b1><a2><b2>
        params = frame_params(a1_int, b1_int, a2_int, b2_int, signed=True)

        extras = {
            "qual_id": "q_components",
            "qual_hash": _hash_ascii("q_components"),
            "coeffs": (a1_int, b1_int, a2_int, b2_int),  # Store for runner to apply
        }

        # Return None for S_fn (signals special handling), params, extras, and qs
        return None, params, extras, qs.copy()

    except (np.linalg.LinAlgError, ValueError):
        return None


def _fit_frame(pairs: list[tuple[str, tuple[int, int], tuple[int, int]]]) -> tuple[SFn, bytes, dict] | None:
    """
    Fit FRAME family: constant output size.

    Contract (00_math_spec.md §2):
    "BBox/frame and pad-to-multiple(k) if exactly implied"

    Contract (WO-02 reconciled patch):
    Implement constant output case: S(H,W) = (R₀, C₀) for all (H,W)

    Note: This handles constant outputs only. Pad-to-multiple(k) deferred.

    Args:
        pairs: [(train_id, (H,W), (R,C)), ...]

    Returns:
        (S_fn, params_bytes, extras) if all outputs same shape, None otherwise
    """
    # Check if all outputs have same shape
    output_shapes = {(R, C) for _, _, (R, C) in pairs}

    if len(output_shapes) != 1:
        return None  # Not constant

    R0, C0 = next(iter(output_shapes))

    # S function (constant)
    def S(H: int, W: int) -> tuple[int, int]:
        return (R0, C0)

    # Serialize params: <2><R0><C0>
    params = frame_params(R0, C0, signed=False)

    extras = {
        "frame": "const",
        "R_const": R0,
        "C_const": C0,
    }

    return S, params, extras


def synthesize_shape(
    train_pairs: list[tuple[str, tuple[int, int], tuple[int, int]]],
    presented_inputs: list[tuple[str, np.ndarray]] | None = None,
    *,
    fail_fast: bool = False,
    test_shape: tuple[int, int] | None = None
) -> tuple[SFn | None, ShapeRc]:
    """
    Synthesize shape function S from training examples.

    Contract (01_engineering_spec.md §3):
    "Synthesize constraints; choose the least branch/integers that fit all trainings"

    Contract (02_determinism_addendum.md §1.1):
    Ordering key: (branch_byte, params_bytes, R, C)

    Contract (WO-02 reconciled patch):
    - Library stays total: never raises by default (fail_fast=False)
    - Returns (None, diagnostic_rc) for SHAPE_CONTRADICTION
    - Harness enforces fail-fast at operational level

    Registered COUNT qualifiers: {"q_rows", "q_cols", "q_hw_bilinear", "q_wh_bilinear", "q_components"}
    Only registered qualifiers may be used.

    Args:
        train_pairs: [(train_id, (H,W), (R',C')), ...]
        presented_inputs: [(train_id, G), ...] - optional, for PERIOD and q_components
        fail_fast: if True, raise on SHAPE_CONTRADICTION (for special use cases)

    Returns:
        (S_fn, ShapeRc): shape function and receipt
        Note: S_fn may be None for q_components or SHAPE_CONTRADICTION

    Raises:
        ValueError: if unregistered COUNT qualifier used (contract violation)
        ValueError: if fail_fast=True and SHAPE_CONTRADICTION (special case only)
    """
    # Registered qualifiers (frozen per WO-02 reconciled patch)
    REGISTERED_COUNT_QUALIFIERS = {"q_rows", "q_cols", "q_hw_bilinear", "q_wh_bilinear", "q_components"}

    candidates: list[tuple[str, SFn | None, bytes, dict]] = []

    # Diagnostic tracking for failure receipts (enhanced: skip vs fail)
    aff_reason = None
    per_reason = None
    count_reason = None
    frame_reason = None

    # Try AFFINE (standard)
    aff = _fit_affine(train_pairs)
    if aff is not None:
        candidates.append(("A",) + aff)
    else:
        aff_reason = "no integer solution for all trainings"

    # Try AFFINE (rational floor) - WO-02 extension
    aff_rational = _fit_affine_rational_floor(train_pairs, max_d=MAX_DENOMINATOR_AFFINE_RATIONAL)
    if aff_rational is not None:
        candidates.append(("A",) + aff_rational)
        # If standard AFFINE failed but rational succeeded, update diagnostic
        if aff_reason is not None:
            aff_reason = "standard AFFINE failed, rational AFFINE succeeded"

    # Try PERIOD (requires grids)
    if presented_inputs is None:
        per_reason = "skipped_no_presented_inputs"
    else:
        period = _fit_period(train_pairs, presented_inputs)
        if period is not None:
            candidates.append(("P",) + period)
        else:
            per_reason = "no_period_detected"

    # Try COUNT (q_rows - registered qualifier)
    count = _fit_count_rows(train_pairs)
    if count is not None:
        candidates.append(("C",) + count)

    # Try COUNT (q_cols - registered qualifier)
    count_cols = _fit_count_cols(train_pairs)
    if count_cols is not None:
        candidates.append(("C",) + count_cols)

    # Try COUNT (q_hw_bilinear - registered qualifier)
    count_hw_bilinear = _fit_count_hw_bilinear(train_pairs)
    if count_hw_bilinear is not None:
        candidates.append(("C",) + count_hw_bilinear)

    # Try COUNT (q_wh_bilinear - new registered qualifier)
    count_wh_bilinear = _fit_count_wh_bilinear(train_pairs)
    if count_wh_bilinear is not None:
        candidates.append(("C",) + count_wh_bilinear)

    # Try COUNT (q_components - new registered qualifier)
    count_comp_result = _fit_count_components(train_pairs, presented_inputs)
    if count_comp_result is not None:
        # Special handling: S_fn is None, need to unpack differently
        S_fn, params, extras, qs = count_comp_result
        candidates.append(("C", S_fn, params, extras))

    # Set count_reason if no COUNT qualifier succeeded
    if not any(branch == "C" for branch, _, _, _ in candidates):
        count_reason = "no registered qualifier fits"

    # Try FRAME
    frame = _fit_frame(train_pairs)
    if frame is not None:
        candidates.append(("F",) + frame)
    else:
        frame_reason = "not_constant_output"

    # Validate COUNT qualifiers (fail-closed on unregistered qualifiers)
    for branch, _, _, extras in candidates:
        if branch == "C":
            qual_id = extras.get("qual_id")
            if qual_id not in REGISTERED_COUNT_QUALIFIERS:
                raise ValueError(
                    f"CONTRACT_VIOLATION: Unregistered COUNT qualifier '{qual_id}'. "
                    f"Registered qualifiers: {REGISTERED_COUNT_QUALIFIERS}"
                )

    # Fail-closed: no family fits (WO-02Z)
    if not candidates:
        # Return None for S_fn and diagnostic receipt with status="NONE"
        rc = ShapeRc(
            status="NONE",  # WO-02Z: explicit None instead of contradiction
            branch_byte=None,
            params_bytes_hex=None,
            R=None,
            C=None,
            verified_train_ids=[],  # Empty when no fit
            extras={
                "shape_fit": "NONE",
                "shape_source": "engine",  # WO-02Z: marks deferral to engines
            },
            height_fit=None,
            width_fit=None,
            attempts=[  # WO-02Z: log rejected families
                {"family": "AFFINE", "reason": aff_reason or "unknown"},
                {"family": "PERIOD", "reason": per_reason or "unknown"},
                {"family": "COUNT", "reason": count_reason or "unknown"},
                {"family": "FRAME", "reason": frame_reason or "unknown"},
            ]
        )

        # Optional fail-fast for special use cases (default: library stays total)
        if fail_fast:
            raise ValueError(f"SHAPE_CONTRADICTION: status=NONE, defer to engines. Attempts: {rc.attempts}")

        return None, rc

    # Select by lex-min on (branch_byte, params_bytes)
    # Note: R, C will be filled by caller after applying to test
    def key(c: tuple[str, SFn | None, bytes, dict]) -> tuple[str, bytes]:
        branch, _, params, _ = c
        return (branch, params)

    branch, S, params_bytes, extras = min(candidates, key=key)

    # FIX: Validate test dimensions (fail-closed on R<=0 or C<=0)
    if test_shape is not None and S is not None:
        test_H, test_W = test_shape
        R_test, C_test = apply_shape(S, test_H, test_W)

        if R_test <= 0 or C_test <= 0:
            # Reject this Shape S (INVALID_DIMENSIONS) - treat as NONE (WO-02Z)
            rc = ShapeRc(
                status="NONE",  # WO-02Z: defer to engines
                branch_byte=None,
                params_bytes_hex=None,
                R=None,
                C=None,
                verified_train_ids=[],
                extras={
                    "shape_fit": "INVALID_DIMENSIONS",
                    "shape_source": "engine",
                    "reason": f"Shape S returns ({R_test}, {C_test}) for test input {test_shape}",
                    "branch_attempted": branch,
                    "params_hex_attempted": params_bytes.hex(),
                },
                height_fit=None,
                width_fit=None,
                attempts=[{"family": branch, "reason": f"returns invalid dimensions ({R_test}, {C_test})"}]
            )

            if fail_fast:
                raise ValueError(f"INVALID_DIMENSIONS: {rc.extras}")

            return None, rc

    # Create receipt (R, C, verified_train_ids will be filled by caller) (WO-02Z)
    rc = ShapeRc(
        status="OK",  # WO-02Z: provably fits all trainings
        branch_byte=branch,
        params_bytes_hex=params_bytes.hex(),
        R=-1,  # placeholder (filled by caller after applying to test)
        C=-1,  # placeholder
        verified_train_ids=[],  # placeholder (filled by caller)
        extras=extras,
        height_fit=None,  # TODO WO-02Z: add per-axis proof tables in future iteration
        width_fit=None,  # TODO WO-02Z: add per-axis proof tables in future iteration
        attempts=None  # Success case: no failed attempts
    )

    return S, rc


def serialize_shape(shape_rc: ShapeRc) -> tuple[str, str, dict] | None:
    """
    Extract serialized params from ShapeRc for reuse.

    Contract (WO-02S, WO-11 gap analysis):
    Makes the WO-11 contract explicit: "reuse S from WO-02 serialized params".

    This is a trivial extraction helper that pulls the frozen encoding from
    ShapeRc. The encoding is already deterministic (created during WO-02 fit).

    Contract (WO-02Z):
    Returns None if shape_rc.status == "NONE" (no shape to serialize)

    Args:
        shape_rc: ShapeRc from WO-02 fit

    Returns:
        (branch_byte, params_bytes_hex, extras): Serialized params suitable
        for deserialize_shape(), or None if status="NONE"

    Mathematical property:
        deserialize(serialize(shape_rc)) ≡ original S function
        (isomorphism: serialize and deserialize are inverse operations)
    """
    # WO-02Z: Handle status="NONE" case
    if shape_rc.status == "NONE" or shape_rc.branch_byte is None:
        return None

    return (shape_rc.branch_byte, shape_rc.params_bytes_hex, shape_rc.extras)


def deserialize_shape(branch_byte: str, params_hex: str, extras: dict) -> SFn:
    """
    Deserialize Shape S from frozen params and branch type.

    Contract (WO-05 fix): Reuse WO-02 Shape S instead of re-synthesizing.

    Args:
        branch_byte: 'A', 'P', 'C', 'F'
        params_hex: hex-encoded parameter bytes
        extras: metadata dict (pr_lcm/pc_lcm for P, qual_id/coeffs for C, etc.)

    Returns:
        S: shape function callable S(H, W) -> (R, C)

    Raises:
        ValueError: if branch_byte unknown or params invalid
    """
    from .bytes import unframe_params

    # Decode params from hex
    params_bytes = bytes.fromhex(params_hex)

    if branch_byte == "A":
        # AFFINE: either standard <4><a><b><c><d> or rational <6><d1><a><b><d2><c><e>
        params = unframe_params(params_bytes, signed=True)

        if len(params) == 4:
            # Standard AFFINE: R = aH + b, C = cW + d
            a, b, c, d = params

            def S(H: int, W: int) -> tuple[int, int]:
                return (a * H + b, c * W + d)

        elif len(params) == 6:
            # Rational AFFINE: R = floor((a·H+b)/d1), C = floor((c·W+e)/d2)
            d1, a, b, d2, c, e = params

            def S(H: int, W: int) -> tuple[int, int]:
                R = (a * H + b) // d1
                C = (c * W + e) // d2
                return (R, C)

        else:
            raise ValueError(f"AFFINE branch expects 4 or 6 params, got {len(params)}")

        return S

    elif branch_byte == "P":
        # PERIOD: <2><kr><kc> + extras {row_periods_lcm, col_periods_lcm}
        params = unframe_params(params_bytes, signed=False)
        if len(params) != 2:
            raise ValueError(f"PERIOD branch expects 2 params, got {len(params)}")

        kr, kc = params
        pr_lcm = extras.get("row_periods_lcm", 1)
        pc_lcm = extras.get("col_periods_lcm", 1)

        def S(H: int, W: int) -> tuple[int, int]:
            return (kr * max(1, pr_lcm), kc * max(1, pc_lcm))

        return S

    elif branch_byte == "C":
        # COUNT: <4><a1><b1><a2><b2> + extras {coeffs, qual_id}
        # For q_components: need to recompute q per input (special handling in caller)
        # For other qualifiers (q_rows, q_hw_bilinear, etc): coefficients are in extras
        qual_id = extras.get("qual_id")

        if qual_id == "q_components":
            # Special case: q_components requires computing q(H,W) from actual grid
            # Return None to signal caller must handle specially
            # Caller will use coeffs from extras and compute q from grid
            raise ValueError(
                "q_components cannot be deserialized without grid context. "
                "Caller must handle q_components specially using extras['coeffs']."
            )

        # For other qualifiers, coeffs are in extras
        coeffs = extras.get("coeffs")
        if coeffs is None:
            # Fallback: try to decode from params_bytes
            params = unframe_params(params_bytes, signed=True)
            if len(params) == 4:
                coeffs = tuple(params)
            elif len(params) == 5:
                coeffs = tuple(params)
            else:
                raise ValueError(f"COUNT branch expects 4 or 5 params, got {len(params)}")

        if len(coeffs) == 4:
            a1, b1, a2, b2 = coeffs

            if qual_id == "q_rows":
                # R = a1·H + b1, C = a2·H + b2
                def S(H: int, W: int) -> tuple[int, int]:
                    return (a1 * H + b1, a2 * H + b2)

            elif qual_id == "q_cols":
                # R = a1·W + b1, C = a2·W + b2
                def S(H: int, W: int) -> tuple[int, int]:
                    return (a1 * W + b1, a2 * W + b2)

            else:
                raise ValueError(f"Unknown COUNT qualifier with 4 coeffs: {qual_id}")

        elif len(coeffs) == 5:
            if qual_id == "q_hw_bilinear":
                # R = a1·H + b1, C = c2·W + d2·H + e2
                a1, b1, c2, d2, e2 = coeffs

                def S(H: int, W: int) -> tuple[int, int]:
                    return (a1 * H + b1, c2 * W + d2 * H + e2)

            elif qual_id == "q_wh_bilinear":
                # R = a1·W + b1·H + c1, C = a2·W + b2
                a1, b1, c1, a2, b2 = coeffs

                def S(H: int, W: int) -> tuple[int, int]:
                    return (a1 * W + b1 * H + c1, a2 * W + b2)

            else:
                raise ValueError(f"Unknown COUNT qualifier with 5 coeffs: {qual_id}")

        else:
            raise ValueError(f"COUNT coeffs must have 4 or 5 elements, got {len(coeffs)}")

        return S

    elif branch_byte == "F":
        # FRAME: <2><R0><C0> (constant output)
        params = unframe_params(params_bytes, signed=False)
        if len(params) != 2:
            raise ValueError(f"FRAME branch expects 2 params, got {len(params)}")

        R0, C0 = params

        def S(H: int, W: int) -> tuple[int, int]:
            return (R0, C0)

        return S

    else:
        raise ValueError(f"Unknown branch_byte: {branch_byte}")


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
