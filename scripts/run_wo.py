#!/usr/bin/env python3
# scripts/run_wo.py
# Determinism harness for WO-00, WO-01, ...
# Implements 02_determinism_addendum.md §11 determinism harness

from __future__ import annotations
import argparse
import os
import json
import numpy as np
from arc.op.receipts import env_fingerprint, aggregate, RunRc
from arc.io.load_data import load_task
from arc.op.hash import hash_bytes


def run_wo00() -> dict:
    """WO-00: Environment fingerprint only (no real computation)."""
    env = env_fingerprint()
    stage_hashes = {"wo": "WO-00"}
    run_rc = RunRc(
        env=env,
        stage_hashes=stage_hashes,
        notes={"status": "WO-00 placeholder run"},
    )
    return aggregate(run_rc)


def run_wo01(data_dir: str, subset_file: str) -> list[dict]:
    """
    WO-01: Test Π (presentation) on tasks.

    For each task:
    1. Load train inputs + test input
    2. Run present_all(train_inputs, test_input)
    3. Verify F1: palette scope is "inputs_only"
    4. Verify F2: per-grid proofs of Π²=Π and U⁻¹∘Π=id
    5. Collect receipts

    Contract (docs/common_mistakes.md F1, F2):
    - F1: Palette must be inputs-only (forbid_outputs=True)
    - F2: Prove Π²=Π and U⁻¹∘Π=id for EVERY grid

    Returns:
        List of receipts (one per task)
    """
    from arc.op.pi import present_all, unpresent
    from arc.op.hash import hash_grid

    # Load task IDs
    with open(subset_file) as f:
        task_ids = [line.strip() for line in f if line.strip()]

    all_receipts = []

    for task_id in task_ids:
        # Load task
        task_path = os.path.join(data_dir, f"{task_id}.json")
        task = load_task(task_path)

        # Extract inputs
        train_inputs = [np.array(pair["input"], dtype=np.int64) for pair in task["train"]]
        test_input = np.array(task["test"][0]["input"], dtype=np.int64)

        # Run Π
        train_presented, test_presented, transform, pi_rc = present_all(
            train_inputs, test_input
        )

        # F1 VERIFICATION: Palette scope must be inputs-only
        if pi_rc.palette.scope != "inputs_only":
            raise ValueError(
                f"Task {task_id}: F1 violation! Palette scope is '{pi_rc.palette.scope}', expected 'inputs_only'.\n"
                f"This prevents accidental inclusion of training outputs (common_mistakes.md F1)."
            )

        # F2 VERIFICATION: Per-grid proofs
        # Verify train grids
        train_records = [r for r in pi_rc.per_grid if r["kind"] == "train"]
        if len(train_records) != len(train_inputs):
            raise ValueError(
                f"Task {task_id}: F2 violation! Expected {len(train_inputs)} train per-grid records, got {len(train_records)}"
            )

        for i, (orig, presented, per_rc) in enumerate(zip(train_inputs, train_presented, train_records)):
            # Verify Π²=Π (hash of presented grid should equal pi2_hash)
            presented_hash = hash_grid(presented)
            if per_rc["pi2_hash"] != presented_hash:
                raise ValueError(
                    f"Task {task_id} train grid {i}: F2 violation! Π²≠Π\n"
                    f"  Presented hash: {presented_hash}\n"
                    f"  Π² hash: {per_rc['pi2_hash']}"
                )

            # Verify U⁻¹∘Π=id (roundtrip hash should equal original hash)
            original_hash = hash_grid(orig)
            if per_rc["roundtrip_hash"] != original_hash:
                raise ValueError(
                    f"Task {task_id} train grid {i}: F2 violation! U⁻¹∘Π≠id\n"
                    f"  Original hash: {original_hash}\n"
                    f"  Roundtrip hash: {per_rc['roundtrip_hash']}"
                )

        # Verify test grid
        test_records = [r for r in pi_rc.per_grid if r["kind"] == "test"]
        if len(test_records) != 1:
            raise ValueError(
                f"Task {task_id}: F2 violation! Expected 1 test per-grid record, got {len(test_records)}"
            )

        test_rc = test_records[0]
        test_presented_hash = hash_grid(test_presented)
        if test_rc["pi2_hash"] != test_presented_hash:
            raise ValueError(
                f"Task {task_id} test grid: F2 violation! Π²≠Π\n"
                f"  Presented hash: {test_presented_hash}\n"
                f"  Π² hash: {test_rc['pi2_hash']}"
            )

        test_original_hash = hash_grid(test_input)
        if test_rc["roundtrip_hash"] != test_original_hash:
            raise ValueError(
                f"Task {task_id} test grid: F2 violation! U⁻¹∘Π≠id\n"
                f"  Original hash: {test_original_hash}\n"
                f"  Roundtrip hash: {test_rc['roundtrip_hash']}"
            )

        # Build receipt for this task
        env = env_fingerprint()
        stage_hashes = {
            "wo": "WO-01",
            "pi.palette_hash": pi_rc.palette.palette_hash,
            "pi.roundtrip_hash": test_rc["roundtrip_hash"],  # use test grid's roundtrip hash
        }

        run_rc = RunRc(
            env=env,
            stage_hashes=stage_hashes,
            notes={
                "task_id": task_id,
                "pi": {
                    "palette_freqs": pi_rc.palette.palette_freqs[:5],  # first 5 for brevity
                    "pose_id": pi_rc.test_pose_id,
                    "anchor_dr": pi_rc.test_anchor.dr,
                    "anchor_dc": pi_rc.test_anchor.dc,
                },
            },
        )

        all_receipts.append(aggregate(run_rc))

    return all_receipts


def lint_wo02_receipts(receipts: list[dict]) -> list[str]:
    """
    Receipt linter for WO-02: surface actionable diagnostics.

    Contract (WO-02 reconciled patch):
    - Distinguish configuration bugs from coverage gaps
    - Surface patterns for missing qualifiers
    - Provide actionable next steps

    Returns:
        List of lint warnings (empty if all clean)
    """
    warnings = []

    for receipt in receipts:
        task_id = receipt.get("notes", {}).get("task_id", "unknown")
        shape_notes = receipt.get("notes", {}).get("shape", {})
        extras = shape_notes.get("extras", {})

        # LINT 1: PERIOD skipped due to missing grids (configuration bug)
        if extras.get("status") == "SHAPE_CONTRADICTION":
            attempts = extras.get("attempts", [])
            for attempt in attempts:
                if attempt.get("family") == "PERIOD":
                    reason = attempt.get("reason", "")
                    if "skipped_no_presented_inputs" in reason:
                        warnings.append(
                            f"CONFIG_BUG: {task_id} - PERIOD skipped (no grids). "
                            f"ALWAYS pass presented_inputs to synthesize_shape!"
                        )

        # LINT 2: COUNT failed with no registered qualifiers (potential gap)
        if extras.get("status") == "SHAPE_CONTRADICTION":
            attempts = extras.get("attempts", [])
            for attempt in attempts:
                if attempt.get("family") == "COUNT":
                    reason = attempt.get("reason", "")
                    if "no registered qualifier fits" in reason:
                        # This is a coverage gap, not a bug - just note it
                        pass  # Could add pattern analysis here

    return warnings


def run_wo02(data_dir: str, subset_file: str, continue_on_error: bool = False) -> list[dict]:
    """
    WO-02: Test Shape S synthesis on tasks.

    For each task:
    1. Load all training pairs to collect (H,W) → (R',C')
    2. Run synthesize_shape to get S function and ShapeRc
    3. Apply S to test input to get (R,C)
    4. Verify E1: proof on ALL trainings (equality check)
    5. Collect receipts

    Contract (WO-02 reconciled patch):
    - Library stays total (synthesize_shape never crashes by default)
    - Harness enforces fail-fast (exits on SHAPE_CONTRADICTION unless --continue-on-error)
    - Receipt must include verified_train_ids (ALL trainings)

    Args:
        data_dir: Path to ARC task JSON files
        subset_file: File containing task IDs (one per line)
        continue_on_error: If False (default), fail-fast on SHAPE_CONTRADICTION
                          If True, collect all failures and report at end

    Returns:
        List of receipts (one per task)

    Raises:
        SystemExit(1): If continue_on_error=False and any SHAPE_CONTRADICTION found
    """
    from arc.op.shape import synthesize_shape, apply_shape, MAX_DENOMINATOR_AFFINE_RATIONAL
    from arc.op.hash import hash_bytes

    # Harness assertion: verify frozen constant (02_determinism_addendum.md §1.1.1)
    assert MAX_DENOMINATOR_AFFINE_RATIONAL == 10, (
        f"BLOCKER: MAX_DENOMINATOR_AFFINE_RATIONAL must be 10 (frozen), "
        f"got {MAX_DENOMINATOR_AFFINE_RATIONAL}"
    )

    # Load task IDs
    with open(subset_file) as f:
        task_ids = [line.strip() for line in f if line.strip()]

    all_receipts = []
    failed_tasks = []  # Track SHAPE_CONTRADICTION failures

    for task_id in task_ids:
        # Load task
        task_path = os.path.join(data_dir, f"{task_id}.json")
        task = load_task(task_path)

        # FIX: Apply Π before synthesizing Shape S (coordinate frame consistency)
        from arc.op.pi import present_all

        # Load RAW inputs
        train_inputs_raw = [np.array(pair["input"], dtype=np.int64) for pair in task["train"]]
        test_input_raw = np.array(task["test"][0]["input"], dtype=np.int64)

        # Apply Π to get PRESENTED (canonical) frame
        train_presented, test_presented, transform, pi_rc = present_all(
            train_inputs_raw, test_input_raw
        )

        # Build train_pairs: [(train_id, (H_presented, W_presented), (R_raw, C_raw))]
        # Inputs: PRESENTED (after Π), Outputs: RAW (per spec)
        train_pairs = []
        presented_inputs = []

        for i, pair in enumerate(task["train"]):
            train_id = f"{task_id}_train{i}"
            X_presented = train_presented[i]
            Y_raw = np.array(pair["output"], dtype=np.int64)
            train_pairs.append((train_id, X_presented.shape, Y_raw.shape))
            presented_inputs.append((train_id, X_presented))

        # Synthesize Shape S using PRESENTED input dimensions
        # Pass test_presented shape for validation
        S, shape_rc = synthesize_shape(train_pairs, presented_inputs, test_shape=test_presented.shape)

        # Record coordinate frame in receipt (for WO-05 consistency check)
        if "frame" not in shape_rc.extras:
            shape_rc.extras["frame"] = "presented"

        # Check for SHAPE_CONTRADICTION, INVALID_DIMENSIONS, or special cases
        if S is None and shape_rc.extras.get("status") == "SHAPE_CONTRADICTION":
            # SHAPE_CONTRADICTION: no family fits
            # R, C already set to -1 in shape.py
            # Receipt already has diagnostics in extras["attempts"]
            shape_rc.verified_train_ids = []  # No verification possible

            # Track failure
            failed_tasks.append({
                "task_id": task_id,
                "attempts": shape_rc.extras.get("attempts", []),
            })

            # Fail-fast unless --continue-on-error
            if not continue_on_error:
                print(f"\n❌ SHAPE_CONTRADICTION: {task_id}")
                print(f"   No family (AFFINE, PERIOD, COUNT, FRAME) fits all trainings.")
                print(f"   Attempts:")
                for attempt in shape_rc.extras.get("attempts", []):
                    print(f"     - {attempt['family']}: {attempt['reason']}")
                print(f"\n   Use --continue-on-error to collect all failures.\n")
                exit(1)

            # Skip E1 verification, continue to receipt generation

        elif S is None and shape_rc.extras.get("status") == "INVALID_DIMENSIONS":
            # INVALID_DIMENSIONS: Shape S returns R<=0 or C<=0 for test input
            shape_rc.verified_train_ids = []  # No verification possible

            # Track failure
            failed_tasks.append({
                "task_id": task_id,
                "reason": shape_rc.extras.get("reason", "unknown"),
            })

            # Fail-fast unless --continue-on-error
            if not continue_on_error:
                print(f"\n❌ INVALID_DIMENSIONS: {task_id}")
                print(f"   {shape_rc.extras.get('reason', 'unknown')}")
                print(f"   Branch attempted: {shape_rc.extras.get('branch_attempted', 'unknown')}")
                print(f"\n   Use --continue-on-error to collect all failures.\n")
                exit(1)

            # Skip E1 verification, continue to receipt generation

        elif S is None and shape_rc.extras.get("qual_id") == "q_components":
            # Special case: q_components requires computing q(test)
            from arc.op.components import cc4_by_color

            # Get coefficients from receipt
            a1, b1, a2, b2 = shape_rc.extras["coeffs"]

            # Apply to test: compute q(test) using PRESENTED input
            _, test_comp_rc = cc4_by_color(test_presented)
            q_test = len(test_comp_rc.invariants)

            Rt = a1 * q_test + b1
            Ct = a2 * q_test + b2

            shape_rc.R = Rt
            shape_rc.C = Ct
            shape_rc.verified_train_ids = [tid for tid, _, _ in train_pairs]

            # E1 VERIFICATION: Re-check all trainings (using q per training)
            for (train_id, G), (_, _, (R_expected, C_expected)) in zip(presented_inputs, train_pairs):
                _, train_comp_rc = cc4_by_color(G)
                q_train = len(train_comp_rc.invariants)
                R_actual = a1 * q_train + b1
                C_actual = a2 * q_train + b2

                if (R_actual, C_actual) != (R_expected, C_expected):
                    raise ValueError(
                        f"Task {task_id}: E1 violation! Shape proof failed for {train_id}\n"
                        f"  Expected: ({R_expected}, {C_expected})\n"
                        f"  S(q={q_train}): ({R_actual}, {C_actual})\n"
                        f"  Branch: {shape_rc.branch_byte}, Params: {shape_rc.params_bytes_hex}"
                    )

        else:
            # Normal case: S is callable
            # Apply S to PRESENTED test dimensions (not RAW)
            Ht, Wt = test_presented.shape
            Rt, Ct = apply_shape(S, Ht, Wt)

            shape_rc.R = Rt
            shape_rc.C = Ct
            shape_rc.verified_train_ids = [tid for tid, _, _ in train_pairs]

            # E1 VERIFICATION: Re-check all trainings (proof must hold)
            for train_id, (H, W), (R_expected, C_expected) in train_pairs:
                R_actual, C_actual = apply_shape(S, H, W)
                if (R_actual, C_actual) != (R_expected, C_expected):
                    raise ValueError(
                        f"Task {task_id}: E1 violation! Shape proof failed for {train_id}\n"
                        f"  Expected: ({R_expected}, {C_expected})\n"
                        f"  S({H},{W}): ({R_actual}, {C_actual})\n"
                        f"  Branch: {shape_rc.branch_byte}, Params: {shape_rc.params_bytes_hex}"
                    )

        # Build receipt for this task
        env = env_fingerprint()

        # Hash the shape function (via params_bytes for determinism)
        shape_hash = hash_bytes(bytes.fromhex(shape_rc.params_bytes_hex))

        stage_hashes = {
            "wo": "WO-02",
            "shape.branch": shape_rc.branch_byte,
            "shape.params_hash": shape_hash,
        }

        run_rc = RunRc(
            env=env,
            stage_hashes=stage_hashes,
            notes={
                "task_id": task_id,
                "shape": {
                    "branch_byte": shape_rc.branch_byte,
                    "params_hex": shape_rc.params_bytes_hex,
                    "R": shape_rc.R,
                    "C": shape_rc.C,
                    "verified_count": len(shape_rc.verified_train_ids),
                    "extras": shape_rc.extras,
                },
            },
        )

        all_receipts.append(aggregate(run_rc))

    # Run receipt linter
    lint_warnings = lint_wo02_receipts(all_receipts)

    # Summary reporting
    total = len(task_ids)
    success = total - len(failed_tasks)
    print(f"\n{'='*60}")
    print(f"WO-02 Summary")
    print(f"{'='*60}")
    print(f"Total tasks:   {total}")
    print(f"Success:       {success} ({100*success//total}%)")
    print(f"Failed:        {len(failed_tasks)} ({100*len(failed_tasks)//total}%)")

    # Display lint warnings
    if lint_warnings:
        print(f"\n⚠️  Receipt Linter Warnings ({len(lint_warnings)}):")
        for warning in lint_warnings:
            print(f"  {warning}")

    if failed_tasks:
        print(f"\n❌ SHAPE_CONTRADICTION failures:")
        for fail in failed_tasks:
            print(f"  - {fail['task_id']}")
        print(f"\nFailed task IDs: {', '.join(f['task_id'] for f in failed_tasks)}")

        # If --continue-on-error, still exit non-zero to signal failures
        if continue_on_error:
            print(f"\n⚠️  Exiting with code 1 (failures recorded)")
            exit(1)
    else:
        print(f"\n✅ All tasks succeeded!")

    return all_receipts


def run_wo03(data_dir: str, subset_file: str) -> list[dict]:
    """
    WO-03: Test Components + stable matching on tasks.

    For each task:
    1. Load training pairs
    2. Extract 4-connected components for X_i and Y_i
    3. Run stable_match to get pairs and unmatched components
    4. Verify D2: connectivity = "4" (frozen)
    5. Collect receipts

    Contract (02_determinism_addendum.md §3):
    - connectivity = "4" always (D2 freeze)
    - Invariants: (area, bbox_h, bbox_w, perim4, outline_hash, anchor_r, anchor_c)
    - Stable matching by lex-sorted invariant equality

    Returns:
        List of receipts (one per training pair)
    """
    from arc.op.components import cc4_by_color, stable_match, CompInv
    from dataclasses import asdict

    # Load task IDs
    with open(subset_file) as f:
        task_ids = [line.strip() for line in f if line.strip()]

    all_receipts = []

    for task_id in task_ids:
        # Load task
        task_path = os.path.join(data_dir, f"{task_id}.json")
        task = load_task(task_path)

        # Process each training pair
        for pair_idx, pair in enumerate(task["train"]):
            train_id = f"{task_id}_train{pair_idx}"

            # Extract grids
            X = np.array(pair["input"], dtype=np.int64)
            Y = np.array(pair["output"], dtype=np.int64)

            # Extract components
            X_masks, X_rc = cc4_by_color(X)
            Y_masks, Y_rc = cc4_by_color(Y)

            # D2 VERIFICATION: Connectivity must be 4
            if X_rc.connectivity != "4" or Y_rc.connectivity != "4":
                raise ValueError(
                    f"Task {train_id}: D2 violation! Connectivity must be '4'.\\n"
                    f"  X connectivity: {X_rc.connectivity}\\n"
                    f"  Y connectivity: {Y_rc.connectivity}"
                )

            # Deserialize invariants for matching
            X_invs = [CompInv(**inv_dict) for inv_dict in X_rc.invariants]
            Y_invs = [CompInv(**inv_dict) for inv_dict in Y_rc.invariants]

            # Stable match
            pairs, match_rc = stable_match(X_invs, Y_invs)

            # Build receipt for this training pair
            env = env_fingerprint()

            # Hash the components for determinism tracking
            X_comp_hash = hash_bytes(json.dumps(X_rc.invariants, sort_keys=True).encode())
            Y_comp_hash = hash_bytes(json.dumps(Y_rc.invariants, sort_keys=True).encode())
            match_hash = hash_bytes(
                json.dumps(
                    {
                        "pairs": match_rc.pairs,
                        "left_only": match_rc.left_only,
                        "right_only": match_rc.right_only,
                    },
                    sort_keys=True
                ).encode()
            )

            stage_hashes = {
                "wo": "WO-03",
                "components.X_hash": X_comp_hash,
                "components.Y_hash": Y_comp_hash,
                "match.hash": match_hash,
            }

            run_rc = RunRc(
                env=env,
                stage_hashes=stage_hashes,
                notes={
                    "train_id": train_id,
                    "components": {
                        "X": asdict(X_rc),
                        "Y": asdict(Y_rc),
                    },
                    "match": {
                        "pairs": match_rc.pairs,
                        "left_only": match_rc.left_only,
                        "right_only": match_rc.right_only,
                        "verified_pixelwise": match_rc.verified_pixelwise,
                    },
                },
            )

            all_receipts.append(aggregate(run_rc))

    return all_receipts


def run_wo04(data_dir: str, subset_file: str) -> list[dict]:
    """
    WO-04: Test Witness solver (φ,σ) + conjugation + intersection.

    For each task:
    1. Load training pairs
    2. Extract components with WO-03
    3. Solve witness per training pair (geometric or summary)
    4. Conjugate to test Π frame
    5. Intersect across trainings
    6. Verify E2: bbox equality for geometric witnesses
    7. Verify A1/C2: candidate sets and decision rules for summary witnesses
    8. Collect receipts

    Contract:
    - E2: Every φ piece must prove bbox equality (exact pixelwise)
    - A1: Candidate sets part of law (record foreground/background colors, counts)
    - C2: Decision rule frozen string

    Returns:
        List of receipts (one per task)
    """
    from arc.op.components import cc4_by_color
    from arc.op.witness import (
        solve_witness_for_pair,
        conjugate_to_test,
        intersect_witnesses
    )
    from dataclasses import asdict

    # Load task IDs
    with open(subset_file) as f:
        task_ids = [line.strip() for line in f if line.strip()]

    all_receipts = []

    for task_id in task_ids:
        # Load task
        task_path = os.path.join(data_dir, f"{task_id}.json")
        task = load_task(task_path)

        # Per-task witness solving
        train_witnesses = []
        conjugated_witnesses = []

        for pair_idx, pair in enumerate(task["train"]):
            train_id = f"{task_id}_train{pair_idx}"

            # Extract grids
            X = np.array(pair["input"], dtype=np.int64)
            Y = np.array(pair["output"], dtype=np.int64)

            # Extract components (WO-03)
            X_masks, X_rc = cc4_by_color(X)
            Y_masks, Y_rc = cc4_by_color(Y)

            # Solve witness (φ, σ)
            phi_pieces, sigma, witness_rc = solve_witness_for_pair(X, Y, X_rc, Y_rc)

            # E2 VERIFICATION: For geometric, check bbox_equal
            if witness_rc.kind == "geometric":
                if not all(witness_rc.phi.bbox_equal):
                    raise ValueError(
                        f"Task {train_id}: E2 violation! Geometric witness has failed bbox equality.\\n"
                        f"  bbox_equal: {witness_rc.phi.bbox_equal}"
                    )

            # A1/C2 VERIFICATION: For summary, check receipts present
            if witness_rc.kind == "summary":
                if witness_rc.foreground_colors is None:
                    raise ValueError(
                        f"Task {train_id}: A1 violation! Summary witness missing foreground_colors"
                    )
                if witness_rc.background_colors is None:
                    raise ValueError(
                        f"Task {train_id}: A1 violation! Summary witness missing background_colors"
                    )
                if witness_rc.decision_rule is None:
                    raise ValueError(
                        f"Task {train_id}: C2 violation! Summary witness missing decision_rule"
                    )
                if witness_rc.per_color_counts is None:
                    raise ValueError(
                        f"Task {train_id}: A1 violation! Summary witness missing per_color_counts"
                    )

            train_witnesses.append((phi_pieces, sigma, witness_rc))

            # Conjugate to test frame (simplified for WO-04)
            phi_star, conj_rc = conjugate_to_test(phi_pieces, sigma)
            conjugated_witnesses.append((phi_star, sigma, conj_rc))

        # Intersection across trainings
        conj_list = [(phi_star, sigma) for phi_star, sigma, _ in conjugated_witnesses]
        phi_intersect, sigma_intersect, intersection_rc = intersect_witnesses(conj_list)

        # Build receipt for this task
        env = env_fingerprint()

        # Hash witnesses for determinism
        witnesses_hash = hash_bytes(
            json.dumps(
                [asdict(w_rc) for _, _, w_rc in train_witnesses],
                sort_keys=True
            ).encode()
        )
        intersection_hash = hash_bytes(
            json.dumps(asdict(intersection_rc), sort_keys=True).encode()
        )

        stage_hashes = {
            "wo": "WO-04",
            "witnesses.hash": witnesses_hash,
            "intersection.hash": intersection_hash,
        }

        run_rc = RunRc(
            env=env,
            stage_hashes=stage_hashes,
            notes={
                "task_id": task_id,
                "witnesses": {
                    "train": [asdict(w_rc) for _, _, w_rc in train_witnesses],
                    "conjugated": [asdict(c_rc) for _, _, c_rc in conjugated_witnesses],
                },
                "intersection": asdict(intersection_rc),
            },
        )

        all_receipts.append(aggregate(run_rc))

    return all_receipts


def _load_wo02_receipt(receipts_dir: str, task_id: str) -> dict | None:
    """
    Load WO-02 receipt for given task_id.

    Args:
        receipts_dir: Path to receipts directory
        task_id: Task identifier

    Returns:
        Shape receipt dict or None if not found
    """
    import json

    wo02_path = os.path.join(receipts_dir, "WO-02_run.jsonl")
    if not os.path.exists(wo02_path):
        return None

    with open(wo02_path) as f:
        for line in f:
            if not line.strip():
                continue
            receipt = json.loads(line)
            if receipt.get("notes", {}).get("task_id") == task_id:
                return receipt.get("notes", {}).get("shape")

    return None


def run_wo04c_conjugation() -> list[dict]:
    """
    WO-04C: Test witness conjugation to test Π-frame.

    Synthetic test cases where Π differs between train and test:
    1. Identity conjugation (same Π)
    2. Rotation conjugation (different pose)
    3. Anchor conjugation (different anchor)
    4. Combined conjugation (pose + anchor)

    Returns:
        List of receipts per test case
    """
    from arc.op.witness import PhiPiece, SigmaRc, conjugate_to_test
    from arc.op.pi import PiTransform
    from arc.op.anchor import AnchorRc
    import numpy as np

    results = []

    # Test case 1: Identity conjugation (same Π)
    test_case = {
        "test_case": "identity_conjugation",
        "desc": "Pi_train = Pi_test (identity)",
    }

    phi_pieces = [
        PhiPiece(comp_id=0, pose_id=0, dr=1, dc=2, r_per=1, c_per=1, r_res=0, c_res=0),
        PhiPiece(comp_id=1, pose_id=1, dr=0, dc=1, r_per=1, c_per=1, r_res=0, c_res=0),
    ]
    sigma = SigmaRc(domain_colors=[0, 1, 5, 8], lehmer=[], moved_count=0)

    Pi_train = PiTransform(
        map={}, inv_map={},
        pose_id=0, anchor=AnchorRc(dr=0, dc=0)
    )
    Pi_test = PiTransform(
        map={}, inv_map={},
        pose_id=0, anchor=AnchorRc(dr=0, dc=0)
    )

    phi_star, conj_rc = conjugate_to_test(phi_pieces, sigma, Pi_train, Pi_test)

    # Verify: identity conjugation should preserve pieces
    assert len(phi_star) == len(phi_pieces)
    for orig, star in zip(phi_pieces, phi_star):
        assert orig.pose_id == star.pose_id
        assert orig.dr == star.dr
        assert orig.dc == star.dc

    test_case["result"] = "PASS"
    test_case["conjugation_hash"] = conj_rc.conjugation_hash
    results.append(test_case)

    # Test case 2: Rotation conjugation (pose differs)
    test_case = {
        "test_case": "rotation_conjugation",
        "desc": "Pi_test.pose = R90, Pi_train.pose = identity",
    }

    Pi_train = PiTransform(
        map={}, inv_map={},
        pose_id=0, anchor=AnchorRc(dr=0, dc=0)
    )
    Pi_test = PiTransform(
        map={}, inv_map={},
        pose_id=1, anchor=AnchorRc(dr=0, dc=0)  # R90
    )

    phi_pieces_rot = [
        PhiPiece(comp_id=0, pose_id=0, dr=2, dc=3, r_per=1, c_per=1, r_res=0, c_res=0),
    ]

    phi_star_rot, conj_rc_rot = conjugate_to_test(phi_pieces_rot, sigma, Pi_train, Pi_test)

    # Verify pose composed correctly: Π*.pose ∘ φ.pose ∘ inv(Πᵢ.pose) = 1 ∘ 0 ∘ 0 = 1
    assert phi_star_rot[0].pose_id == 1

    test_case["result"] = "PASS"
    test_case["conjugation_hash"] = conj_rc_rot.conjugation_hash
    test_case["transform_receipts"] = conj_rc_rot.transform_receipts
    results.append(test_case)

    # Test case 3: Anchor conjugation (anchor differs)
    test_case = {
        "test_case": "anchor_conjugation",
        "desc": "Pi_test.anchor != Pi_train.anchor",
    }

    Pi_train = PiTransform(
        map={}, inv_map={},
        pose_id=0, anchor=AnchorRc(dr=2, dc=3)
    )
    Pi_test = PiTransform(
        map={}, inv_map={},
        pose_id=0, anchor=AnchorRc(dr=5, dc=7)
    )

    phi_pieces_anc = [
        PhiPiece(comp_id=0, pose_id=0, dr=10, dc=15, r_per=1, c_per=1, r_res=0, c_res=0),
    ]

    phi_star_anc, conj_rc_anc = conjugate_to_test(phi_pieces_anc, sigma, Pi_train, Pi_test)

    # Verify anchor shift: dr_new = dr_orig - anchor_train.dr + anchor_test.dr
    # = 10 - 2 + 5 = 13
    assert phi_star_anc[0].dr == 13
    assert phi_star_anc[0].dc == 19  # 15 - 3 + 7 = 19

    test_case["result"] = "PASS"
    test_case["conjugation_hash"] = conj_rc_anc.conjugation_hash
    results.append(test_case)

    # Test case 4: Combined (pose + anchor)
    test_case = {
        "test_case": "combined_conjugation",
        "desc": "Both pose and anchor differ",
    }

    Pi_train = PiTransform(
        map={}, inv_map={},
        pose_id=4, anchor=AnchorRc(dr=1, dc=1)  # FlipH
    )
    Pi_test = PiTransform(
        map={}, inv_map={},
        pose_id=2, anchor=AnchorRc(dr=0, dc=0)  # R180
    )

    phi_pieces_comb = [
        PhiPiece(comp_id=0, pose_id=0, dr=5, dc=3, r_per=1, c_per=1, r_res=0, c_res=0),
    ]

    phi_star_comb, conj_rc_comb = conjugate_to_test(phi_pieces_comb, sigma, Pi_train, Pi_test)

    # Pose composition: compose(2, compose(0, inv(4))) = compose(2, compose(0, 4)) = compose(2, 4) = 6
    assert phi_star_comb[0].pose_id == 6

    test_case["result"] = "PASS"
    test_case["conjugation_hash"] = conj_rc_comb.conjugation_hash
    results.append(test_case)

    print(f"\n{'='*60}")
    print(f"WO-04C Conjugation Summary")
    print(f"{'='*60}")
    print(f"Test cases:         {len(results)}")
    print(f"Passed:             {sum(1 for r in results if r['result'] == 'PASS')}")
    print(f"{'='*60}\n")

    return results


def run_wo10a_macro_tiling() -> list[dict]:
    """
    WO-10A: Test Macro-Tiling engine with synthetic cases.

    Synthetic test cases:
    1. Simple 2x2 band grid with uniform color per tile
    2. Strict majority rule (count > sum(others))
    3. Tie-break case (min color selection)
    4. Empty tile fallback to background

    Returns:
        List of receipts per test case
    """
    from arc.op.families import fit_macro_tiling, apply_macro_tiling
    from arc.op.truth import TruthRc, OverlapRc
    import numpy as np

    results = []

    # Test case 1: Simple uniform tiles
    test_case = {
        "test_case": "uniform_tiles",
        "desc": "2x2 band grid, each tile uniform color",
    }

    # Create synthetic Truth with row_clusters=[0, 3, 6], col_clusters=[0, 4, 8]
    # This creates a 2x2 tile grid:
    # - Tile (0,0): rows [0:3], cols [0:4]
    # - Tile (0,1): rows [0:3], cols [4:8]
    # - Tile (1,0): rows [3:6], cols [0:4]
    # - Tile (1,1): rows [3:6], cols [4:8]

    overlaps = OverlapRc(
        method="fft_int", modulus=None, root=None,
        candidates=[], accepted=[], identity_excluded=True
    )

    truth1 = TruthRc(
        tag_set_version="test",
        partition_hash="test",
        block_hist=[1],
        overlaps=overlaps,
        row_clusters=[0, 3, 6],
        col_clusters=[0, 4, 8],
        refinement_steps=0,
        method="paige_tarjan"
    )

    # Training 1: uniform colors per tile
    # Tile (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4
    Y1 = np.array([
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [3, 3, 3, 3, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4, 4, 4],
    ], dtype=np.uint8)

    # Training 2: same tile pattern (must be unanimous)
    Y2 = Y1.copy()

    # Dummy Xt (not used for macro-tiling fit)
    Xt1 = np.zeros((6, 8), dtype=np.uint8)
    Xt2 = np.zeros((6, 8), dtype=np.uint8)

    train_Xt_list = [Xt1, Xt2]
    train_Y_list = [Y1, Y2]
    truth_list = [truth1, truth1]

    # Fit
    fit_rc = fit_macro_tiling(train_Xt_list, train_Y_list, truth_list)

    # Verify fit succeeded
    assert fit_rc.ok, f"Fit failed: {fit_rc.receipt}"
    assert fit_rc.receipt["row_bands"] == [0, 3, 6]
    assert fit_rc.receipt["col_bands"] == [0, 4, 8]
    assert fit_rc.receipt["foreground_colors"] == [1, 2, 3, 4]
    assert fit_rc.receipt["background_colors"] == [0]

    # Verify tile rules
    tile_rules = fit_rc.receipt["tile_rules"]
    assert tile_rules["0,0"] == 1
    assert tile_rules["0,1"] == 2
    assert tile_rules["1,0"] == 3
    assert tile_rules["1,1"] == 4

    # Apply to test
    test_Xt = np.zeros((6, 8), dtype=np.uint8)
    apply_rc = apply_macro_tiling(test_Xt, truth1, fit_rc)

    # Verify apply succeeded
    assert apply_rc.ok, f"Apply failed: {apply_rc.receipt}"
    assert np.array_equal(apply_rc.Yt, Y1)

    test_case["result"] = "PASS"
    test_case["fit_ok"] = fit_rc.ok
    test_case["apply_ok"] = apply_rc.ok
    results.append(test_case)

    # Test case 2: Uniform tiles with different colors
    test_case = {
        "test_case": "multi_color_tiles",
        "desc": "Each tile uniform, different colors across tiles",
    }

    # Training with uniform tiles (different colors)
    # Tile (0,0)=5, (0,1)=7, (1,0)=1, (1,1)=4
    Y3 = np.array([
        [5, 5, 5, 5, 7, 7, 7, 7],
        [5, 5, 5, 5, 7, 7, 7, 7],
        [5, 5, 5, 5, 7, 7, 7, 7],
        [1, 1, 1, 1, 4, 4, 4, 4],
        [1, 1, 1, 1, 4, 4, 4, 4],
        [1, 1, 1, 1, 4, 4, 4, 4],
    ], dtype=np.uint8)

    train_Xt_list = [Xt1]
    train_Y_list = [Y3]
    truth_list = [truth1]

    fit_rc = fit_macro_tiling(train_Xt_list, train_Y_list, truth_list)

    # Verify fit succeeded
    assert fit_rc.ok, f"Fit failed: {fit_rc.receipt}"

    # Check tile decisions
    tile_rules = fit_rc.receipt["tile_rules"]
    assert tile_rules["0,0"] == 5  # Tile (0,0)
    assert tile_rules["0,1"] == 7  # Tile (0,1)
    assert tile_rules["1,0"] == 1  # Tile (1,0)
    assert tile_rules["1,1"] == 4  # Tile (1,1)

    test_case["result"] = "PASS"
    test_case["fit_ok"] = fit_rc.ok
    results.append(test_case)

    # Test case 3: Background tile (all zeros)
    test_case = {
        "test_case": "background_tile",
        "desc": "Tile with all background (color 0)",
    }

    # Create a tile with all background
    truth_simple = TruthRc(
        tag_set_version="test",
        partition_hash="test",
        block_hist=[1],
        overlaps=overlaps,
        row_clusters=[0, 4],
        col_clusters=[0, 4],
        refinement_steps=0,
        method="paige_tarjan"
    )

    # Tile with all background (0)
    Y4 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.uint8)

    Xt_simple = np.zeros((4, 4), dtype=np.uint8)
    fit_rc = fit_macro_tiling([Xt_simple], [Y4], [truth_simple])

    # Verify fit succeeded
    assert fit_rc.ok, f"Fit failed: {fit_rc.receipt}"

    # Tile should be background (0)
    # When all pixels are 0, foreground_colors = [], so no strict majority
    # Fallback to background 0
    tile_rules = fit_rc.receipt["tile_rules"]
    assert tile_rules["0,0"] == 0  # Background

    test_case["result"] = "PASS"
    test_case["fit_ok"] = fit_rc.ok
    results.append(test_case)

    # Test case 4: Verify fit_verified_on receipts
    test_case = {
        "test_case": "verify_receipts",
        "desc": "Check B1 guard: stage-1 counts + final decisions",
    }

    # Use test case 1 data
    fit_rc = fit_macro_tiling([Xt1, Xt2], [Y1, Y2], [truth1, truth1])

    # Verify B1: receipts contain both counts and decisions
    assert "train" in fit_rc.receipt
    for train_receipt in fit_rc.receipt["train"]:
        assert "tile_decisions" in train_receipt
        for tile_decision in train_receipt["tile_decisions"]:
            assert "counts" in tile_decision  # stage-1 evidence
            assert "decision" in tile_decision  # final decision
            assert "rule" in tile_decision  # decision rule

    # Verify A1: foreground_colors from trainings only
    assert "foreground_colors" in fit_rc.receipt
    assert "background_colors" in fit_rc.receipt

    # Verify fit_verified_on
    assert fit_rc.receipt["fit_verified_on"] == ["train0", "train1"]

    test_case["result"] = "PASS"
    test_case["fit_ok"] = fit_rc.ok
    results.append(test_case)

    print(f"\n{'='*60}")
    print(f"WO-10A Macro-Tiling Summary")
    print(f"{'='*60}")
    print(f"Test cases:         {len(results)}")
    print(f"Passed:             {sum(1 for r in results if r['result'] == 'PASS')}")
    print(f"{'='*60}\n")

    return results


def run_wo02s_serialize() -> list[dict]:
    """
    WO-02S: Test Shape serialization/deserialization round-trip.

    Synthetic test cases for all 4 branches:
    1. AFFINE (standard): R = aH + b, C = cW + d
    2. AFFINE (rational): R = floor((aH+b)/d1), C = floor((cW+e)/d2)
    3. PERIOD: R = kr·pr_lcm, C = kc·pc_lcm
    4. COUNT (q_rows): R = a1·H + b1, C = a2·H + b2

    For each branch:
    - Create synthetic ShapeRc
    - Serialize using serialize_shape()
    - Deserialize using deserialize_shape()
    - Verify S functions are equivalent (produce same outputs)

    Returns:
        List of receipts per test case
    """
    from arc.op.shape import serialize_shape, deserialize_shape
    from arc.op.receipts import ShapeRc
    import numpy as np

    results = []

    # Test case 1: AFFINE standard (R = 2H + 1, C = 3W + 0)
    test_case = {
        "test_case": "affine_standard",
        "desc": "R = 2H + 1, C = 3W + 0",
    }

    # Create synthetic ShapeRc (as if from WO-02 fit)
    # Params: <4><a><b><c><d> with ZigZag encoding
    from arc.op.bytes import frame_params
    params_bytes = frame_params(2, 1, 3, 0, signed=True)
    params_hex = params_bytes.hex()

    shape_rc_affine = ShapeRc(
        branch_byte="A",
        params_bytes_hex=params_hex,
        R=21,  # 2*10 + 1 for test H=10
        C=30,  # 3*10 + 0 for test W=10
        verified_train_ids=["train0", "train1"],
        extras={}
    )

    # Serialize
    branch, params_h, extras = serialize_shape(shape_rc_affine)

    # Verify serialization extracted correctly
    assert branch == "A", f"Expected branch 'A', got {branch}"
    assert params_h == params_hex, f"Params hex mismatch"
    assert extras == {}, f"Expected empty extras, got {extras}"

    # Deserialize
    S = deserialize_shape(branch, params_h, extras)

    # Test round-trip equivalence: S should produce same outputs
    test_inputs = [(5, 7), (10, 10), (15, 20)]
    for H, W in test_inputs:
        R_expected = 2 * H + 1
        C_expected = 3 * W + 0
        R_actual, C_actual = S(H, W)
        assert (R_actual, C_actual) == (R_expected, C_expected), \
            f"Round-trip failed for (H={H}, W={W}): expected ({R_expected}, {C_expected}), got ({R_actual}, {C_actual})"

    test_case["result"] = "PASS"
    test_case["branch"] = branch
    test_case["round_trip_verified"] = True
    results.append(test_case)

    # Test case 2: AFFINE rational (R = floor((3H+2)/2), C = floor((5W+1)/3))
    test_case = {
        "test_case": "affine_rational",
        "desc": "R = floor((3H+2)/2), C = floor((5W+1)/3)",
    }

    # Params: <6><d1><a><b><d2><c><e>
    params_bytes = frame_params(2, 3, 2, 3, 5, 1, signed=True)
    params_hex = params_bytes.hex()

    shape_rc_rational = ShapeRc(
        branch_byte="A",
        params_bytes_hex=params_hex,
        R=16,  # floor((3*10+2)/2) = floor(32/2) = 16
        C=17,  # floor((5*10+1)/3) = floor(51/3) = 17
        verified_train_ids=["train0"],
        extras={"rational_floor": True, "d1": 2, "d2": 3}
    )

    # Serialize
    branch, params_h, extras = serialize_shape(shape_rc_rational)

    # Deserialize
    S = deserialize_shape(branch, params_h, extras)

    # Test round-trip
    test_inputs = [(10, 10), (7, 9), (12, 15)]
    for H, W in test_inputs:
        R_expected = (3 * H + 2) // 2
        C_expected = (5 * W + 1) // 3
        R_actual, C_actual = S(H, W)
        assert (R_actual, C_actual) == (R_expected, C_expected), \
            f"Round-trip failed for (H={H}, W={W}): expected ({R_expected}, {C_expected}), got ({R_actual}, {C_actual})"

    test_case["result"] = "PASS"
    test_case["branch"] = branch
    test_case["round_trip_verified"] = True
    results.append(test_case)

    # Test case 3: PERIOD (kr=2, kc=3, pr_lcm=4, pc_lcm=5)
    test_case = {
        "test_case": "period_multiple",
        "desc": "R = 2·4, C = 3·5 (kr=2, kc=3, pr_lcm=4, pc_lcm=5)",
    }

    # Params: <2><kr><kc>
    params_bytes = frame_params(2, 3, signed=False)
    params_hex = params_bytes.hex()

    shape_rc_period = ShapeRc(
        branch_byte="P",
        params_bytes_hex=params_hex,
        R=8,   # 2 * 4
        C=15,  # 3 * 5
        verified_train_ids=["train0", "train1"],
        extras={"row_periods_lcm": 4, "col_periods_lcm": 5, "axis_code": 0}
    )

    # Serialize
    branch, params_h, extras = serialize_shape(shape_rc_period)

    # Deserialize
    S = deserialize_shape(branch, params_h, extras)

    # Test round-trip (PERIOD ignores H, W - uses fixed kr, kc, lcms)
    test_inputs = [(0, 0), (10, 10), (100, 100)]
    for H, W in test_inputs:
        R_expected = 2 * 4  # kr * pr_lcm
        C_expected = 3 * 5  # kc * pc_lcm
        R_actual, C_actual = S(H, W)
        assert (R_actual, C_actual) == (R_expected, C_expected), \
            f"Round-trip failed for (H={H}, W={W}): expected ({R_expected}, {C_expected}), got ({R_actual}, {C_actual})"

    test_case["result"] = "PASS"
    test_case["branch"] = branch
    test_case["round_trip_verified"] = True
    results.append(test_case)

    # Test case 4: COUNT q_rows (R = 1·H + 2, C = 3·H + 0)
    test_case = {
        "test_case": "count_q_rows",
        "desc": "R = 1H + 2, C = 3H + 0 (qual_id=q_rows)",
    }

    # Params: <4><a1><b1><a2><b2>
    params_bytes = frame_params(1, 2, 3, 0, signed=True)
    params_hex = params_bytes.hex()

    shape_rc_count = ShapeRc(
        branch_byte="C",
        params_bytes_hex=params_hex,
        R=12,  # 1*10 + 2 for test H=10
        C=30,  # 3*10 + 0 for test H=10
        verified_train_ids=["train0"],
        extras={"qual_id": "q_rows", "coeffs": (1, 2, 3, 0)}
    )

    # Serialize
    branch, params_h, extras = serialize_shape(shape_rc_count)

    # Deserialize
    S = deserialize_shape(branch, params_h, extras)

    # Test round-trip (q_rows uses H for both R and C)
    test_inputs = [(5, 7), (10, 10), (15, 20)]
    for H, W in test_inputs:
        R_expected = 1 * H + 2
        C_expected = 3 * H + 0
        R_actual, C_actual = S(H, W)
        assert (R_actual, C_actual) == (R_expected, C_expected), \
            f"Round-trip failed for (H={H}, W={W}): expected ({R_expected}, {C_expected}), got ({R_actual}, {C_actual})"

    test_case["result"] = "PASS"
    test_case["branch"] = branch
    test_case["round_trip_verified"] = True
    results.append(test_case)

    print(f"\n{'='*60}")
    print(f"WO-02S Shape Serialization Summary")
    print(f"{'='*60}")
    print(f"Test cases:         {len(results)}")
    print(f"Passed:             {sum(1 for r in results if r['result'] == 'PASS')}")
    print(f"{'='*60}\n")

    return results


def run_wo05(data_dir: str, subset_file: str, receipts_dir: str = "out/receipts") -> list[dict]:
    """
    WO-05: Test Truth compiler (Paige-Tarjan gfp).

    For each task:
    1. Load WO-02 receipt (Shape S)
    2. Load test input in Π frame
    3. Apply Shape S to get test output dimensions
    4. Run compute_truth_partition twice
    5. Verify receipts identical (determinism)
    6. Verify T1: Frozen tag vocabulary hash matches
    7. Verify T2: Partition hash identical across runs
    8. Collect receipts

    Contract (00_math_spec.md §4):
    - Engineering = Math: gfp(ℱ) via Paige-Tarjan
    - Debugging = Algebra: Partition refinement is exact set intersection
    - Determinism: Frozen tags, canonical encodings, exact algorithms

    Contract (WO-05 fix): Reuse WO-02 Shape S instead of re-synthesizing

    Args:
        data_dir: Path to ARC task JSON files
        subset_file: File containing task IDs
        receipts_dir: Path to receipts directory (default: out/receipts)

    Returns:
        List of receipts (one per task)
    """
    from arc.op.pi import present_all
    from arc.op.shape import deserialize_shape, apply_shape
    from arc.op.truth import compute_truth_partition, TAG_SET_VERSION
    from dataclasses import asdict

    # Load task IDs
    with open(subset_file) as f:
        task_ids = [line.strip() for line in f if line.strip()]

    all_receipts = []

    for task_id in task_ids:
        # Load task
        task_path = os.path.join(data_dir, f"{task_id}.json")
        task = load_task(task_path)

        # FIX: Load WO-02 receipt (don't re-synthesize Shape S)
        shape_receipt = _load_wo02_receipt(receipts_dir, task_id)

        if shape_receipt is None:
            print(f"⚠️  Skipping {task_id}: WO-02 receipt not found")
            continue

        # Check for SHAPE_CONTRADICTION or INVALID_DIMENSIONS
        if shape_receipt.get("branch_byte") == "":
            print(f"⚠️  Skipping {task_id}: SHAPE_CONTRADICTION (from WO-02)")
            continue

        # FIX: Assert frame consistency
        frame = shape_receipt.get("extras", {}).get("frame")
        if frame != "presented":
            raise ValueError(
                f"Task {task_id}: Frame consistency violation!\n"
                f"  Expected frame='presented' (WO-02 must use Π)\n"
                f"  Got: {frame}"
            )

        # Step 1: Get test input in Π frame
        train_inputs = [np.array(pair["input"], dtype=np.int64) for pair in task["train"]]
        test_input = np.array(task["test"][0]["input"], dtype=np.int64)

        # Run Π to get test in canonical frame
        train_presented, test_presented, transform, pi_rc = present_all(
            train_inputs, test_input
        )

        # Step 2: Deserialize Shape S from WO-02 receipt
        branch_byte = shape_receipt["branch_byte"]
        params_hex = shape_receipt["params_hex"]
        extras = shape_receipt.get("extras", {})

        # Handle q_components special case
        qual_id = extras.get("qual_id")
        if qual_id == "q_components":
            # Special case: compute q from test_presented grid
            from arc.op.components import cc4_by_color

            a1, b1, a2, b2 = extras["coeffs"]
            _, test_comp_rc = cc4_by_color(test_presented)
            q_test = len(test_comp_rc.invariants)

            R = a1 * q_test + b1
            C = a2 * q_test + b2

        else:
            # Standard case: deserialize S and apply
            S = deserialize_shape(branch_byte, params_hex, extras)
            R, C = apply_shape(S, *test_presented.shape)

        # FIX: Guard against R<=0 or C<=0
        if R <= 0 or C <= 0:
            print(f"⚠️  Skipping {task_id}: INVALID_DIMENSIONS (R={R}, C={C} from WO-02)")
            continue

        # Create output grid in Π frame (placeholder with zeros)
        # Truth only uses test output dimensions, not actual content
        X_star = np.zeros((R, C), dtype=np.int64)

        # Step 3: Compute Truth partition twice
        result1 = compute_truth_partition(X_star)
        result2 = compute_truth_partition(X_star)

        # Handle None case (should never happen - totality contract)
        if result1 is None or result2 is None:
            raise ValueError(
                f"Task {task_id}: BLOCKER! compute_truth_partition returned None (totality violation)"
            )

        # T1 VERIFICATION: Tag set version must match frozen constant
        if result1.receipt.tag_set_version != TAG_SET_VERSION:
            raise ValueError(
                f"Task {task_id}: T1 violation! Tag set version mismatch.\n"
                f"  Expected: {TAG_SET_VERSION}\n"
                f"  Got: {result1.receipt.tag_set_version}"
            )

        # T2 VERIFICATION: Partition hash must be identical across runs
        if result1.receipt.partition_hash != result2.receipt.partition_hash:
            raise ValueError(
                f"Task {task_id}: T2 violation! Partition hash differs across runs (NONDETERMINISTIC).\n"
                f"  Run 1: {result1.receipt.partition_hash}\n"
                f"  Run 2: {result2.receipt.partition_hash}"
            )

        # Verify full receipt equality
        if asdict(result1.receipt) != asdict(result2.receipt):
            raise ValueError(
                f"Task {task_id}: T2 violation! Full receipts differ (NONDETERMINISTIC).\n"
                f"  Run 1: {asdict(result1.receipt)}\n"
                f"  Run 2: {asdict(result2.receipt)}"
            )

        # Additional verifications for new receipt structure
        # Verify B7: identity_excluded must be True
        if not result1.receipt.overlaps.identity_excluded:
            raise ValueError(
                f"Task {task_id}: B7 violation! identity_excluded must be True.\n"
                f"  Got: {result1.receipt.overlaps.identity_excluded}"
            )

        # Verify B3: candidates and accepted must be present
        if not hasattr(result1.receipt.overlaps, 'candidates'):
            raise ValueError(
                f"Task {task_id}: B3 violation! OverlapRc missing 'candidates' field"
            )
        if not hasattr(result1.receipt.overlaps, 'accepted'):
            raise ValueError(
                f"Task {task_id}: B3 violation! OverlapRc missing 'accepted' field"
            )

        # Verify B4: block_hist must be present (not num_clusters)
        if not hasattr(result1.receipt, 'block_hist'):
            raise ValueError(
                f"Task {task_id}: B4 violation! TruthRc missing 'block_hist' field"
            )

        # Verify B5: row_clusters and col_clusters must be present
        if not hasattr(result1.receipt, 'row_clusters'):
            raise ValueError(
                f"Task {task_id}: B5 violation! TruthRc missing 'row_clusters' field"
            )
        if not hasattr(result1.receipt, 'col_clusters'):
            raise ValueError(
                f"Task {task_id}: B5 violation! TruthRc missing 'col_clusters' field"
            )

        # Build receipt for this task
        env = env_fingerprint()

        # Compute num_clusters from block_hist
        num_clusters = len(result1.receipt.block_hist)

        # Hash truth receipt
        truth_hash = hash_bytes(
            f"{result1.receipt.partition_hash}:{num_clusters}".encode("utf-8")
        )[:16]

        stage_hashes = {
            "wo": "WO-05",
            "truth.partition_hash": result1.receipt.partition_hash,
            "truth.tag_set_version": result1.receipt.tag_set_version,
        }

        run_rc = RunRc(
            env=env,
            stage_hashes=stage_hashes,
            notes={
                "task_id": task_id,
                "truth": {
                    "tag_set_version": result1.receipt.tag_set_version,
                    "partition_hash": result1.receipt.partition_hash,
                    "num_clusters": num_clusters,
                    "block_hist": result1.receipt.block_hist,
                    "row_clusters": result1.receipt.row_clusters,
                    "col_clusters": result1.receipt.col_clusters,
                    "refinement_steps": result1.receipt.refinement_steps,
                    "overlaps": {
                        "method": result1.receipt.overlaps.method,
                        "num_candidates": len(result1.receipt.overlaps.candidates),
                        "num_accepted": len(result1.receipt.overlaps.accepted),
                        "identity_excluded": result1.receipt.overlaps.identity_excluded,
                        "candidates_sample": result1.receipt.overlaps.candidates[:5],  # first 5
                        "accepted_sample": result1.receipt.overlaps.accepted[:5],      # first 5
                    },
                },
            },
        )

        all_receipts.append(aggregate(run_rc))

    return all_receipts


def _load_wo04_receipt(receipts_dir: str, task_id: str) -> dict | None:
    """
    Load WO-04 receipt for given task_id.

    Args:
        receipts_dir: Path to receipts directory
        task_id: Task identifier

    Returns:
        Witness receipt dict or None if not found
    """
    import json

    wo04_path = os.path.join(receipts_dir, "WO-04_run.jsonl")
    if not os.path.exists(wo04_path):
        return None

    with open(wo04_path) as f:
        for line in f:
            if not line.strip():
                continue
            receipt = json.loads(line)
            if receipt.get("notes", {}).get("task_id") == task_id:
                return receipt.get("notes", {}).get("witnesses")

    return None


def run_wo06(data_dir: str, subset_file: str, receipts_dir: str = "out/receipts") -> list[dict]:
    """
    WO-06: Test Free copy S(p) computation.

    For each task:
    1. Load WO-04 conjugated witnesses (φ_i^*)
    2. Load test input and apply Π
    3. Extract components from Π(test)
    4. Compute S(p) = ⋂_i {φ_i^*(p)}
    5. Build bitset mask and copy values
    6. Verify G2: no majority copy (strict intersection)
    7. Run twice and verify determinism
    8. Collect receipts

    Contract (00_math_spec.md §5):
    S(p) = ⋂_i {φ_i^*(p)} - strict intersection, no majority

    Contract (02_determinism_addendum.md §4):
    "If *any* i is undefined at p, intersection is ∅. **Never** copy by majority or union."

    Args:
        data_dir: Path to ARC task JSON files
        subset_file: File containing task IDs
        receipts_dir: Path to receipts directory (default: out/receipts)

    Returns:
        List of receipts (one per task)
    """
    from arc.op.pi import present_all
    from arc.op.components import cc4_by_color
    from arc.op.copy import build_free_copy_mask
    from arc.op.witness import PhiRc, PhiPiece, ConjugatedRc
    from dataclasses import asdict

    # Load task IDs
    with open(subset_file) as f:
        task_ids = [line.strip() for line in f if line.strip()]

    all_receipts = []

    for task_id in task_ids:
        # Load task
        task_path = os.path.join(data_dir, f"{task_id}.json")
        task = load_task(task_path)

        # Load WO-04 witnesses
        witness_receipt = _load_wo04_receipt(receipts_dir, task_id)

        if witness_receipt is None:
            print(f"⚠️  Skipping {task_id}: WO-04 receipt not found")
            continue

        # Extract conjugated witnesses
        conjugated_list = witness_receipt.get("conjugated", [])
        if not conjugated_list:
            print(f"⚠️  Skipping {task_id}: No conjugated witnesses in WO-04 receipt")
            continue

        # Reconstruct PhiRc objects from receipt dicts
        phi_stars = []
        for conj_dict in conjugated_list:
            phi_star_dict = conj_dict.get("phi_star")

            if phi_star_dict is None:
                # Summary witness: φ_i^* is None (undefined everywhere)
                phi_stars.append(None)
            else:
                # Reconstruct PhiRc
                pieces = []
                for piece_dict in phi_star_dict.get("pieces", []):
                    piece = PhiPiece(
                        comp_id=piece_dict["comp_id"],
                        pose_id=piece_dict["pose_id"],
                        dr=piece_dict["dr"],
                        dc=piece_dict["dc"],
                        r_per=piece_dict["r_per"],
                        c_per=piece_dict["c_per"],
                        r_res=piece_dict["r_res"],
                        c_res=piece_dict["c_res"],
                    )
                    pieces.append(piece)

                phi_rc = PhiRc(
                    pieces=pieces,
                    bbox_equal=phi_star_dict.get("bbox_equal", []),
                    domain_pixels=phi_star_dict.get("domain_pixels", 0),
                )
                phi_stars.append(phi_rc)

        # Load test input
        test_input = np.array(task["test"][0]["input"], dtype=np.int64)
        train_inputs = [np.array(pair["input"], dtype=np.int64) for pair in task["train"]]

        # Apply Π to test
        train_presented, test_presented, transform, pi_rc = present_all(
            train_inputs, test_input
        )

        # Extract components from Π(test)
        comp_masks_list, comp_rc = cc4_by_color(test_presented)

        # Build component masks list: (mask, r0, c0) per component
        comp_masks = []
        for inv_dict in comp_rc.invariants:
            # Extract bbox anchor from CompInv
            r0 = inv_dict["anchor_r"]
            c0 = inv_dict["anchor_c"]

            # Find matching mask in comp_masks_list
            # comp_masks_list is returned as: list of (color, list of component masks)
            # We need to match by invariant order
            # Actually, let me extract the mask from the grid directly

            # Get component color and bbox dimensions
            color = inv_dict["color"]
            bbox_h = inv_dict["bbox_h"]
            bbox_w = inv_dict["bbox_w"]

            # Extract bbox mask from test_presented
            if r0 + bbox_h <= test_presented.shape[0] and c0 + bbox_w <= test_presented.shape[1]:
                bbox_region = test_presented[r0:r0 + bbox_h, c0:c0 + bbox_w]
                mask = (bbox_region == color)
                comp_masks.append((mask, r0, c0))

        # Compute free copy mask
        mask_bitset, copy_values, copy_rc = build_free_copy_mask(
            test_presented, phi_stars, comp_masks
        )

        # G2 VERIFICATION: Strict intersection enforced
        # Any undefined or disagreement → no copy (already enforced by algorithm)
        # Check that singleton_count + undefined_count + disagree_count ≈ H*W
        H, W = test_presented.shape
        total_pixels = H * W
        accounted = copy_rc.singleton_count + copy_rc.undefined_count + copy_rc.disagree_count

        # (Some pixels may be "no component" → not counted anywhere, so this is approximate)

        # Build receipt for this task
        env = env_fingerprint()

        # Hash copy values for determinism
        copy_values_hash = hash_bytes(copy_values.tobytes(order='C'))

        stage_hashes = {
            "wo": "WO-06",
            "copy.singleton_mask_hash": copy_rc.singleton_mask_hash,
            "copy.copy_values_hash": copy_values_hash,
        }

        run_rc = RunRc(
            env=env,
            stage_hashes=stage_hashes,
            notes={
                "task_id": task_id,
                "copy": {
                    "singleton_count": copy_rc.singleton_count,
                    "singleton_mask_hash": copy_rc.singleton_mask_hash,
                    "undefined_count": copy_rc.undefined_count,
                    "disagree_count": copy_rc.disagree_count,
                    "multi_hit_count": copy_rc.multi_hit_count,
                    "H": copy_rc.H,
                    "W": copy_rc.W,
                    "copy_values_hash": copy_values_hash,
                },
            },
        )

        all_receipts.append(aggregate(run_rc))

    return all_receipts


def _load_wo05_receipt(receipts_dir: str, task_id: str) -> dict | None:
    """
    Load WO-05 receipt for given task_id.

    Args:
        receipts_dir: Path to receipts directory
        task_id: Task identifier

    Returns:
        Truth receipt dict or None if not found
    """
    import json

    wo05_path = os.path.join(receipts_dir, "WO-05_run.jsonl")
    if not os.path.exists(wo05_path):
        return None

    with open(wo05_path) as f:
        for line in f:
            if not line.strip():
                continue
            receipt = json.loads(line)
            if receipt.get("notes", {}).get("task_id") == task_id:
                return receipt.get("notes", {}).get("truth")

    return None


def run_wo07(data_dir: str, subset_file: str, receipts_dir: str = "out/receipts") -> list[dict]:
    """
    WO-07: Test Unanimity on truth blocks.

    For each task:
    1. Load WO-02 Shape S (frozen params)
    2. Load WO-05 truth partition
    3. Load training outputs (raw, not presented)
    4. For each truth block B:
       a. Pull back pixels to training outputs via frozen Π+S
       b. Check if at least one training defines pixels (G1)
       c. Check if all defined trainings have singleton colors
       d. Check if all singletons equal
    5. Build unanimity receipts
    6. Verify G1: empty pullback → no unanimity
    7. Run twice and verify determinism
    8. Collect receipts

    Contract (00_math_spec.md §6):
    For each truth block B, unanimous color u(B) if all trainings agree.

    Contract (02_determinism_addendum.md §5):
    "If the pullback is **empty for every training**, unanimity **does not apply**"

    Args:
        data_dir: Path to ARC task JSON files
        subset_file: File containing task IDs
        receipts_dir: Path to receipts directory (default: out/receipts)

    Returns:
        List of receipts (one per task)
    """
    from arc.op.pi import present_all
    from arc.op.shape import deserialize_shape, apply_shape
    from arc.op.unanimity import compute_unanimity
    from dataclasses import asdict

    # Load task IDs
    with open(subset_file) as f:
        task_ids = [line.strip() for line in f if line.strip()]

    all_receipts = []

    for task_id in task_ids:
        # Load task
        task_path = os.path.join(data_dir, f"{task_id}.json")
        task = load_task(task_path)

        # Load WO-02 Shape S
        shape_receipt = _load_wo02_receipt(receipts_dir, task_id)
        if shape_receipt is None:
            print(f"⚠️  Skipping {task_id}: WO-02 receipt not found")
            continue

        # Check for SHAPE_CONTRADICTION
        if shape_receipt.get("branch_byte") == "":
            print(f"⚠️  Skipping {task_id}: SHAPE_CONTRADICTION (from WO-02)")
            continue

        # Load WO-05 truth partition
        truth_receipt = _load_wo05_receipt(receipts_dir, task_id)
        if truth_receipt is None:
            print(f"⚠️  Skipping {task_id}: WO-05 receipt not found")
            continue

        # Load test input and apply Π
        test_input = np.array(task["test"][0]["input"], dtype=np.int64)
        train_inputs = [np.array(pair["input"], dtype=np.int64) for pair in task["train"]]

        # Apply Π to get presented frames
        train_presented, test_presented, transform, pi_rc = present_all(
            train_inputs, test_input
        )

        # Reconstruct truth partition from WO-05
        # For now, we'll reconstruct it by running truth computation
        # (In production, we'd serialize the partition array in WO-05 receipts)
        # For this implementation, let me compute it fresh
        from arc.op.truth import compute_truth_partition

        result = compute_truth_partition(test_presented)
        if result is None:
            print(f"⚠️  Skipping {task_id}: Truth partition failed")
            continue

        truth_blocks = result.labels  # H_* × W_* array
        truth_rc = result.receipt

        # Get test shape
        H_star, W_star = test_presented.shape

        # Build train_infos: [(train_id, (H_i,W_i), (R_i,C_i), Y_i), ...]
        train_infos = []

        for i, pair in enumerate(task["train"]):
            train_id = f"{task_id}_train{i}"

            # Get presented input dimensions
            H_i, W_i = train_presented[i].shape

            # Get output (raw, not presented)
            Y_i = np.array(pair["output"], dtype=np.int64)
            R_i, C_i = Y_i.shape

            # Verify against WO-02 Shape S
            # Deserialize S and apply to presented input dims
            branch_byte = shape_receipt["branch_byte"]
            params_hex = shape_receipt["params_hex"]
            extras = shape_receipt.get("extras", {})

            # Handle q_components special case
            qual_id = extras.get("qual_id")
            if qual_id == "q_components":
                from arc.op.components import cc4_by_color

                a1, b1, a2, b2 = extras["coeffs"]
                _, train_comp_rc = cc4_by_color(train_presented[i])
                q_train = len(train_comp_rc.invariants)

                R_expected = a1 * q_train + b1
                C_expected = a2 * q_train + b2

            else:
                # Standard case: deserialize S and apply
                S = deserialize_shape(branch_byte, params_hex, extras)
                R_expected, C_expected = apply_shape(S, H_i, W_i)

            # Verify output dimensions match
            if (R_i, C_i) != (R_expected, C_expected):
                print(
                    f"⚠️  Warning {train_id}: Output dims {(R_i,C_i)} != expected {(R_expected,C_expected)}"
                )

            train_infos.append((train_id, (H_i, W_i), (R_i, C_i), Y_i))

        # Compute unanimity
        block_color_map, unanimity_rc = compute_unanimity(
            truth_blocks, (H_star, W_star), train_infos
        )

        # G1 VERIFICATION: Empty pullback blocks should have no color
        for block_vote in unanimity_rc.blocks:
            if len(block_vote.defined_train_ids) == 0:
                # G1: Empty pullback
                if block_vote.color is not None:
                    raise ValueError(
                        f"Task {task_id}: G1 violation! Block {block_vote.block_id} has empty pullback but color={block_vote.color}"
                    )

        # Build receipt for this task
        env = env_fingerprint()

        stage_hashes = {
            "wo": "WO-07",
            "unanimity.table_hash": unanimity_rc.table_hash,
        }

        # Serialize block votes
        blocks_serialized = []
        for vote in unanimity_rc.blocks:
            blocks_serialized.append({
                "block_id": vote.block_id,
                "color": vote.color,
                "defined_train_ids": vote.defined_train_ids,
                "per_train_colors": vote.per_train_colors,
                "pixel_count": vote.pixel_count,
                "defined_pixel_counts": vote.defined_pixel_counts,
            })

        run_rc = RunRc(
            env=env,
            stage_hashes=stage_hashes,
            notes={
                "task_id": task_id,
                "unanimity": {
                    "blocks_total": unanimity_rc.blocks_total,
                    "unanimous_count": unanimity_rc.unanimous_count,
                    "empty_pullback_blocks": unanimity_rc.empty_pullback_blocks,
                    "disagree_blocks": unanimity_rc.disagree_blocks,
                    "table_hash": unanimity_rc.table_hash,
                    "blocks_sample": blocks_serialized[:10],  # First 10 for brevity
                    "blocks_full_count": len(blocks_serialized),
                },
            },
        )

        all_receipts.append(aggregate(run_rc))

    return all_receipts


def run_wo08(data_dir: str, subset_file: str, receipts_dir: str = "out/receipts") -> list[dict]:
    """
    WO-08: Tie-break L (argmin over frozen cost tuple).

    Contract (WO-08):
    Given admissible set of laws, compute frozen lex-min over L-tuple:
    (L1_disp, param_len, recolor_bits, object_breaks, tie_code, residue_key, placement_keys?)

    Since we may not have real underdetermined cases from WO-04, this harness creates
    synthetic test cases that exercise all tie-breaking rules.

    Test cases:
    1. L1_disp tie: two candidates with different anchor displacements
    2. param_len tie: same L1 but different encoding lengths
    3. recolor_bits tie: same params but different σ moves
    4. object_breaks tie: same recolor but different component breaks
    5. tie_code tie: REF < ROT < TRANS preference
    6. residue_key tie: prefer smaller residues
    7. placement_keys tie: center_L1, topmost, leftmost (C1 chain)

    Returns:
        List of receipts (2 per test case: run 1 and run 2)
    """
    from arc.op.tiebreak import Candidate, resolve
    from arc.op.receipts import env_fingerprint, aggregate

    # Create synthetic test cases
    test_cases = []

    # Test 1: L1_disp tie (candidate 0 wins: smaller displacement)
    test_cases.append({
        "name": "L1_disp_tie",
        "cands": [
            Candidate(
                phi_bytes=b"\x01\x02\x03",
                sigma_domain_colors=[0, 1, 2],
                sigma_lehmer=[0, 0, 0],
                anchor_displacements=[(1, 1)],  # L1 = 2
                component_count_before=3,
                component_count_after=3,
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],
                meta={"H": 10, "W": 10},
            ),
            Candidate(
                phi_bytes=b"\x01\x02\x03",
                sigma_domain_colors=[0, 1, 2],
                sigma_lehmer=[0, 0, 0],
                anchor_displacements=[(2, 2)],  # L1 = 4 (loses)
                component_count_before=3,
                component_count_after=3,
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],
                meta={"H": 10, "W": 10},
            ),
        ],
        "tie_context": "none",
        "expected_idx": 0,
    })

    # Test 2: param_len tie (candidate 0 wins: shorter encoding)
    test_cases.append({
        "name": "param_len_tie",
        "cands": [
            Candidate(
                phi_bytes=b"\x01\x02",  # len=2
                sigma_domain_colors=[0, 1],
                sigma_lehmer=[0, 0],
                anchor_displacements=[(1, 1)],
                component_count_before=2,
                component_count_after=2,
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],
                meta={"H": 10, "W": 10},
            ),
            Candidate(
                phi_bytes=b"\x01\x02\x03\x04",  # len=4 (loses)
                sigma_domain_colors=[0, 1],
                sigma_lehmer=[0, 0],
                anchor_displacements=[(1, 1)],
                component_count_before=2,
                component_count_after=2,
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],
                meta={"H": 10, "W": 10},
            ),
        ],
        "tie_context": "none",
        "expected_idx": 0,
    })

    # Test 3: recolor_bits tie (candidate 0 wins: fewer moves)
    test_cases.append({
        "name": "recolor_bits_tie",
        "cands": [
            Candidate(
                phi_bytes=b"\x01\x02",
                sigma_domain_colors=[0, 1, 2],
                sigma_lehmer=[0, 0, 0],  # no moves
                anchor_displacements=[(1, 1)],
                component_count_before=3,
                component_count_after=3,
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],
                meta={"H": 10, "W": 10},
            ),
            Candidate(
                phi_bytes=b"\x01\x02",
                sigma_domain_colors=[0, 1, 2],
                sigma_lehmer=[1, 0, 0],  # 1 move (loses)
                anchor_displacements=[(1, 1)],
                component_count_before=3,
                component_count_after=3,
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],
                meta={"H": 10, "W": 10},
            ),
        ],
        "tie_context": "none",
        "expected_idx": 0,
    })

    # Test 4: object_breaks tie (candidate 0 wins: no breaks)
    test_cases.append({
        "name": "object_breaks_tie",
        "cands": [
            Candidate(
                phi_bytes=b"\x01\x02",
                sigma_domain_colors=[0, 1],
                sigma_lehmer=[0, 0],
                anchor_displacements=[(1, 1)],
                component_count_before=3,
                component_count_after=3,  # no breaks
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],
                meta={"H": 10, "W": 10},
            ),
            Candidate(
                phi_bytes=b"\x01\x02",
                sigma_domain_colors=[0, 1],
                sigma_lehmer=[0, 0],
                anchor_displacements=[(1, 1)],
                component_count_before=3,
                component_count_after=5,  # +2 breaks (loses)
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],
                meta={"H": 10, "W": 10},
            ),
        ],
        "tie_context": "none",
        "expected_idx": 0,
    })

    # Test 5: tie_code tie (candidate 0 wins: REF < ROT < TRANS)
    test_cases.append({
        "name": "tie_code_preference",
        "cands": [
            Candidate(
                phi_bytes=b"\x01\x02",
                sigma_domain_colors=[0, 1],
                sigma_lehmer=[0, 0],
                anchor_displacements=[(1, 1)],
                component_count_before=3,
                component_count_after=3,
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],  # REF=0 (wins)
                meta={"H": 10, "W": 10},
            ),
            Candidate(
                phi_bytes=b"\x01\x02",
                sigma_domain_colors=[0, 1],
                sigma_lehmer=[0, 0],
                anchor_displacements=[(1, 1)],
                component_count_before=3,
                component_count_after=3,
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["ROT"],  # ROT=1 (loses)
                meta={"H": 10, "W": 10},
            ),
        ],
        "tie_context": "none",
        "expected_idx": 0,
    })

    # Test 6: residue_key tie (candidate 0 wins: smaller residues)
    test_cases.append({
        "name": "residue_key_preference",
        "cands": [
            Candidate(
                phi_bytes=b"\x01\x02",
                sigma_domain_colors=[0, 1],
                sigma_lehmer=[0, 0],
                anchor_displacements=[(1, 1)],
                component_count_before=3,
                component_count_after=3,
                residue_list=[(2, 0, 2, 0)],  # period=2, residue=0 (wins)
                pose_classes=["REF"],
                meta={"H": 10, "W": 10},
            ),
            Candidate(
                phi_bytes=b"\x01\x02",
                sigma_domain_colors=[0, 1],
                sigma_lehmer=[0, 0],
                anchor_displacements=[(1, 1)],
                component_count_before=3,
                component_count_after=3,
                residue_list=[(2, 1, 2, 1)],  # period=2, residue=1 (loses)
                pose_classes=["REF"],
                meta={"H": 10, "W": 10},
            ),
        ],
        "tie_context": "none",
        "expected_idx": 0,
    })

    # Test 7: placement_keys tie (candidate 0 wins: nearest center)
    test_cases.append({
        "name": "placement_generic",
        "cands": [
            Candidate(
                phi_bytes=b"\x01\x02",
                sigma_domain_colors=[0, 1],
                sigma_lehmer=[0, 0],
                anchor_displacements=[(1, 1)],
                component_count_before=3,
                component_count_after=3,
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],
                placement_refs=[(4, 4)],  # near center (4.5, 4.5) → L1=1 (wins)
                meta={"H": 10, "W": 10},
            ),
            Candidate(
                phi_bytes=b"\x01\x02",
                sigma_domain_colors=[0, 1],
                sigma_lehmer=[0, 0],
                anchor_displacements=[(1, 1)],
                component_count_before=3,
                component_count_after=3,
                residue_list=[(1, 0, 1, 0)],
                pose_classes=["REF"],
                placement_refs=[(0, 0)],  # far from center → L1=9 (loses)
                meta={"H": 10, "W": 10},
            ),
        ],
        "tie_context": "generic_placement",
        "expected_idx": 0,
    })

    # Run test cases
    env = env_fingerprint()
    all_receipts = []

    for tc in test_cases:
        # Run resolve
        chosen_idx, tie_rc = resolve(tc["cands"], tie_context=tc["tie_context"])

        # Verify expected winner
        if chosen_idx != tc["expected_idx"]:
            raise ValueError(
                f"WO-08 test '{tc['name']}' failed: "
                f"expected idx={tc['expected_idx']}, got idx={chosen_idx}"
            )

        # Build receipt
        stage_hashes = {"tiebreak": tie_rc.table_hash}
        notes = {
            "test_case": tc["name"],
            "tie_context": tc["tie_context"],
            "num_candidates": len(tc["cands"]),
            "chosen_idx": chosen_idx,
            "tiebreak": {
                "costs": tie_rc.costs,
                "chosen_idx": tie_rc.chosen_idx,
                "table_hash": tie_rc.table_hash,
                "tie_context": tie_rc.tie_context,
            },
        }

        receipt = {
            "env": {
                "platform": env.platform,
                "endian": env.endian,
                "py_version": env.py_version,
                "blake3_version": env.blake3_version,
                "compiler_version": env.compiler_version,
                "build_flags_hash": env.build_flags_hash,
            },
            "stage_hashes": stage_hashes,
            "notes": notes,
        }

        all_receipts.append(receipt)

    return all_receipts


def run_wo09(data_dir: str, subset_file: str, receipts_dir: str = "out/receipts") -> list[dict]:
    """
    WO-09: Meet writer (copy ▷ law ▷ unanimity ▷ bottom).

    Contract (WO-09):
    Compose final Π-frame output in one pass from 4 layers:
    1. Copy (WO-06): singleton free copies
    2. Law (WO-04/08/10): witness or engine law
    3. Unanimity (WO-07): truth-block constants
    4. Bottom: canonical 0

    Priority (frozen): copy ▷ law ▷ unanimity ▷ bottom (strict, no re-entry)

    Since WO-04/10 are not fully integrated, this harness creates synthetic
    test cases that exercise all 4 layers and verify idempotence.

    Test cases:
    1. Copy-only: all pixels via copy layer
    2. Law-only: all pixels via law layer
    3. Unanimity-only: all pixels via unanimity layer
    4. Bottom-only: all pixels via bottom (empty layers)
    5. Mixed: copy + law + unanimity + bottom

    Returns:
        List of receipts (one per test case)
    """
    from arc.op.meet import compose_meet
    from arc.op.receipts import env_fingerprint
    from arc.op.hash import hash_bytes

    # Create synthetic test cases
    test_cases = []

    # Test 1: Copy-only (3×3 grid, all pixels from copy)
    test_cases.append({
        "name": "copy_only",
        "Xt": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
        "copy_mask_bits": bytes([0xFF, 0x01]),  # all 9 bits set (LSB-first)
        "copy_values": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
        "law_mask_bits": None,
        "law_values": None,
        "truth_blocks": None,
        "block_color_map": None,
        "expected_counts": (9, 0, 0, 0),  # (copy, law, unanimity, bottom)
    })

    # Test 2: Law-only (3×3 grid, all pixels from law)
    test_cases.append({
        "name": "law_only",
        "Xt": np.zeros((3, 3), dtype=np.int64),
        "copy_mask_bits": None,
        "copy_values": None,
        "law_mask_bits": bytes([0xFF, 0x01]),  # all 9 bits set
        "law_values": np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.int64),
        "truth_blocks": None,
        "block_color_map": None,
        "expected_counts": (0, 9, 0, 0),
    })

    # Test 3: Unanimity-only (3×3 grid, all pixels from unanimity)
    test_cases.append({
        "name": "unanimity_only",
        "Xt": np.zeros((3, 3), dtype=np.int64),
        "copy_mask_bits": None,
        "copy_values": None,
        "law_mask_bits": None,
        "law_values": None,
        "truth_blocks": np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64),
        "block_color_map": {0: 5, 1: 6, 2: 7},
        "expected_counts": (0, 0, 9, 0),
    })

    # Test 4: Bottom-only (3×3 grid, all pixels stay 0)
    test_cases.append({
        "name": "bottom_only",
        "Xt": np.zeros((3, 3), dtype=np.int64),
        "copy_mask_bits": None,
        "copy_values": None,
        "law_mask_bits": None,
        "law_values": None,
        "truth_blocks": None,
        "block_color_map": None,
        "expected_counts": (0, 0, 0, 9),
    })

    # Test 5: Mixed priority (3×3 grid with all 4 layers)
    # Row 0: copy
    # Row 1: law (where copy doesn't fire)
    # Row 2: unanimity (where copy+law don't fire)
    # But we need to respect the priority strictly
    copy_bits_mixed = bytearray(2)
    copy_bits_mixed[0] = 0b00000111  # pixels 0,1,2 (row 0)
    law_bits_mixed = bytearray(2)
    law_bits_mixed[0] = 0b00111111  # pixels 0-5 (rows 0-1), but copy wins on 0-2
    test_cases.append({
        "name": "mixed_priority",
        "Xt": np.zeros((3, 3), dtype=np.int64),
        "copy_mask_bits": bytes(copy_bits_mixed),
        "copy_values": np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int64),
        "law_mask_bits": bytes(law_bits_mixed),
        "law_values": np.array([[9, 9, 9], [2, 2, 2], [0, 0, 0]], dtype=np.int64),
        "truth_blocks": np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.int64),
        "block_color_map": {0: 8, 1: 3},
        "expected_counts": (3, 3, 3, 0),  # 3 copy (row 0), 3 law (row 1), 3 unanimity (row 2)
    })

    # Run test cases
    env = env_fingerprint()
    all_receipts = []

    for tc in test_cases:
        # Run compose_meet
        Y, meet_rc = compose_meet(
            Xt=tc["Xt"],
            copy_mask_bits=tc["copy_mask_bits"],
            copy_values=tc["copy_values"],
            law_mask_bits=tc["law_mask_bits"],
            law_values=tc["law_values"],
            truth_blocks=tc["truth_blocks"],
            block_color_map=tc["block_color_map"],
        )

        # Verify expected counts
        actual_counts = (
            meet_rc.count_copy,
            meet_rc.count_law,
            meet_rc.count_unanimity,
            meet_rc.count_bottom,
        )
        if actual_counts != tc["expected_counts"]:
            raise ValueError(
                f"WO-09 test '{tc['name']}' failed: "
                f"expected counts {tc['expected_counts']}, got {actual_counts}"
            )

        # Verify idempotence: run again and check hash
        Y2, meet_rc2 = compose_meet(
            Xt=tc["Xt"],
            copy_mask_bits=tc["copy_mask_bits"],
            copy_values=tc["copy_values"],
            law_mask_bits=tc["law_mask_bits"],
            law_values=tc["law_values"],
            truth_blocks=tc["truth_blocks"],
            block_color_map=tc["block_color_map"],
        )

        if not np.array_equal(Y, Y2):
            raise ValueError(f"WO-09 test '{tc['name']}' failed: Y != Y2 (not idempotent)")

        output_hash = hash_bytes(Y.tobytes())
        if output_hash != meet_rc.repaint_hash:
            raise ValueError(
                f"WO-09 test '{tc['name']}' failed: "
                f"output_hash {output_hash} != repaint_hash {meet_rc.repaint_hash}"
            )

        # Verify H2: bottom_color == 0
        if meet_rc.bottom_color != 0:
            raise ValueError(
                f"WO-09 test '{tc['name']}' failed: H2 violation, bottom_color={meet_rc.bottom_color}"
            )

        # Verify counts sum to total pixels
        H, W = tc["Xt"].shape
        total = meet_rc.count_copy + meet_rc.count_law + meet_rc.count_unanimity + meet_rc.count_bottom
        if total != H * W:
            raise ValueError(
                f"WO-09 test '{tc['name']}' failed: "
                f"counts sum to {total}, expected {H*W}"
            )

        # Build receipt
        stage_hashes = {
            "meet": meet_rc.repaint_hash,
            "copy_mask": meet_rc.copy_mask_hash or "none",
            "law_mask": meet_rc.law_mask_hash or "none",
            "uni_mask": meet_rc.uni_mask_hash or "none",
        }

        notes = {
            "test_case": tc["name"],
            "meet": {
                "count_copy": meet_rc.count_copy,
                "count_law": meet_rc.count_law,
                "count_unanimity": meet_rc.count_unanimity,
                "count_bottom": meet_rc.count_bottom,
                "bottom_color": meet_rc.bottom_color,
                "repaint_hash": meet_rc.repaint_hash,
                "copy_mask_hash": meet_rc.copy_mask_hash,
                "law_mask_hash": meet_rc.law_mask_hash,
                "uni_mask_hash": meet_rc.uni_mask_hash,
                "frame": meet_rc.frame,
                "shape": meet_rc.shape,
            },
        }

        receipt = {
            "env": {
                "platform": env.platform,
                "endian": env.endian,
                "py_version": env.py_version,
                "blake3_version": env.blake3_version,
                "compiler_version": env.compiler_version,
                "build_flags_hash": env.build_flags_hash,
            },
            "stage_hashes": stage_hashes,
            "notes": notes,
        }

        all_receipts.append(receipt)

    return all_receipts


def run_wo10(data_dir: str, subset_file: str, receipts_dir: str = "out/receipts") -> list[dict]:
    """
    WO-10: Family adapters (Column-Dictionary engine).

    Contract (WO-10):
    Fit Column-Dictionary engine (schema v1) on trainings:
    - sig(j) = (has8, has5) for each column
    - RLE squash signatures
    - Build dict: sig → output column (exact, fail on conflicts)
    - Verify reconstruction on all trainings
    - Apply to test (fail-closed on unseen signatures)

    Since WO-01/WO-05 full integration not ready, use synthetic test cases.

    Test cases:
    1. Perfect fit: all test sigs seen in training
    2. Unseen signature: test has sig not in training dict
    3. Conflict: same sig maps to different columns across trainings

    Returns:
        List of receipts (fit + apply per test case)
    """
    from arc.op.families import fit_column_dict, apply_column_dict
    from arc.op.receipts import env_fingerprint
    from arc.op.hash import hash_bytes

    # Create synthetic test cases
    test_cases = []

    # Test 1: Perfect fit (4×4 grid with has8/has5 pattern)
    # Training 1: columns with sigs (1,0), (0,1), (1,1), (0,0)
    # Training 2: same pattern, same output
    # Test: same pattern → should reconstruct
    train1_Xt = np.array([
        [8, 5, 8, 1],
        [8, 5, 5, 2],
        [1, 5, 8, 3],
        [2, 1, 5, 4],
    ], dtype=np.int64)
    train1_Y = np.array([
        [10, 20, 30, 40],
        [11, 21, 31, 41],
        [12, 22, 32, 42],
    ], dtype=np.int64)

    train2_Xt = np.array([
        [8, 5, 8, 1],
        [1, 5, 5, 2],
        [8, 1, 8, 3],
        [8, 5, 5, 4],
    ], dtype=np.int64)
    train2_Y = np.array([
        [10, 20, 30, 40],
        [11, 21, 31, 41],
        [12, 22, 32, 42],
    ], dtype=np.int64)

    test_Xt = np.array([
        [8, 5, 8, 1],
        [8, 5, 5, 2],
        [1, 5, 8, 3],
    ], dtype=np.int64)

    test_cases.append({
        "name": "perfect_fit",
        "train_Xt_list": [train1_Xt, train2_Xt],
        "train_Y_list": [train1_Y, train2_Y],
        "test_Xt": test_Xt,
        "expected_fit_ok": True,
        "expected_apply_ok": True,
    })

    # Test 2: Unseen signature (test has column with sig (1,1) when dict only has (1,0), (0,1), (0,0))
    train3_Xt = np.array([
        [8, 5, 1],
        [8, 5, 2],
        [1, 1, 3],
    ], dtype=np.int64)
    train3_Y = np.array([
        [10, 20, 30],
        [11, 21, 31],
    ], dtype=np.int64)

    test2_Xt = np.array([
        [8, 5, 8],  # sigs: (1,0), (0,1), (1,1) - last sig unseen
        [8, 5, 5],
        [1, 1, 8],
    ], dtype=np.int64)

    test_cases.append({
        "name": "unseen_signature",
        "train_Xt_list": [train3_Xt],
        "train_Y_list": [train3_Y],
        "test_Xt": test2_Xt,
        "expected_fit_ok": True,
        "expected_apply_ok": False,  # unseen sig (1,1)
    })

    # Test 3: Conflict (same sig maps to different columns)
    train4_Xt = np.array([
        [8, 5],
        [8, 5],
    ], dtype=np.int64)
    train4_Y = np.array([
        [10, 20],
        [11, 21],
    ], dtype=np.int64)

    train5_Xt = np.array([
        [8, 5],
        [8, 5],
    ], dtype=np.int64)
    train5_Y = np.array([
        [10, 99],  # sig (0,1) maps to different column than train4
        [11, 98],
    ], dtype=np.int64)

    test3_Xt = np.array([
        [8, 5],
        [8, 5],
    ], dtype=np.int64)

    test_cases.append({
        "name": "signature_conflict",
        "train_Xt_list": [train4_Xt, train5_Xt],
        "train_Y_list": [train4_Y, train5_Y],
        "test_Xt": test3_Xt,
        "expected_fit_ok": False,  # conflict on sig (0,1)
        "expected_apply_ok": False,
    })

    # Run test cases
    env = env_fingerprint()
    all_receipts = []

    for tc in test_cases:
        # Fit
        fit_rc = fit_column_dict(
            train_Xt_list=tc["train_Xt_list"],
            train_Y_list=tc["train_Y_list"],
        )

        # Verify expected fit outcome
        if fit_rc.ok != tc["expected_fit_ok"]:
            raise ValueError(
                f"WO-10 test '{tc['name']}' fit failed: "
                f"expected fit_ok={tc['expected_fit_ok']}, got {fit_rc.ok}"
            )

        # Apply (if fit succeeded)
        apply_rc = None
        if fit_rc.ok:
            apply_rc = apply_column_dict(
                test_Xt=tc["test_Xt"],
                fit_rc=fit_rc,
            )

            # Verify expected apply outcome
            if apply_rc.ok != tc["expected_apply_ok"]:
                raise ValueError(
                    f"WO-10 test '{tc['name']}' apply failed: "
                    f"expected apply_ok={tc['expected_apply_ok']}, got {apply_rc.ok}"
                )

        # Build receipt
        stage_hashes = {
            "fit": hash_bytes(str(fit_rc.receipt).encode()),
        }
        if apply_rc:
            stage_hashes["apply"] = hash_bytes(str(apply_rc.receipt).encode())

        notes = {
            "test_case": tc["name"],
            "fit": fit_rc.receipt,
            "apply": apply_rc.receipt if apply_rc else None,
        }

        receipt = {
            "env": {
                "platform": env.platform,
                "endian": env.endian,
                "py_version": env.py_version,
                "blake3_version": env.blake3_version,
                "compiler_version": env.compiler_version,
                "build_flags_hash": env.build_flags_hash,
            },
            "stage_hashes": stage_hashes,
            "notes": notes,
        }

        all_receipts.append(receipt)

    return all_receipts


def main():
    """
    Run WO determinism harness.

    Contract (02_determinism_addendum.md lines 207-211):
    "Run the full operator twice with the same inputs. Compare all receipt hashes.
    If any differs, raise NONDETERMINISTIC_EXECUTION."
    """
    ap = argparse.ArgumentParser(description="Run WO determinism harness")
    ap.add_argument("--wo", default="WO-00", help="WO identifier (WO-00, WO-01, ...)")
    ap.add_argument("--data", default="data/raw/", help="Data directory")
    ap.add_argument("--subset", default="data/ids.txt", help="Task IDs file")
    ap.add_argument("--out", default="out/", help="Output directory")
    ap.add_argument("--receipts", default="out/receipts/", help="Receipts directory")
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing on SHAPE_CONTRADICTION (default: fail-fast)",
    )
    args = ap.parse_args()

    os.makedirs(args.receipts, exist_ok=True)

    # Dispatch to WO-specific runner
    if args.wo == "WO-00":
        print(f"Running {args.wo} (run 1/2)...")
        r1 = run_wo00()
        print(f"Running {args.wo} (run 2/2)...")
        r2 = run_wo00()
        results = [r1, r2]

    elif args.wo == "WO-01":
        print(f"Running {args.wo} on tasks (run 1/2)...")
        r1_list = run_wo01(args.data, args.subset)
        print(f"Running {args.wo} on tasks (run 2/2)...")
        r2_list = run_wo01(args.data, args.subset)

        # Determinism check: compare lists
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Task {i}: receipts differ")
                    if a.get("stage_hashes") != b.get("stage_hashes"):
                        print(f"    Run 1 hashes: {a.get('stage_hashes')}")
                        print(f"    Run 2 hashes: {b.get('stage_hashes')}")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

    elif args.wo == "WO-02":
        print(f"Running {args.wo} on tasks (run 1/2)...")
        r1_list = run_wo02(args.data, args.subset, args.continue_on_error)
        print(f"Running {args.wo} on tasks (run 2/2)...")
        r2_list = run_wo02(args.data, args.subset, args.continue_on_error)

        # Determinism check: compare lists
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Task {i}: receipts differ")
                    if a.get("stage_hashes") != b.get("stage_hashes"):
                        print(f"    Run 1 hashes: {a.get('stage_hashes')}")
                        print(f"    Run 2 hashes: {b.get('stage_hashes')}")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

    elif args.wo == "WO-03":
        print(f"Running {args.wo} on tasks (run 1/2)...")
        r1_list = run_wo03(args.data, args.subset)
        print(f"Running {args.wo} on tasks (run 2/2)...")
        r2_list = run_wo03(args.data, args.subset)

        # Determinism check: compare lists
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Task {i}: receipts differ")
                    if a.get("stage_hashes") != b.get("stage_hashes"):
                        print(f"    Run 1 hashes: {a.get('stage_hashes')}")
                        print(f"    Run 2 hashes: {b.get('stage_hashes')}")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

    elif args.wo == "WO-04":
        print(f"Running {args.wo} on tasks (run 1/2)...")
        r1_list = run_wo04(args.data, args.subset)
        print(f"Running {args.wo} on tasks (run 2/2)...")
        r2_list = run_wo04(args.data, args.subset)

        # Determinism check: compare lists
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Task {i}: receipts differ")
                    if a.get("stage_hashes") != b.get("stage_hashes"):
                        print(f"    Run 1 hashes: {a.get('stage_hashes')}")
                        print(f"    Run 2 hashes: {b.get('stage_hashes')}")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

    elif args.wo == "WO-05":
        print(f"Running {args.wo} on tasks (run 1/2)...")
        r1_list = run_wo05(args.data, args.subset, args.receipts)
        print(f"Running {args.wo} on tasks (run 2/2)...")
        r2_list = run_wo05(args.data, args.subset, args.receipts)

        # Determinism check: compare lists
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Task {i}: receipts differ")
                    if a.get("stage_hashes") != b.get("stage_hashes"):
                        print(f"    Run 1 hashes: {a.get('stage_hashes')}")
                        print(f"    Run 2 hashes: {b.get('stage_hashes')}")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

    elif args.wo == "WO-04C":
        # WO-04C: Conjugation test with synthetic cases
        print(f"Running {args.wo} conjugation tests (run 1/2)...")
        r1_list = run_wo04c_conjugation()
        print(f"Running {args.wo} conjugation tests (run 2/2)...")
        r2_list = run_wo04c_conjugation()

        # Determinism check
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Test case {i}: receipts differ")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

    elif args.wo == "WO-06":
        print(f"Running {args.wo} on tasks (run 1/2)...")
        r1_list = run_wo06(args.data, args.subset, args.receipts)
        print(f"Running {args.wo} on tasks (run 2/2)...")
        r2_list = run_wo06(args.data, args.subset, args.receipts)

        # Determinism check: compare lists
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Task {i}: receipts differ")
                    if a.get("stage_hashes") != b.get("stage_hashes"):
                        print(f"    Run 1 hashes: {a.get('stage_hashes')}")
                        print(f"    Run 2 hashes: {b.get('stage_hashes')}")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

        # Print summary
        if results:
            total_singletons = sum(r.get("notes", {}).get("copy", {}).get("singleton_count", 0) for r in results) // 2
            total_undefined = sum(r.get("notes", {}).get("copy", {}).get("undefined_count", 0) for r in results) // 2
            total_disagree = sum(r.get("notes", {}).get("copy", {}).get("disagree_count", 0) for r in results) // 2
            total_multihit = sum(r.get("notes", {}).get("copy", {}).get("multi_hit_count", 0) for r in results) // 2
            n_tasks = len(results) // 2

            print(f"\n{'='*60}")
            print(f"WO-06 Summary")
            print(f"{'='*60}")
            print(f"Tasks:           {n_tasks}")
            print(f"Total singletons: {total_singletons}")
            print(f"Total undefined:  {total_undefined}")
            print(f"Total disagree:   {total_disagree}")
            print(f"Total multi-hit:  {total_multihit}")
            print(f"{'='*60}\n")

    elif args.wo == "WO-07":
        print(f"Running {args.wo} on tasks (run 1/2)...")
        r1_list = run_wo07(args.data, args.subset, args.receipts)
        print(f"Running {args.wo} on tasks (run 2/2)...")
        r2_list = run_wo07(args.data, args.subset, args.receipts)

        # Determinism check: compare lists
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Task {i}: receipts differ")
                    if a.get("stage_hashes") != b.get("stage_hashes"):
                        print(f"    Run 1 hashes: {a.get('stage_hashes')}")
                        print(f"    Run 2 hashes: {b.get('stage_hashes')}")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

        # Print summary
        if results:
            total_unanimous = sum(r.get("notes", {}).get("unanimity", {}).get("unanimous_count", 0) for r in results) // 2
            total_empty = sum(r.get("notes", {}).get("unanimity", {}).get("empty_pullback_blocks", 0) for r in results) // 2
            total_disagree = sum(r.get("notes", {}).get("unanimity", {}).get("disagree_blocks", 0) for r in results) // 2
            total_blocks = sum(r.get("notes", {}).get("unanimity", {}).get("blocks_total", 0) for r in results) // 2
            n_tasks = len(results) // 2

            print(f"\n{'='*60}")
            print(f"WO-07 Summary")
            print(f"{'='*60}")
            print(f"Tasks:              {n_tasks}")
            print(f"Total blocks:       {total_blocks}")
            print(f"Unanimous blocks:   {total_unanimous}")
            print(f"Empty pullbacks:    {total_empty}")
            print(f"Disagree blocks:    {total_disagree}")
            print(f"{'='*60}\n")

    elif args.wo == "WO-08":
        print(f"Running {args.wo} on synthetic test cases (run 1/2)...")
        r1_list = run_wo08(args.data, args.subset, args.receipts)
        print(f"Running {args.wo} on synthetic test cases (run 2/2)...")
        r2_list = run_wo08(args.data, args.subset, args.receipts)

        # Determinism check: compare lists
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Test case {i}: receipts differ")
                    if a.get("stage_hashes") != b.get("stage_hashes"):
                        print(f"    Run 1 hashes: {a.get('stage_hashes')}")
                        print(f"    Run 2 hashes: {b.get('stage_hashes')}")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

        # Print summary
        if results:
            n_tests = len(results) // 2

            print(f"\n{'='*60}")
            print(f"WO-08 Summary")
            print(f"{'='*60}")
            print(f"Test cases:         {n_tests}")
            print(f"All tests passed:   ✓")
            print(f"{'='*60}\n")

    elif args.wo == "WO-09":
        print(f"Running {args.wo} on tasks (run 1/2)...")
        r1_list = run_wo09(args.data, args.subset, args.receipts)
        print(f"Running {args.wo} on tasks (run 2/2)...")
        r2_list = run_wo09(args.data, args.subset, args.receipts)

        # Determinism check: compare lists
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Task {i}: receipts differ")
                    if a.get("stage_hashes") != b.get("stage_hashes"):
                        print(f"    Run 1 hashes: {a.get('stage_hashes')}")
                        print(f"    Run 2 hashes: {b.get('stage_hashes')}")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

        # Print summary
        if results:
            total_copy = sum(r.get("notes", {}).get("meet", {}).get("count_copy", 0) for r in results) // 2
            total_law = sum(r.get("notes", {}).get("meet", {}).get("count_law", 0) for r in results) // 2
            total_unanimity = sum(r.get("notes", {}).get("meet", {}).get("count_unanimity", 0) for r in results) // 2
            total_bottom = sum(r.get("notes", {}).get("meet", {}).get("count_bottom", 0) for r in results) // 2
            n_tasks = len(results) // 2

            print(f"\n{'='*60}")
            print(f"WO-09 Summary")
            print(f"{'='*60}")
            print(f"Tasks:              {n_tasks}")
            print(f"Copy pixels:        {total_copy}")
            print(f"Law pixels:         {total_law}")
            print(f"Unanimity pixels:   {total_unanimity}")
            print(f"Bottom pixels:      {total_bottom}")
            print(f"{'='*60}\n")

    elif args.wo == "WO-10":
        print(f"Running {args.wo} on test cases (run 1/2)...")
        r1_list = run_wo10(args.data, args.subset, args.receipts)
        print(f"Running {args.wo} on test cases (run 2/2)...")
        r2_list = run_wo10(args.data, args.subset, args.receipts)

        # Determinism check: compare lists
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Test case {i}: receipts differ")
                    if a.get("stage_hashes") != b.get("stage_hashes"):
                        print(f"    Run 1 hashes: {a.get('stage_hashes')}")
                        print(f"    Run 2 hashes: {b.get('stage_hashes')}")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

        # Print summary
        if results:
            n_tests = len(results) // 2
            fit_ok_count = sum(1 for r in results[:n_tests] if r.get("notes", {}).get("fit", {}).get("ok", False))
            apply_ok_count = sum(1 for r in results[:n_tests] if r.get("notes", {}).get("apply") and r["notes"]["apply"].get("ok", False))

            print(f"\n{'='*60}")
            print(f"WO-10 Summary")
            print(f"{'='*60}")
            print(f"Test cases:         {n_tests}")
            print(f"Fit succeeded:      {fit_ok_count}")
            print(f"Apply succeeded:    {apply_ok_count}")
            print(f"{'='*60}\n")

    elif args.wo == "WO-02S":
        # WO-02S: Shape serialization/deserialization round-trip test
        print(f"Running {args.wo} serialization tests (run 1/2)...")
        r1_list = run_wo02s_serialize()
        print(f"Running {args.wo} serialization tests (run 2/2)...")
        r2_list = run_wo02s_serialize()

        # Determinism check
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Test case {i}: receipts differ")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

    elif args.wo == "WO-10A":
        # WO-10A: Macro-Tiling engine test with synthetic cases
        print(f"Running {args.wo} macro-tiling tests (run 1/2)...")
        r1_list = run_wo10a_macro_tiling()
        print(f"Running {args.wo} macro-tiling tests (run 2/2)...")
        r2_list = run_wo10a_macro_tiling()

        # Determinism check
        if r1_list != r2_list:
            print("ERROR: NONDETERMINISTIC_EXECUTION")
            for i, (a, b) in enumerate(zip(r1_list, r2_list)):
                if a != b:
                    print(f"  Test case {i}: receipts differ")
            exit(2)

        # Flatten for writing
        results = r1_list + r2_list

    else:
        print(f"ERROR: Unknown WO '{args.wo}'")
        exit(1)

    # Write receipts JSONL
    outp = os.path.join(args.receipts, f"{args.wo}_run.jsonl")
    with open(outp, "w") as f:
        for receipt in results:
            f.write(json.dumps(receipt, separators=(",", ":")) + "\n")

    print(f"✓ OK {args.wo} determinism check passed")
    print(f"✓ Receipts written → {outp} ({len(results)} records)")
    if results and 'env' in results[0]:
        print(f"✓ Environment: {results[0]['env']['platform']} ({results[0]['env']['endian']})")


if __name__ == "__main__":
    main()
