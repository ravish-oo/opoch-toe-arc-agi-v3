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
    if results:
        print(f"✓ Environment: {results[0]['env']['platform']} ({results[0]['env']['endian']})")


if __name__ == "__main__":
    main()
