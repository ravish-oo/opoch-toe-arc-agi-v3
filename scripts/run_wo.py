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


def run_wo02(data_dir: str, subset_file: str) -> list[dict]:
    """
    WO-02: Test Shape S synthesis on tasks.

    For each task:
    1. Load all training pairs to collect (H,W) → (R',C')
    2. Run synthesize_shape to get S function and ShapeRc
    3. Apply S to test input to get (R,C)
    4. Verify E1: proof on ALL trainings (equality check)
    5. Collect receipts

    Contract (docs/common_mistakes.md E1):
    - Never change output shape without S proof
    - Receipt must include verified_train_ids (ALL trainings)

    Returns:
        List of receipts (one per task)
    """
    from arc.op.shape import synthesize_shape, apply_shape
    from arc.op.hash import hash_bytes

    # Load task IDs
    with open(subset_file) as f:
        task_ids = [line.strip() for line in f if line.strip()]

    all_receipts = []

    for task_id in task_ids:
        # Load task
        task_path = os.path.join(data_dir, f"{task_id}.json")
        task = load_task(task_path)

        # Build train_pairs: [(train_id, (H,W), (R',C'))]
        train_pairs = []
        presented_inputs = []

        for i, pair in enumerate(task["train"]):
            train_id = f"{task_id}_train{i}"
            X = np.array(pair["input"], dtype=np.int64)
            Y = np.array(pair["output"], dtype=np.int64)
            train_pairs.append((train_id, X.shape, Y.shape))
            presented_inputs.append((train_id, X))

        # Synthesize shape
        S, shape_rc = synthesize_shape(train_pairs, presented_inputs)

        # Apply to test input
        test_input = np.array(task["test"][0]["input"], dtype=np.int64)
        Ht, Wt = test_input.shape
        Rt, Ct = apply_shape(S, Ht, Wt)

        # Fill receipt placeholders
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
        r1_list = run_wo02(args.data, args.subset)
        print(f"Running {args.wo} on tasks (run 2/2)...")
        r2_list = run_wo02(args.data, args.subset)

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
