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
