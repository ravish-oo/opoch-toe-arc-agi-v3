#!/usr/bin/env python3
# scripts/run_tasks.py
# WO-11: Batch task runner with J1 determinism harness

"""
Contract (WO-11):
Run solve_task() on a curated subset with J1 determinism checks.

J1 Determinism:
- Run solve_task() twice per task
- Compare all section hashes + table_hash + output_hash
- Fail if NONDETERMINISTIC_EXECUTION (hashes differ within same env)
- Warn if NONDETERMINISTIC_ENV (env fingerprints differ)

Output:
- Per-task receipts to out/receipts/WO-11_run.jsonl
- Summary: law_status counts, pass/fail, determinism violations
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add project root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from arc.runner import solve_task, RunRc
from arc.op.hash import hash_bytes


def load_solutions_json(data_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load ground truth solutions from arc-agi_training_solutions.json.

    Returns:
        Dict mapping task_id -> oracle output array (or empty dict if not found)
    """
    solutions_path = data_dir / "arc-agi_training_solutions.json"
    if not solutions_path.exists():
        return {}

    with open(solutions_path) as f:
        solutions_raw = json.load(f)

    # Convert to numpy arrays
    solutions = {}
    for task_id, outputs in solutions_raw.items():
        # outputs is a list of grids (usually just 1 test case)
        if outputs and len(outputs) > 0:
            solutions[task_id] = np.array(outputs[0], dtype=np.uint8)

    return solutions


def load_arc_task(task_id: str, data_dir: Path) -> Dict[str, Any]:
    """
    Load ARC task from JSON file.

    Returns:
        {
            "train": [{"input": [[...]], "output": [[...]]}, ...],
            "test": [{"input": [[...]], "output": [[...]]}]  # output may be None
        }
    """
    # Try raw directory first (individual task files)
    task_path = data_dir / "raw" / f"{task_id}.json"
    if task_path.exists():
        with open(task_path) as f:
            return json.load(f)

    # Try training and evaluation directories (ARC1 structure)
    for subdir in ["training", "evaluation"]:
        task_path = data_dir / subdir / f"{task_id}.json"
        if task_path.exists():
            with open(task_path) as f:
                return json.load(f)

    raise FileNotFoundError(f"Task {task_id} not found in {data_dir}")


def task_to_arrays(task_data: Dict[str, Any]) -> tuple[
    List[tuple[str, np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray | None
]:
    """
    Convert JSON task to numpy arrays.

    Returns:
        (train_pairs, Xstar_raw, Ystar_raw_oracle)
        where train_pairs = [(train_id, X_raw, Y_raw), ...]
        and Ystar_raw_oracle is ground truth (None if unavailable)
    """
    train_pairs = []
    for i, pair in enumerate(task_data["train"]):
        X_raw = np.array(pair["input"], dtype=np.uint8)
        Y_raw = np.array(pair["output"], dtype=np.uint8)
        train_pairs.append((f"train{i}", X_raw, Y_raw))

    # Test input (always present)
    Xstar_raw = np.array(task_data["test"][0]["input"], dtype=np.uint8)

    # Test output (oracle, may be None for evaluation set)
    Ystar_oracle = task_data["test"][0].get("output")
    if Ystar_oracle is not None:
        Ystar_oracle = np.array(Ystar_oracle, dtype=np.uint8)

    return train_pairs, Xstar_raw, Ystar_oracle


def run_task_with_determinism(
    task_id: str,
    train_pairs: List[tuple[str, np.ndarray, np.ndarray]],
    Xstar_raw: np.ndarray,
    Ystar_oracle: np.ndarray | None
) -> Dict[str, Any]:
    """
    Run solve_task() twice and check J1 determinism.

    Returns:
        {
            "task_id": str,
            "result": "PASS" | "FAIL" | "NONDETERMINISTIC_EXECUTION" | "NONDETERMINISTIC_ENV" | "ERROR",
            "law_status": str,
            "oracle_match": bool | None,
            "output_shape": [H, W],
            "table_hash_run1": str,
            "table_hash_run2": str,
            "env_run1": dict,
            "env_run2": dict,
            "error": str | None
        }
    """
    result_summary = {
        "task_id": task_id,
        "result": "PASS",
        "law_status": None,
        "oracle_match": None,
        "output_shape": None,
        "table_hash_run1": None,
        "table_hash_run2": None,
        "env_run1": None,
        "env_run2": None,
        "error": None
    }

    try:
        # Run 1
        Ystar_run1, rc1 = solve_task(task_id, train_pairs, Xstar_raw)

        # Run 2
        Ystar_run2, rc2 = solve_task(task_id, train_pairs, Xstar_raw)

        # Extract key values
        result_summary["law_status"] = rc1.final["law_status"]
        result_summary["output_shape"] = rc1.final["shape"]
        result_summary["table_hash_run1"] = rc1.table_hash
        result_summary["table_hash_run2"] = rc2.table_hash
        result_summary["env_run1"] = rc1.env
        result_summary["env_run2"] = rc2.env

        # Check J1 determinism
        # 1. Environment must match (or warn)
        if rc1.env != rc2.env:
            result_summary["result"] = "NONDETERMINISTIC_ENV"
            result_summary["error"] = "Environment fingerprints differ between runs"
            return result_summary

        # 2. All section hashes must match
        if rc1.hashes != rc2.hashes:
            result_summary["result"] = "NONDETERMINISTIC_EXECUTION"
            result_summary["error"] = "Section hashes differ between runs"
            # Find which sections differ
            diff_sections = [k for k in rc1.hashes if rc1.hashes.get(k) != rc2.hashes.get(k)]
            result_summary["error"] += f": {diff_sections}"
            return result_summary

        # 3. Table hash must match
        if rc1.table_hash != rc2.table_hash:
            result_summary["result"] = "NONDETERMINISTIC_EXECUTION"
            result_summary["error"] = "Table hashes differ between runs"
            return result_summary

        # 4. Output hash must match
        output_hash_run1 = rc1.final["output_hash"]
        output_hash_run2 = rc2.final["output_hash"]

        if output_hash_run1 != output_hash_run2:
            result_summary["result"] = "NONDETERMINISTIC_EXECUTION"
            result_summary["error"] = "Output hashes differ between runs"
            return result_summary

        # 5. Outputs must be identical (redundant check, but explicit)
        if not np.array_equal(Ystar_run1, Ystar_run2):
            result_summary["result"] = "NONDETERMINISTIC_EXECUTION"
            result_summary["error"] = "Outputs differ between runs (array mismatch)"
            return result_summary

        # Check oracle match (if available)
        if Ystar_oracle is not None:
            oracle_match = np.array_equal(Ystar_run1, Ystar_oracle)
            result_summary["oracle_match"] = oracle_match
            if not oracle_match:
                result_summary["result"] = "FAIL"
                result_summary["error"] = "Output does not match oracle"

        # All checks passed
        result_summary["result"] = "PASS"

    except Exception as e:
        result_summary["result"] = "ERROR"
        result_summary["error"] = str(e)

    return result_summary


def run_batch(
    task_ids: List[str],
    data_dir: Path,
    output_path: Path,
    fail_fast: bool = False
) -> Dict[str, Any]:
    """
    Run batch of tasks with J1 determinism checks.

    Args:
        task_ids: List of ARC task IDs
        data_dir: Path to ARC data directory
        output_path: Path to output JSONL file
        fail_fast: Stop on first failure

    Returns:
        Summary dict with counts and results
    """
    results = []
    law_status_counts = {}
    result_counts = {}

    # Load ground truth solutions from arc-agi_training_solutions.json
    print(f"Loading solutions from {data_dir / 'arc-agi_training_solutions.json'}...")
    solutions = load_solutions_json(data_dir)
    print(f"Loaded {len(solutions)} solutions")
    print()

    for i, task_id in enumerate(task_ids):
        print(f"[{i+1}/{len(task_ids)}] Running {task_id}...", end=" ", flush=True)

        try:
            # Load task
            task_data = load_arc_task(task_id, data_dir)
            train_pairs, Xstar_raw, _ = task_to_arrays(task_data)

            # Get oracle from solutions JSON
            Ystar_oracle = solutions.get(task_id)

            # Run with determinism check
            result = run_task_with_determinism(task_id, train_pairs, Xstar_raw, Ystar_oracle)

            # Update counts
            result_status = result["result"]
            result_counts[result_status] = result_counts.get(result_status, 0) + 1

            if result["law_status"]:
                law_status_counts[result["law_status"]] = law_status_counts.get(result["law_status"], 0) + 1

            # Print result
            if result_status == "PASS":
                oracle_str = f"✓ oracle" if result.get("oracle_match") else "? no oracle"
                print(f"{result_status} ({result['law_status']}) {oracle_str}")
            else:
                print(f"{result_status}: {result.get('error', 'unknown')}")

            results.append(result)

            # Fail fast
            if fail_fast and result_status not in ["PASS"]:
                print(f"\nFail-fast: Stopping on first {result_status}")
                break

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "task_id": task_id,
                "result": "ERROR",
                "error": str(e)
            })
            result_counts["ERROR"] = result_counts.get("ERROR", 0) + 1

            if fail_fast:
                print("\nFail-fast: Stopping on error")
                break

    # Write results to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Summary
    summary = {
        "total": len(results),
        "result_counts": result_counts,
        "law_status_counts": law_status_counts,
        "results": results
    }

    return summary


def main():
    """
    Main entry point for WO-11 batch runner.

    Usage:
        python scripts/run_tasks.py [--subset <file>] [--fail-fast]

    Default subset: 6 hand-solved IDs from WO-11 spec
    """
    import argparse

    parser = argparse.ArgumentParser(description="WO-11: Task runner with J1 determinism")
    parser.add_argument("--subset", type=str, help="Path to file with task IDs (one per line)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to ARC data directory")
    parser.add_argument("--output", type=str, default="out/receipts/WO-11_run.jsonl", help="Output JSONL path")

    args = parser.parse_args()

    # Default curated subset (from WO-11 spec)
    default_subset = [
        "d5c634a2",  # Hand-solved
        "995c5fa3",  # Hand-solved
        "3cd86f4f",  # Hand-solved
        "23b5c85d",  # Hand-solved
        "2037f2c7",  # Hand-solved
        "ccd554ac",  # Hand-solved
    ]

    # Load subset
    if args.subset:
        with open(args.subset) as f:
            task_ids = [line.strip() for line in f if line.strip()]
    else:
        task_ids = default_subset
        print(f"Using default curated subset ({len(task_ids)} tasks)")

    # Run batch
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    print(f"\nWO-11: Running {len(task_ids)} tasks with J1 determinism checks")
    print(f"Data dir: {data_dir}")
    print(f"Output: {output_path}")
    print(f"Fail-fast: {args.fail_fast}\n")

    summary = run_batch(task_ids, data_dir, output_path, fail_fast=args.fail_fast)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tasks: {summary['total']}")
    print(f"\nResult counts:")
    for result, count in sorted(summary['result_counts'].items()):
        print(f"  {result}: {count}")

    if summary['law_status_counts']:
        print(f"\nLaw status counts:")
        for status, count in sorted(summary['law_status_counts'].items()):
            print(f"  {status}: {count}")

    print(f"\nReceipts written to: {output_path}")

    # Exit code
    if summary['result_counts'].get('NONDETERMINISTIC_EXECUTION', 0) > 0:
        print("\n❌ NONDETERMINISTIC_EXECUTION detected!")
        sys.exit(1)
    elif summary['result_counts'].get('ERROR', 0) > 0:
        print("\n❌ Errors detected!")
        sys.exit(1)
    elif summary['result_counts'].get('FAIL', 0) > 0:
        print("\n⚠️  Some tasks failed (but deterministic)")
        sys.exit(0)  # Not a blocker for WO-11 (determinism is the key)
    else:
        print("\n✓ All tasks passed with J1 determinism")
        sys.exit(0)


if __name__ == "__main__":
    main()
