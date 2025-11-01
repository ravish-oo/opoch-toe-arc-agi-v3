#!/usr/bin/env python3
# scripts/run_wo.py
# WO-00: Determinism harness shell
# Implements 02_determinism_addendum.md §11 determinism harness

from __future__ import annotations
import argparse
import os
import json
from arc.op.receipts import env_fingerprint, aggregate, RunRc


def run_once(wo: str) -> dict:
    """
    Execute one run of the specified WO.

    For WO-00, this only captures env and placeholder stage hashes.
    Later WOs will add real operator computations.

    Args:
        wo: WO identifier (e.g., "WO-00")

    Returns:
        dict: aggregated receipts
    """
    env = env_fingerprint()

    # Placeholder stage_hashes; WO-01+ will add real hashes.
    # MUST be deterministic - no timestamps or volatile fields.
    # (e.g., "pi.roundtrip_hash", "truth.partition_hash")
    stage_hashes = {
        "wo": wo,
    }

    # Build RunRc
    run_rc = RunRc(
        env=env,
        stage_hashes=stage_hashes,
        notes={"status": "WO-00 placeholder run"},
    )

    return aggregate(run_rc)


def main():
    """
    Run WO determinism harness.

    Contract (02_determinism_addendum.md lines 207-211):
    "Run the full operator twice with the same inputs. Compare all receipt hashes.
    If any differs, raise NONDETERMINISTIC_EXECUTION."

    For WO-00, we only check env equality (stage hashes are placeholders).
    Later WOs enforce full receipt equality.
    """
    ap = argparse.ArgumentParser(description="Run WO determinism harness")
    ap.add_argument("--wo", default="WO-00", help="WO identifier")
    ap.add_argument("--data", default="data/raw/", help="Data directory (unused in WO-00)")
    ap.add_argument("--subset", default="data/ids.txt", help="Task IDs (unused in WO-00)")
    ap.add_argument("--out", default="out/", help="Output directory (unused in WO-00)")
    ap.add_argument("--receipts", default="out/receipts/", help="Receipts directory")
    args = ap.parse_args()

    os.makedirs(args.receipts, exist_ok=True)

    # Double-run determinism check
    print(f"Running {args.wo} (run 1/2)...")
    r1 = run_once(args.wo)

    print(f"Running {args.wo} (run 2/2)...")
    r2 = run_once(args.wo)

    # Determinism check: entire receipt must match
    # Contract (02_determinism_addendum.md §11): "Compare all receipt hashes"
    if r1 != r2:
        print("ERROR: NONDETERMINISTIC_EXECUTION")
        if r1["env"] != r2["env"]:
            print("  Environment differs:")
            print(f"    Run 1 env: {r1['env']}")
            print(f"    Run 2 env: {r2['env']}")
        if r1.get("stage_hashes") != r2.get("stage_hashes"):
            print("  Stage hashes differ:")
            print(f"    Run 1: {r1.get('stage_hashes')}")
            print(f"    Run 2: {r2.get('stage_hashes')}")
        exit(2)

    # Write receipts JSONL
    outp = os.path.join(args.receipts, f"{args.wo}_run.jsonl")
    with open(outp, "w") as f:
        f.write(json.dumps(r1, separators=(",", ":")) + "\n")
        f.write(json.dumps(r2, separators=(",", ":")) + "\n")

    print(f"✓ OK {args.wo} determinism check passed")
    print(f"✓ Receipts written → {outp}")
    print(f"✓ Environment: {r1['env']['platform']} ({r1['env']['endian']})")


if __name__ == "__main__":
    main()
