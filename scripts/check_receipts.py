#!/usr/bin/env python3
# scripts/check_receipts.py
# WO-00: Receipt comparison tool
# Implements receipt diffing for determinism verification

from __future__ import annotations
import json
import sys


def load_jsonl(path: str) -> list[dict]:
    """
    Load JSONL file as list of records.

    Args:
        path: path to JSONL file

    Returns:
        list of parsed JSON objects
    """
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def deep_diff(a: dict, b: dict, path: str = "") -> list[str]:
    """
    Recursively find differences between two dicts.

    Args:
        a, b: dicts to compare
        path: current path for error messages

    Returns:
        list of difference descriptions
    """
    diffs = []

    # Check keys
    a_keys = set(a.keys())
    b_keys = set(b.keys())

    if a_keys != b_keys:
        only_a = a_keys - b_keys
        only_b = b_keys - a_keys
        if only_a:
            diffs.append(f"{path}: keys only in A: {only_a}")
        if only_b:
            diffs.append(f"{path}: keys only in B: {only_b}")

    # Check values for common keys
    for key in a_keys & b_keys:
        new_path = f"{path}.{key}" if path else key
        val_a = a[key]
        val_b = b[key]

        if isinstance(val_a, dict) and isinstance(val_b, dict):
            diffs.extend(deep_diff(val_a, val_b, new_path))
        elif val_a != val_b:
            diffs.append(f"{new_path}: {val_a!r} != {val_b!r}")

    return diffs


def main():
    """
    Compare two receipt JSONL files for differences.

    Usage:
        python -m scripts.check_receipts <file1.jsonl> <file2.jsonl>

    Exit codes:
        0: receipts match
        1: receipts differ
    """
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.check_receipts <file1.jsonl> <file2.jsonl>")
        sys.exit(1)

    file_a, file_b = sys.argv[1], sys.argv[2]

    print(f"Comparing receipts:")
    print(f"  A: {file_a}")
    print(f"  B: {file_b}")

    records_a = load_jsonl(file_a)
    records_b = load_jsonl(file_b)

    if len(records_a) != len(records_b):
        print(f"✗ RECEIPTS_DIFFER: record count mismatch ({len(records_a)} vs {len(records_b)})")
        sys.exit(1)

    all_match = True
    for i, (rec_a, rec_b) in enumerate(zip(records_a, records_b)):
        diffs = deep_diff(rec_a, rec_b, f"record[{i}]")
        if diffs:
            all_match = False
            print(f"\n✗ Differences in record {i}:")
            for diff in diffs[:10]:  # Show first 10 diffs
                print(f"  {diff}")
            if len(diffs) > 10:
                print(f"  ... and {len(diffs) - 10} more differences")

    if all_match:
        print(f"✓ RECEIPTS_MATCH ({len(records_a)} records)")
        return

    print("\n✗ RECEIPTS_DIFFER")
    sys.exit(1)


if __name__ == "__main__":
    main()
