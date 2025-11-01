# arc/io/save.py
# WO-00: Minimal JSON writer for outputs and receipts

from __future__ import annotations
import json
import os
from typing import Any


def write_json(path: str, obj: Any) -> None:
    """
    Write object as JSON to file.

    Creates parent directories if needed.
    Uses compact JSON (no whitespace) for determinism.

    Args:
        path: output file path
        obj: JSON-serializable object
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))


def write_jsonl(path: str, records: list[Any]) -> None:
    """
    Write list of objects as JSONL (one JSON object per line).

    Used for receipts output.

    Args:
        path: output file path
        records: list of JSON-serializable objects
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
