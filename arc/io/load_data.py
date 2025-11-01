# arc/io/load_data.py
# WO-00: Minimal ARC task JSON loader

from __future__ import annotations
import json
from typing import Any


def load_task(path: str) -> dict[str, Any]:
    """
    Load ARC task from JSON file.

    Expected format:
    {
        "train": [{"input": [[...]], "output": [[...]]}, ...],
        "test": [{"input": [[...]], "output": [[...]]}]
    }

    Args:
        path: path to task JSON file

    Returns:
        dict: parsed task with train/test pairs
    """
    with open(path, "r") as f:
        return json.load(f)
