# arc/op/receipts.py
# WO-00: Receipts kernel and environment fingerprinting
# Implements 02_determinism_addendum.md §10 receipts schema

from __future__ import annotations
import platform
import sys
import json
from dataclasses import dataclass, asdict
from typing import Any
from .hash import hash_bytes


@dataclass
class EnvRc:
    """
    Environment fingerprint.

    Contract (02_determinism_addendum.md line 210):
    "env_fingerprint with {platform, endian, blake3_version, compiler_version, build_flags_hash}"

    All fields are BLOCKER for determinism harness.
    """
    platform: str
    endian: str
    py_version: str
    blake3_version: str
    compiler_version: str | None
    build_flags_hash: str


def env_fingerprint() -> EnvRc:
    """
    Capture environment fingerprint for determinism checking.

    Contract (02_determinism_addendum.md lines 207-211):
    "Run the full operator twice... env_fingerprint must include
    {platform, endian, blake3_version, compiler_version, build_flags_hash}"

    Returns:
        EnvRc: environment receipt
    """
    endian = sys.byteorder  # "little" or "big"

    # blake3 version from package metadata (frozen in pyproject.toml)
    b3v = "0.4.1"

    # Python compiler string
    comp = platform.python_compiler()

    # Build flags hash: combine Python version and implementation
    # This captures interpreter-level differences
    build_info = {
        "py_version": sys.version,
        "implementation": platform.python_implementation(),
        "version_info": list(sys.version_info),
    }
    flags = hash_bytes(json.dumps(build_info, sort_keys=True).encode())

    return EnvRc(
        platform=platform.platform(),
        endian=endian,
        py_version=platform.python_version(),
        blake3_version=b3v,
        compiler_version=comp,
        build_flags_hash=flags,
    )


@dataclass
class ShapeRc:
    """
    Shape synthesis receipt.

    Contract (02_determinism_addendum.md §10):
    S: branch_byte, params_bytes_hex, R, C, verified_train_ids

    Contract (02_determinism_addendum.md §1.1):
    Branch codes: 'A' (AFFINE), 'P' (PERIOD_MULTIPLE), 'C' (COUNT_BASED), 'F' (FRAME)
    Ordering key: (branch_byte, params_bytes, R, C)
    """
    branch_byte: str  # 'A', 'P', 'C', 'F'
    params_bytes_hex: str  # deterministic serialization
    R: int  # output height for test
    C: int  # output width for test
    verified_train_ids: list[str]  # must include ALL trainings
    extras: dict[str, Any]  # axis_code for P, qual_id/qual_hash for C, etc.


@dataclass
class ComponentsRc:
    """
    Components extraction receipt.

    Contract (02_determinism_addendum.md §3):
    - connectivity = "4" (frozen by D2)
    - per_color_counts: color → #components
    - invariants: CompInv list in lex order

    Note: CompInv details are in arc/op/components.py
    This receipt stores the serialized invariant data.
    """
    connectivity: str  # "4" (frozen)
    per_color_counts: dict[int, int]  # color → count
    invariants: list[dict[str, Any]]  # serialized CompInv tuples
    note: str | None = None


@dataclass
class RunRc:
    """
    Root receipt container for a single task run.

    Contract (02_determinism_addendum.md §10):
    Minimum fields: env, stage_hashes, optional notes

    Later WOs will add:
    - pi: PiRc
    - shape: ShapeRc
    - witness: WitnessRc
    - truth: TruthRc
    - copy: CopyRc
    - unanimity: UnanimityRc
    - tiebreak: TieRc
    - meet: MeetRc
    """
    env: EnvRc
    stage_hashes: dict[str, str]
    notes: dict[str, Any] | None = None


def aggregate(run: dict | RunRc) -> dict:
    """
    Convert nested receipts (dataclasses or dicts) to JSON-serializable dict.

    Contract (02_determinism_addendum.md line 77):
    "Per WO, emit a single JSONL... with nested receipts per module"

    Args:
        run: RunRc or dict containing receipts

    Returns:
        dict: flattened, JSON-serializable representation
    """
    def to_plain(x: Any) -> Any:
        """Recursively convert dataclasses to dicts."""
        if hasattr(x, "__dataclass_fields__"):
            return {k: to_plain(v) for k, v in asdict(x).items()}
        if isinstance(x, dict):
            return {k: to_plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_plain(v) for v in x]
        return x

    return to_plain(run)
