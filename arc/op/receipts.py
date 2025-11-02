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
class CopyRc:
    """
    Free copy receipt.

    Contract (02_determinism_addendum.md §10 line 245):
    "Free copy: singleton_count and singleton_mask_hash (bitset bytes)"

    Contract (00_math_spec.md §5):
    S(p) = ⋂_i {φ_i^*(p)} - strict intersection, no majority
    """
    singleton_count: int           # |{p : |S(p)| = 1}|
    singleton_mask_hash: str       # BLAKE3(bitset) - row-major LSB-first
    undefined_count: int           # |{p : ∃i, p ∉ dom(φ_i^*)}|
    disagree_count: int            # |{p : φ_i^*(p) ≠ φ_j^*(p)}|
    multi_hit_count: int           # |{p : |φ_i^*(p)| > 1}| (should be 0)
    H: int
    W: int


@dataclass
class BlockVote:
    """
    Per-block unanimity vote.

    Contract (02_determinism_addendum.md §5):
    Record (block_id, color, defined_train_ids) for each truth block.
    """
    block_id: int
    color: int | None                        # None if not unanimous
    defined_train_ids: list[str]             # trainings that defined ≥1 pixel
    per_train_colors: dict[str, list[int]]   # unique colors per training
    pixel_count: int                         # |B|
    defined_pixel_counts: dict[str, int]     # #defined pixels per training


@dataclass
class UnanimityRc:
    """
    Unanimity receipt.

    Contract (02_determinism_addendum.md §10):
    "Unanimity: list of (block_id, color, defined_train_ids)"

    Contract (00_math_spec.md §6):
    For each truth block B, unanimous color u(B) if all trainings agree.
    """
    blocks_total: int                # total truth blocks
    unanimous_count: int             # blocks with unanimous color
    empty_pullback_blocks: int       # G1: blocks where no training defines pixels
    disagree_blocks: int             # blocks where trainings disagree
    table_hash: str                  # BLAKE3 of decision table
    blocks: list[dict[str, Any]]     # BlockVote serialized (or minimal if large)


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
