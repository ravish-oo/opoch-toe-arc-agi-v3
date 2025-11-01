# ARC-AGI Operator

**A search-free, receipts-first solver for ARC-AGI.**

## Overview

This solver implements the **single commuting operator** described in the mathematical specification:

```
Y* = U⁻¹(Π ∘ gfp(F) ∘ ⋂Conj(φᵢ,σᵢ) ∘ Meet)(X*)
```

Where:
- **Π** — Idempotent presentation (palette, D4 pose, anchor)
- **gfp(F)** — Truth: largest bisimulation under provable equalities
- **⋂Conj(φᵢ,σᵢ)** — Co-observed witness laws, transported and intersected
- **Meet** — Least write: copy ▷ law ▷ unanimity ▷ bottom
- **U⁻¹** — Exact inverse of Π

## Principles

1. **Engineering = Math**: The program is the proof. No heuristics.
2. **Receipts-first**: Every function returns `(value, receipt)`.
3. **Determinism**: All computations use frozen encodings (uint32_le, ZigZag LEB128, BLAKE3).
4. **Fail-closed**: Contradictions abort with explicit flags; never guess.

## Installation

```bash
# Python 3.11+ required
pip install -e .
```

## Usage

### WO-00: Determinism Harness

Run the determinism check (executes twice, compares receipts):

```bash
python -m scripts.run_wo --wo WO-00 --receipts out/receipts/
```

Compare two receipt files:

```bash
python -m scripts.check_receipts out/receipts/WO-00_run.jsonl out/receipts/WO-00_run.jsonl
```

## Repository Structure

```
arc-agi-operator/
├── docs/anchors/          # Mathematical specifications (frozen)
│   ├── 00_math_spec.md
│   ├── 01_engineering_spec.md
│   ├── 02_determinism_addendum.md
│   └── 03_repo_structure.md
├── arc/
│   ├── op/                # Operator modules (WO-00 through WO-10)
│   │   ├── bytes.py       # WO-00: uint32_le, ZigZag LEB128
│   │   ├── hash.py        # WO-00: BLAKE3 helpers
│   │   └── receipts.py    # WO-00: receipt dataclasses
│   └── io/                # I/O utilities
│       ├── load_data.py
│       └── save.py
├── scripts/               # Runnable entrypoints
│   ├── run_wo.py          # Run WO with determinism check
│   └── check_receipts.py  # Diff receipt files
├── data/
│   └── raw/               # ARC task JSON files
└── out/
    ├── receipts/          # JSONL receipts per run
    └── y_pred/            # Predicted outputs
```

## Work Orders (WOs)

- **WO-00**: ✅ Repo scaffold + receipts kernel (bytes, hash, receipts, determinism harness)
- **WO-01**: Π (presentation): palette canon, D4 lex pose, anchor
- **WO-02**: Shape synthesis (exact + least)
- **WO-03**: Components (4-conn, invariants, matching)
- **WO-04**: Witness solver (φ, σ, conjugation, intersection)
- **WO-05**: Truth (frozen tags, Paige-Tarjan gfp)
- **WO-06**: Free copy sets
- **WO-07**: Unanimity
- **WO-08**: Tie-break (minimum-bit L)
- **WO-09**: Meet (copy ▷ law ▷ unanimity ▷ bottom)
- **WO-10**: Symbolic families (band/slice helpers)

## Determinism Guarantees

All computations use:
- **uint32_le** grid serialization (02_determinism_addendum.md §0)
- **ZigZag LEB128** for signed integers
- **BLAKE3** for all hashing
- **Frozen tag vocabulary** (no dynamic tags)
- **Fixed tie-breaks** (no tunable parameters)

Every run captures `env_fingerprint` (platform, endian, Python version, compiler, build flags).
The harness runs all operations twice and compares receipt hashes; any mismatch raises `NONDETERMINISTIC_EXECUTION`.

## Specifications

See `docs/anchors/` for complete mathematical and engineering specifications.

## License

See LICENSE file.
