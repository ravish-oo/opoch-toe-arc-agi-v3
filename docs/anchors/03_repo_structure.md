# ARC‑AGI Solver — Repository Structure v1.0

This document freezes the repository layout so Implementer and Reviewer never diverge. Paths are canonical; do not invent new directories. All modules are receipts‑first.

---

## 0) Top‑level tree

```
arc-agi-operator/
├─ pyproject.toml              # Python 3.11+, build + deps (blake3, numpy; no heavy ML)
├─ README.md                   # how to run scripts, determinism note
├─ LICENSE
├─ .python-version             # 3.11.x
├─ .gitignore
├─ docs/
│  └─ anchors/
│     ├─ 00_math_spec.md
│     ├─ 01_engineering_spec.md
│     └─ 02_determinism_addendum.md
├─ data/                       # ARC tasks (inputs/outputs) + subsets
│  ├─ raw/                     # original JSON/tasks as provided
│  ├─ ids.txt                  # mini-suite seed (edit only by PR)
│  └─ subsets/                 # optional curated lists per WO
├─ out/
│  ├─ receipts/                # JSONL receipts per run
│  └─ y_pred/                  # produced outputs per task
├─ scripts/                    # runnable entrypoints
│  ├─ run_wo.py                # run a single WO on a subset
│  ├─ run_tasks.py             # full operator on many tasks
│  ├─ check_receipts.py        # diff receipts across runs/branches
│  └─ make_subset.py           # build ids lists
├─ arc/                        # source
│  ├─ __init__.py
│  ├─ op/                      # operator modules (WO‑00 … WO‑10)
│  │  ├─ __init__.py
│  │  ├─ bytes.py              # WO‑00: uint32_le, ZigZag LEB128, framing
│  │  ├─ hash.py               # WO‑00: BLAKE3 helpers
│  │  ├─ receipts.py           # WO‑00: receipts structs + aggregator
│  │  ├─ pi.py                 # WO‑01: Π present/unpresent
│  │  ├─ palette.py            # WO‑01: inputs‑only palette canon
│  │  ├─ d4.py                 # WO‑01: D4 lex pose ids
│  │  ├─ anchor.py             # WO‑01: bbox to (0,0)
│  │  ├─ shape.py              # WO‑02: exact + least S
│  │  ├─ components.py         # WO‑03: CC4 + invariants + matching
│  │  ├─ witness.py            # WO‑04: (φ,σ), conjugation, intersection
│  │  ├─ truth.py              # WO‑05: frozen tags + Paige–Tarjan
│  │  ├─ copy.py               # WO‑06: free singletons S(p)
│  │  ├─ unanimity.py          # WO‑07: block constants
│  │  ├─ tiebreak.py           # WO‑08: L argmin
│  │  ├─ meet.py               # WO‑09: copy ▷ law ▷ unanimity ▷ bottom
│  │  └─ families.py           # WO‑10: symbolic band/slice helpers
│  ├─ runner.py                # WO‑11: single commuting operator
│  └─ io/
│     ├─ __init__.py
│     ├─ load_data.py          # parse ARC JSON → python objects
│     └─ save.py               # write y_pred, receipts JSONL
└─ tests/                      # optional ad‑hoc property tests (no CI)
   └─ props_repaint_idem.py    # local property scripts only
```

---

## 1) Conventions

* **Python**: 3.11.x. Deterministic deps: `blake3`, `numpy` (or `numba` optional), no float FFT unless verified by exact equality; prefer integer NTT.
* **Style**: receipts‑first. Every public function returns `(value, receipt)`.
* **Hashes**: BLAKE3 only. Grid hashing uses **uint32_le row‑major** serialization.
* **Encodings**: ZigZag LEB128 for signed ints; plain LEB128 for unsigned.
* **No CI**: scripts are the contract. Determinism harness is mandatory.

---

## 2) Receipts format

* Per WO, emit a **single JSONL** under `out/receipts/WO-XX_<timestamp>.jsonl` with one record per task (or synthetic run), containing nested receipts per module. Include `env_fingerprint` per run.
* Field names and minimal sets are frozen by **02_determinism_addendum.md §10–§11**.

---

## 3) Scripts

### scripts/run_wo.py

```
python -m scripts.run_wo --wo WO-05 \
  --data data/raw/ --subset data/ids.txt \
  --out out/ --receipts out/receipts/
```

Behavior: runs only the modules needed for that WO, twice, compares receipt hashes, writes JSONL.

### scripts/run_tasks.py

Runs full operator across ids, writes `out/y_pred/<task_id>.json` and a combined `out/receipts/full_run.jsonl`.

### scripts/check_receipts.py

Diff two receipts JSONL files; fails on any hash mismatch, highlights first differing key.

### scripts/make_subset.py

Create curated lists for families (band, slices, geometric, min‑rect, skyline).

---

## 4) Module contracts (summary)

* `arc/op/bytes.py` — `to_bytes_grid(G)->bytes`, `zigzag_encode(i)->bytes`, `varint(u)->bytes`.
* `arc/op/hash.py` — `hash_grid(G)->hex`, `hash_bytes(b)->hex`.
* `arc/op/receipts.py` — dataclasses: `PiRc, ShapeRc, ComponentsRc, WitnessRc, TruthRc, CopyRc, UnanimityRc, TieRc, MeetRc, RunRc`; `aggregate(rcs)->dict`.
* `arc/op/pi.py` — `present(G)->(G~, T, PiRc)`, `unpresent(T, Y~)->Y`.
* `arc/op/shape.py` — `synthesize(train_pairs)->(S_fn, ShapeRc)`, `apply(S_fn, X~)->(R,C)`.
* `arc/op/components.py` — `components(G~)->list[Comp]`, `match(Xcomps, Ycomps)->(pairs, ComponentsRc)`.
* `arc/op/witness.py` — `solve_witness(X~,Y~)->((φ,σ), WitnessRc)`, `conjugate((φ,σ),Pi_train,Pi_test)->((φ*,σ*), Rc)`, `intersect(list)->(law|flags, Rc)`.
* `arc/op/truth.py` — `compile_tags(X~, trainings)->(TruthPartition, TruthRc)`.
* `arc/op/copy.py` — `free_singletons(phi_star_list)->(mask_bitset, CopyRc)`.
* `arc/op/unanimity.py` — `unanimous_colors(truth, trainings)->(map, UnanimityRc)`.
* `arc/op/tiebreak.py` — `choose(laws)->(law, TieRc)`.
* `arc/op/meet.py` — `write(X~, law, u_map, mask)->(Y~, MeetRc)`.
* `arc/runner.py` — `solve_task(train_pairs, X*)->(Y, RunRc)`.

---

## 5) Data layout

* `data/raw/` holds the canonical ARC JSON tasks. Each file named `<task_id>.json` with keys `{train: [...], test: [...]}`.
* `data/ids.txt` baseline list:

```
d5c634a2
995c5fa3
3cd86f4f
7bb29440
23b5c85d
2037f2c7
ccd554ac
652646ff
```

* `data/subsets/WO-XX.txt` optional per WO.

---

## 6) Determinism harness

* Every script runs the module **twice** and asserts equality of all receipt hashes (including `env_fingerprint`). On mismatch, exit with `NONDETERMINISTIC_EXECUTION` or `NONDETERMINISTIC_ENV`.

---

## 7) Expansion rules (later WOs)

* Do not add new directories. Add only files listed above as WOs expand.
* If a WO must split, suffix files with `_a.py`, `_b.py` but keep within the same folder and receipts unchanged.

---

**End.** This structure is canonical; any deviation is a bug. Implementers must place code exactly under these paths, and reviewers test strictly via `scripts/run_wo.py` and `scripts/run_tasks.py`.
