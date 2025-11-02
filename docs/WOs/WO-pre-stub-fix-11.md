> Anchors referenced:
> • Math Spec: Π→S→Truth→law(Meet) flow, conjugation, totality. 
> • Engg Spec: “one commuting operator”, reuse S params, engines as Spec-B, receipts first-class. 
> • Determinism Addendum: frozen tag set; overlap verify; tie-tuple; bottom=0; determinism/J1; no dynamic knobs. 
> • Common-Mistakes (guards A1/A2/B1/B2/C1/C2/G1/G2/J1): freeze candidate sets, strict majority, tie chains, no majority copy, fail-closed, determinism. 

---

# WO-10B — **Border-Scalar Engine** (MAJOR)

**Intent**: Reconstruct small outputs that are **constant border/interior fills** derived from **largest 4-CC border color** and **minimal interior 4-CC color**, as per archetype (B-a).
**Fit** is input-only + output check; **Apply** paints small frame/interior with verified colors.

## Deliverables

`arc/op/engines/border_scalar.py` (≈300–400 LOC)

```python
@dataclass
class BorderScalarFitRc:
    engine: Literal["border_scalar"]
    border_color: int          # chosen color id
    interior_color: int        # chosen color id (may equal border_color)
    rule: str                  # e.g. "border_max_cc → interior_min_cc; tie→min_color"
    band_source: str           # "truth.frame" (from WO-05: which rows/cols are border)
    fit_verified_on: List[str] # all train ids
    counts: Dict[str,Any]      # diagnostics: per-color cc areas on border, per-color interior cc sizes
    hash: str
```

```python
def fit_border_scalar(train_Xt_list: List[Tuple[str, np.ndarray]],  # Π-train inputs
                      train_Y_raw_list: List[Tuple[str, np.ndarray]]) -> Tuple[bool, BorderScalarFitRc]
def apply_border_scalar(test_Xt: np.ndarray, truth_rc, fit_rc: BorderScalarFitRc,
                        expected_shape: Optional[Tuple[int,int]]=None) -> Tuple[np.ndarray, Dict]
```

## Frozen algorithm (Spec-B “a”)

1. **Frame bands**: From Truth (WO-05) pick **outermost rows/cols** as border bands (exact: `row=0 | row=H*-1 | col=0 | col=W*-1`). 
2. **Border color**: collect 4-CC components restricted to border band; choose **max area** component’s **color**; **ties by smallest color**. (A1/C1) 
3. **Interior color**: in interior region (exclude border band), compute 4-CC per color; choose **min area >0** component’s **color**; **ties by smallest color**. If no interior CC exists, set interior=0 (background). (A2)
4. **Verify trainings**: for each Yᵢ, check Yᵢ’s outer border rows/cols are exactly `border_color` and interior (if present) is constant `interior_color`. Otherwise **ok=False** with diagnostic counts.
5. **Apply**: small shape is from **WO-02** if available; else derive from training outputs (must be constant across trainings). Paint outermost row/col positions with `border_color`, fill interior with `interior_color`.
6. **Receipts**: record `border_color`, `interior_color`, rule string, per-color CC area tables, `fit_verified_on`, `final_shape`, `apply_hash`.

## PO / Red-team

* PO: Ties are recorded and resolved lex-min; receiver can recompute CC areas by reading receipts.
* Red-team: perturb border by one pixel in a train Y → fit must fail with mismatch; change border color on test → application receipt still shows chosen `border_color`, failing `final_hash` vs gold in acceptance.

---

# WO-10C — **Pooled-Blocks Engine** (MAJOR)

**Intent**: Two-stage block voting (e.g., 2×2 pool) from band grids: **block votes → pooled quadrants** (archetype B-d). Frozen **strict majority** and tie chains.

## Deliverables

`arc/op/engines/pooled_blocks.py` (≈400–500 LOC)

```python
@dataclass
class PooledBlocksFitRc:
    engine: Literal["pooled_blocks"]
    row_bands: List[int]; col_bands: List[int]   # from Truth row/col clusters
    block_shape: Tuple[int,int]                  # (#rows,#cols) of block grid
    pool_shape: Tuple[int,int]                   # e.g., (2,2)
    foreground_colors: List[int]; background_colors: List[int]
    stage1_counts: Dict[Tuple[int,int], Dict[int,int]]   # (br,bc) -> {color:count}
    stage2_pooled: List[List[int]]               # pooled grid colors
    decision_rule: str                           # "strict_majority_foreground_fallback_0"
    fit_verified_on: List[str]
    hash: str
```

```python
def fit_pooled_blocks(train_Xt_list, train_Y_raw_list, truth_train_list) -> Tuple[bool, PooledBlocksFitRc]
def apply_pooled_blocks(test_Xt, truth_test, fit_rc: PooledBlocksFitRc,
                        expected_shape: Optional[Tuple[int,int]]) -> Tuple[np.ndarray, Dict]
```

## Frozen algorithm

1. **Bands**: From WO-05 `row/col_clusters`. (D1) 
2. **Stage-1**: For each band cell `(br,bc)`, compute **foreground counts** over its Π input patch (colors in (\mathbb F) from trainings — A1); winner by **strict majority**; no majority ⇒ background (A2/C2). Log counts and decisions.
3. **Stage-2 (pooling)**: Pool fixed windows (e.g., 2×2) of stage-1 votes to form small grid; rule is strict majority per pooled cell; ties → smallest color (C2).
4. **Verify**: Reconstruct each training Yᵢ exactly; else ok=False with diagnostics (`stage1_mismatches`, `stage2_mismatches`).
5. **Apply**: Apply same bands and rule on `Xt`; emit small `Y*` and receipts (`row_bands`, `col_bands`, per-cell counts/decisions). If `expected_shape` is set (S known), assert sizes match; else set `final_shape`.
6. **Tie-chains**: if multiple pooled placements exist (rare), build `tiebreak.Candidates` with `placement_keys` and call WO-08; log `TieRc`.

## PO / Red-team

* PO: strict-majority enforced; A1/A2/C2 rules logged; determinism over order.
* Red-team: change counting to “mode” → rule string mismatch + acceptance failure. Create pooled tie → verify `TieRc` chosen_idx stable under candidate reorder.

---

# WO-10D — **Markers-Grid Engine** (MAJOR)

**Intent**: Detect a **grid of 2×2 solid markers** (from Truth features), cluster into a rectangular lattice, and fill each cell by rule.

## Deliverables

`arc/op/engines/markers_grid.py` (≈400–500 LOC)

```python
@dataclass
class MarkersFitRc:
    engine: Literal["markers_grid"]
    marker_size: Tuple[int,int]        # e.g., (2,2)
    marker_color_set: List[int]        # from trainings (A1)
    centroids: Dict[str, List[Tuple[int,int]]]  # per-train centroids in Π frame
    grid_shape: Tuple[int,int]         # (#rows,#cols) cells
    cell_rule: str                     # e.g., "cell_color = marker_color" or "majority in cell"
    fit_verified_on: List[str]
    hash: str
```

```python
def fit_markers_grid(train_Xt_list, train_Y_raw_list, truth_train_list) -> Tuple[bool, MarkersFitRc]
def apply_markers_grid(test_Xt, truth_test, fit_rc: MarkersFitRc,
                       expected_shape: Optional[Tuple[int,int]]) -> Tuple[np.ndarray, Dict]
```

## Frozen algorithm

1. **Marker detection**: use **Truth** `exact_tile` and `bbox_mirror/rotate` to detect 2×2 solid blocks (nonzero in (\mathbb C)) and compute their **centroids** (exact integer centers). (D1, I1) 
2. **Grid inference**: sort centroids lex; compute **row/col bands** from distinct y/x sets (no epsilon). Validate a consistent rectangular lattice across all trainings; else ok=False with conflicts.
3. **Cell fill rule**: For each cell, choose a frozen rule (A1/C2), e.g., “if a marker occupies its corner, fill that cell with marker color; else background” or “strict majority of (\mathbb F) within bounding box”. Lock the rule string, verify all trainings reproduce.
4. **Apply**: compute centroids on test, assemble grid, paint cells via rule; set `final_shape` if S was absent; receipts include `schema_id`, centroids, grid shape, per-cell fills.

## PO / Red-team

* PO: centroids derived by exact mask ops; no clustering tolerance; lattice indices deterministic; conflicts logged.
* Red-team: jitter marker by 1 px → `centroids` change and acceptance fails (as desired); if two colors in a cell → strict majority tie path tested.

---

# WO-10E — **Slice-Stack Engine** (MAJOR)

**Intent**: Reconstruct outputs that are **concatenations of learned column (or row) slices**—i.e., an ordered dictionary of “slices” applied left→right or top→bottom (common in e.g., “barcodes” / stripe tasks).

## Deliverables

`arc/op/engines/slice_stack.py` (≈350–450 LOC)

```python
@dataclass
class SliceStackFitRc:
    engine: Literal["slice_stack"]
    axis: Literal["cols","rows"]
    slice_height: int; slice_width: int
    dict: Dict[bytes, bytes]           # input-slice (mask or color tuple) → output-slice bytes
    argmax_windows: List[Tuple[int,int]]  # for replication; optional
    decision_rule: str                 # e.g., "exact_slice_match_only"
    fit_verified_on: List[str]
    hash: str
```

```python
def fit_slice_stack(train_Xt_list, train_Y_raw_list, truth_train_list) -> Tuple[bool, SliceStackFitRc]
def apply_slice_stack(test_Xt, truth_test, fit_rc: SliceStackFitRc,
                      expected_shape: Optional[Tuple[int,int]]) -> Tuple[np.ndarray, Dict]
```

## Frozen algorithm

1. **Axis selection**: if truth `col_clusters` determines column grouping, set `axis="cols"`, else rows; record chosen axis—no heuristics.
2. **Slice extraction**: build per-axis **signature bytes** of fixed size (slice width/height from trained outputs) from Π-inputs; **signature schema** is frozen (document in receipts).
3. **Dictionary**: align train `k`-th signature slice to `k`-th output slice (Yᵢ); record mapping; verify all trainings; conflicts → ok=False.
4. **Apply**: compute signatures on test; for each, `dict[signature]` must exist; else `UNSEEN_SIGNATURE` fail-closed. Concatenate slices left→right (or top→bottom).
5. **Placement tie (C1)**: if multiple ways to segment tie (rare), produce candidates with `placement_keys` (center_L1,topmost,leftmost or skyline) and call WO-08.

---

# WO-10F — **Kronecker Tiling Engine** (MAJOR)

**Intent**: outputs that are repetitions of a learned base tile (T) — Kronecker product/tile replication (e.g., 652646ff). No search; exact equality.

## Deliverables

`arc/op/engines/kronecker.py` (≈300–400 LOC)

```python
@dataclass
class KroneckerFitRc:
    engine: Literal["kronecker"]
    base_tile: bytes         # serialized tile (r0×c0)
    tile_shape: Tuple[int,int]  # (r0, c0)
    reps: Tuple[int,int]        # (k_r, k_c) for each training
    fit_verified_on: List[str]
    hash: str
```

```python
def fit_kronecker(train_Xt_list, train_Y_raw_list, truth_train_list) -> Tuple[bool, KroneckerFitRc]
def apply_kronecker(test_Xt, truth_test, fit_rc: KroneckerFitRc,
                    expected_shape: Optional[Tuple[int,int]]) -> Tuple[np.ndarray, Dict]
```

## Frozen algorithm

1. **Base tile inference**: For each training `Yᵢ` with shape `(r_i,c_i)`, compute gcd-based candidate tile sizes `(r0|r_i, c0|c_i)`, and test the **smallest** tile whose Kronecker replication matches `Yᵢ` exactly. Cross-intersect across trainings to a single `(r0,c0)`; record `base_tile=Y₁[:r0,:c0]`; verify all trainings replicate to exactly `(r_i,c_i)` (no phase offsets).
2. **Apply**: compute repeats ((k_r, k_c) = (R*/r0, C*/c0)). Guard: must divide; else ok=False. Output is `T ⊗ 1_{k_r×k_c}`; receipts: `tile_hash`, `reps`, `final_shape`.
3. **Tie handling**: if multiple minimal tiles exist, choose **lex-smallest tile bytes** (frozen family tie).

---

## Runner (WO-11) integration points

* Add these engines into the frozen list after Column-Dict (as per plan).
* If **WO-02** returned `SHAPE_CONTRADICTION`, the engine’s `apply.final_shape` becomes `shape.R/C` with `shape_source:"engine"`.
* If engine `fit.ok=False`, record its diagnostics and continue to next engine, then to witness.

---

## Receipts & Determinism (J1)

Each engine emits:

* `fit_rc` with **full feature tables** (signatures, bands, counts), a `hash`, and `fit_verified_on`.
* `apply_rc` with **application-time evidence** (squashed signatures, tile reps, per-tile decisions), `final_shape`, and `apply_hash`.
* No dynamic knobs; any tie handled by **WO-08** with full **candidate tables** and frozen tie-chains (C1).
* Harness and runner log **section hashes** and `env_fingerprint`; double-run equality is mandatory. 

---

## Acceptance & Red-team (per engine)

* Provide 2–3 **concrete training/test IDs** from `docs/worked_examples.md` or `ids_test50.txt` (e.g., `3f7978a0` → Column-Dict already; add a border-frame task, a macro-tiling task, a kronecker task like `652646ff`).
* For each engine, build a small synthetic case in `scripts/wo10_{engine}_demo.py` and verify receipts and idempotence.
* **Red-team** the frozen rules (A1/A2/B1/C1/C2): change data to provoke empty-pullback, equal slicing, or tie; ensure receipts capture the violation (e.g., `UNSEEN_SIGNATURE`, `empty:true`, or `chosen_idx` via WO-08) and runs **fail** when they should.

---

## Nothing “extra” is being added

Everything above is already implied by the anchors:

* Engines are the finite Spec-B archetypes, implemented with **exact** methods and **frozen tie rules**, no thresholds. 
* They rely only on Truth features, Π, and S (reused), as specified. 
* Receipts and determinism are enforced per the **Determinism Addendum** (no dynamic tag sets; identity Δ excluded; section hashes; table_hash; env_fingerprint). 
