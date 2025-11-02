# Stats Summary 
## Intersection Results on 50 sweep of WO-04
  Singleton:       33/50 (66%) - Law uniquely determined
  Contradictory:   14/50 (28%) - Mixed geometric/summary (EXPECTED)
  Underdetermined:  3/50 (6%)  - Multiple valid laws (EXPECTED)

## Full Scale Test Results on 892 sweep of WO-04

  Tasks processed:       892/892 (100%)
  Training pairs:        2,863
  Determinism:           ✓ VERIFIED (all hashes match)
  Contract compliance:   ✓ E2, A1, C2 (0 violations)

  Intersection Distribution

  Singleton:             601/892 (67.4%) ✓
  Contradictory:         202/892 (22.6%) ✓ EXPECTED
  Underdetermined:       89/892 (10.0%) ✓ EXPECTED
  Unexpected statuses:   0

##Summary
Short version: those 17 “non-singleton” results are **exactly what WO-04 is supposed to surface**, not bugs. They’re the receipts-proven signal that some tasks can’t be unified by a single (φ, σ) law across all trainings without more structure. The battle-test and raw receipts back that up: 14 tasks are **mixed witness types ⇒ `contradictory` by contract**, and 3 tasks are **all-geometric but with different piece layouts ⇒ `underdetermined`**; both are flagged as expected in the report and in the intersection logic you shipped.   

### What they are

* **Contradictory (14/50, e.g., `3cd86f4f`, `0d87d2a6`, …):** Some trainings admit a geometric φ with E2-proofed bbox equality, others only fit a summary law (A1/C2 receipts). Per WO-04, mixing “geom” and “summary” makes the cross-train intersection **`contradictory`**; that’s the designed fail-closed outcome, not a missed implementation. (See the receipt snippets and the code path noted in the report.)  

* **Underdetermined (3/50, e.g., `4df5b0ae`, `b27ca6d3`, `5168d44c`):** All trainings yield geometric φ, but **piece schemas differ** (e.g., 4/4/3 pieces), so the parameter vectors don’t unify. WO-04 correctly returns **`undetermined`** with `admissible_count≥2`. Again, per spec this is expected, not an error.  

### Where they’re handled next (not WO-04)

* **WO-05 (Truth) → Truth-assisted unification.** Use frozen features (row/col bands, per-color overlaps, exact clusters) to define **coframes** that make previously “summary” trainings admit a geometric φ (e.g., row-coframes for row-shift families). Then extend WO-04 to **consume Truth**: if component-coframes fail, retry geometric φ on truth-coframes and re-intersect. This is how you turn many `contradictory` into `singleton`. (Planned enhancement; not automatic without coding the WO-04→WO-05 handoff.) 

* **WO-08 (Tie-break L).** Only relevant if you end up with **multiple identical** admissible φ/σ parameterizations; then the frozen cost tuple resolves it. Your three “underdetermined” examples differ in piece counts, so they’re not just relabelings; they’ll typically need **higher-order patterning from Truth/engines**, not a tie-break. 

* **WO-10/WO-11 (Finite engines + hybrid runner).** Some “summary vs geometric” conflicts are actually **finite archetypes** (e.g., column-dictionary / macro-tiling / pooled-blocks). Per the hybrid plan, have the runner try those finite engines first; only fall back to witness if none fits. That resolves many of the `contradictory` cases without changing WO-04 at all. (E.g., `3f7978a0` via a column-dictionary engine.) This lives in **WO-10 (engine)** and **WO-11 (decision tree)**, not in WO-04. 

### Bottom line

* **WO-04 is already “frozen” and behaving to spec**: 100% determinism; E2/A1/C2 receipts present; non-singleton intersections are legitimate signals, not defects. 
* To **collapse contradictions** and **resolve real ambiguities**, the work belongs in **future WOs**:

  * add **Truth-assisted coframes** to WO-04 after WO-05 lands,
  * integrate **finite law engines** in WO-10 and wire the **hybrid runner** in WO-11.

If you want, I can draft the WO-05→WO-04 “truth-assisted φ” patch outline next (where to pass `row_clusters/col_clusters` into `solve_witness_for_pair` and how to receipt the coframe retry).
