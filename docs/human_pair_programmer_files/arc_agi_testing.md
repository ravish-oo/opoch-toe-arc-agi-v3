## Reviewer prompt — TDD with `ids.txt` ( basis  docs/worked_examples.md )

**Context files to keep open:**
`docs/anchors/00_math_spec.md`, `docs/anchors/01_engineering_spec.md`, `docs/anchors/02_determinism_addendum.md`, `docs/common_mistakes.md`, and `docs/worked_examples.md` (oracle outputs and notes). 

**Goal:** For each WO, run the harness on **`data/ids.txt`** (curated to worked examples), verify **determinism** and **receipts**, and where applicable compare outputs to the oracle in `worked_examples.md`.

### Commands (for every WO)

```bash
# Run the WO twice (determinism baked in); writes JSONL receipts
python -m scripts.run_wo --wo WO-0X --data data/raw/ --subset data/ids.txt \
  --out out/ --receipts out/receipts/

# Optional: compare two runs or branches
python -m scripts.check_receipts out/receipts/WO-0X_run.jsonl out/receipts/WO-0X_run.jsonl
```

### What to check (per WO)

* **Determinism:** The two runs produce **identical JSONL receipts**. Any diff = fail (no timestamps, no random ids).

* **Compliance:** Use the **BLOCKER items** in §14 of the addendum (Π/S/Truth/Witness/Tie/Meet/final).

* **WO-specific receipts:**

  * **WO-01 Π:** `scope="inputs_only"`, per-grid `pi2_hash == hash(Π(G))`, `roundtrip_hash == hash(G)`.
  * **WO-02 S:** `branch_byte`, `params_bytes_hex`, `(R,C)`, `verified_train_ids` includes **all**; if COUNT: `qual_id` is registered (`q_rows` or `q_hw_bilinear`) and `qual_hash` matches; PERIOD axis_code recorded when used.
  * **WO-03 Components:** `connectivity="4"`, full invariant tuples; `stable_match` identical across runs.
  * **WO-04 Witness:** For geometric, `bbox_equal=True` per piece; for summary, **A1/C2** receipts present (`foreground_colors`, `background_colors`, `per_color_counts`, `decision_rule`). `IntersectionRc.status` ∈ {singleton, underdetermined, contradictory}.

* **Oracle outputs (when WO produces Y):** For WOs that yield final Y (e.g., WO-09 via Meet or WO-11 runner), compare the produced outputs for the ids in `data/ids.txt` against the expected matrices in `docs/worked_examples.md`. Any mismatch = fail. 

### Pass/Fail criteria

* **Pass:** determinism OK, all BLOCKER receipts present and correct, and (when applicable) outputs match oracle for these ids.
* **Fail-closed:** if a proof cannot be established (e.g., shape without equality, mixed summary/geometric), receipts must show `contradictory` or absence of a candidate. Do **not** accept heuristics.

### Notes

* Add/remove ids in `data/ids.txt` to target edge families (bands/slices/kronecker/2×2-completion/row-from-5/etc.)—all have worked examples in `docs/worked_examples.md`. 
* Use `docs/common_mistakes.md` pre-flight bullets for each WO (e.g., **WO-02: E1 + period-axis tie; WO-03: D2; WO-04: E2, A1, C2**). Block the PR if any apply.

---

If you want, I can also drop a tiny wrapper script `scripts/run_worked_examples.sh` that runs WO-01…WO-11 sequentially on `ids.txt` and prints a one-line PASS/FAIL per WO based on receipts.
