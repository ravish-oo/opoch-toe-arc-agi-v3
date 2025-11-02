# WO-04 FROZEN - Full Scale Test Results

**Test Date**: 2025-11-01
**Test Scope**: 892 tasks from `ids_all1000_wo02.txt` (all WO-02 scope)
**Status**: ✓ **FROZEN** - 0 bugs, 0 unexpected failures

---

## Test Progression

| Phase | Tasks | Result | Findings |
|-------|-------|--------|----------|
| Initial sweep | 50 | ✓ PASSED | Identified contradictory/underdetermined patterns |
| Full scale | 892 | ✓ PASSED | Confirmed patterns, no new issues |

---

## Full Scale Results (892 tasks)

### Execution Summary
```
Tasks processed:       892/892 (100%)
Training pairs:        2,863
Receipts written:      1,784 (2 runs × 892)
Determinism:           ✓ VERIFIED (all hashes match)
Contract compliance:   ✓ E2, A1, C2 (0 violations)
Environment:           macOS-15.3-arm64-arm-64bit (little)
```

### Witness Distribution
```
Geometric witnesses:   723 (25.2%)
Summary witnesses:     2,140 (74.8%)
Total:                 2,863 (100%)
Other/unexpected:      0
```

### Intersection Results
```
Singleton:             601/892 (67.4%) - Law uniquely determined
Contradictory:         202/892 (22.6%) - Mixed witness types (EXPECTED)
Underdetermined:       89/892 (10.0%) - Geometric structure diff (EXPECTED)
Unexpected statuses:   0
```

---

## Scale Comparison (50 → 892 tasks)

| Metric | 50 tasks | 892 tasks | Δ |
|--------|----------|-----------|---|
| Singleton % | 66.0% | 67.4% | +1.4% |
| Contradictory % | 28.0% | 22.6% | -5.4% |
| Underdetermined % | 6.0% | 10.0% | +4.0% |

**Analysis**: Distribution remains stable at scale. No new patterns emerged.

---

## Failure Analysis

### Crashes and Exceptions
- **Count**: 0
- **Verification**: All 892 tasks processed cleanly

### Unexpected Witness Types
- **Count**: 0
- **Verification**: All witnesses are 'geometric' or 'summary' (no 'none' or other)

### Unexpected Intersection Statuses
- **Count**: 0
- **Verification**: Only singleton/contradictory/underdetermined (per spec)

### Contract Violations
- **E2 (Bbox Equality)**: 0 violations
- **A1 (Candidate Sets)**: 0 violations
- **C2 (Decision Rule)**: 0 violations

### Determinism Issues
- **Count**: 0
- **Verification**: All 892 tasks have matching hashes across 2 runs

---

## Non-Singleton Cases (Expected Behavior)

### Contradictory (202 tasks, 22.6%)
**Pattern**: Mixed witness types (some trainings geometric, others summary)

**Why expected**: Per `arc/op/witness.py:574-578`, mixing geometric and summary is contradictory by design.

**Resolution path**: WO-05 (Truth-assisted φ) + WO-10/11 (finite engines)

### Underdetermined (89 tasks, 10.0%)
**Pattern**: All-geometric witnesses but different piece structures

**Why expected**: Per `arc/op/witness.py:610-614`, differing geometric parameters → underdetermined.

**Resolution path**: WO-05 (higher-order patterns) + Truth coframes

---

## Freeze Declaration

**WO-04 is FROZEN** at 892-task coverage with the following guarantees:

1. ✓ **Determinism**: 100% (all tasks produce identical hashes across runs)
2. ✓ **Contract compliance**: E2, A1, C2 satisfied (0 violations)
3. ✓ **Fail-closed**: No silent failures, crashes, or unexpected patterns
4. ✓ **Receipt coverage**: All decisions backed by receipts
5. ✓ **Expected behavior**: Non-singleton cases correctly identify tasks needing WO-05+

**Next WOs**:
- WO-05 (Truth): Enable truth-assisted φ solving for contradictory cases
- WO-10 (Engines): Finite law engines for content-dependent patterns
- WO-11 (Hybrid runner): Decision tree integrating witness + engines

---

## Files

- `out/receipts/WO-04_run.jsonl` - 1,784 receipts (892 tasks × 2 runs)
- `wo04_sweep892_run.log` - Execution log
- `ids_all1000_wo02.txt` - Task list (892 WO-02 scope tasks)
- `WO04_BATTLE_TEST_REPORT.md` - Detailed 50-task analysis with receipt evidence
- `wo04_raw_receipts_for_examination.json` - Raw receipts for 17 example cases
