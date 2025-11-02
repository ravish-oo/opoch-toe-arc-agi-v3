# ARC-AGI Task Sweep Files

**Purpose**: Organized task ID lists for testing different WO (Work Order) scopes.

---

## File Structure

### Original Sweep Files (All Tasks)

**`ids_sweep50.txt`** (50 tasks)
- Random sample from 1000 training tasks
- Includes 3 worked examples + 47 random tasks
- Mixed: WO-02 scope + content-dependent

**`ids_sweep100.txt`** (100 tasks)
- Larger random sample
- **ZERO overlap with ids_sweep50.txt**
- Mixed: WO-02 scope + content-dependent

**`ids_all1000.txt`** (1000 tasks)
- Complete ARC-AGI training set
- All tasks from the corpus
- Mixed: WO-02 scope + content-dependent

---

### Content-Dependent Tasks (WO-05+ Scope)

**`ids_content_dependent.txt`** (108 tasks)
- Tasks requiring TRUTH layer for shape synthesis
- Output dimensions depend on INPUT CONTENT (not just H,W)
- Will be addressed in WO-05+ implementation

**Why separated:**
- These tasks CANNOT pass WO-02 (shape algebra only)
- Will also fail WO-03/WO-04 (pipeline dependency)
- Need content features: components, colors, symmetries, etc.

**Categories:**
1. Same input → different outputs (48 tasks, 44%)
2. Non-integer AFFINE coefficients (32 tasks, 30%)
3. No integer AFFINE solution (28 tasks, 26%)

**See:** `WO02_CONTENT_DEPENDENT_ANALYSIS.md` for detailed analysis

---

### Cleaned Sweep Files (WO-02 Scope Only)

**`ids_sweep50_wo02.txt`** (34 tasks)
- Original 50 tasks **minus** 16 content-dependent
- ✅ All 34 should PASS WO-02
- ✅ Safe for WO-03/WO-04 testing

**`ids_sweep100_wo02.txt`** (91 tasks)
- Original 100 tasks **minus** 9 content-dependent
- ✅ All 91 should PASS WO-02
- ✅ Safe for WO-03/WO-04 testing

**`ids_all1000_wo02.txt`** (892 tasks)
- Original 1000 tasks **minus** 108 content-dependent
- ✅ All 892 should PASS WO-02
- ✅ Safe for WO-03/WO-04 testing

---

## Usage Guide

### For WO-02 Testing (Shape Synthesis)

**✅ Use cleaned files:**
```bash
# Should achieve 100% pass rate
python scripts/run_wo.py --wo WO-02 --subset ids_sweep50_wo02.txt
python scripts/run_wo.py --wo WO-02 --subset ids_sweep100_wo02.txt
python scripts/run_wo.py --wo WO-02 --subset ids_all1000_wo02.txt
```

**Expected:**
- ✅ 34/34 pass (50-task sweep)
- ✅ 91/91 pass (100-task sweep)
- ✅ 892/892 pass (1000-task sweep)

**⚠️ Don't use original files for WO-02:**
```bash
# These will have failures (content-dependent tasks)
python scripts/run_wo.py --wo WO-02 --subset ids_sweep50.txt      # 34/50 pass
python scripts/run_wo.py --wo WO-02 --subset ids_sweep100.txt     # 91/100 pass
python scripts/run_wo.py --wo WO-02 --subset ids_all1000.txt      # 892/1000 pass
```

---

### For WO-03 Testing (Components)

**✅ Use cleaned files:**
```bash
# These tasks have valid output shapes (R,C) from WO-02
python scripts/run_wo.py --wo WO-03 --subset ids_sweep50_wo02.txt
python scripts/run_wo.py --wo WO-03 --subset ids_sweep100_wo02.txt
python scripts/run_wo.py --wo WO-03 --subset ids_all1000_wo02.txt
```

**Why:** WO-03 requires output size (R,C) from WO-02. Content-dependent tasks fail WO-02, so they block WO-03.

---

### For WO-04 Testing (Witness)

**✅ Use cleaned files:**
```bash
# These tasks have components from WO-03
python scripts/run_wo.py --wo WO-04 --subset ids_sweep50_wo02.txt
python scripts/run_wo.py --wo WO-04 --subset ids_sweep100_wo02.txt
python scripts/run_wo.py --wo WO-04 --subset ids_all1000_wo02.txt
```

**Why:** WO-04 requires components from WO-03. Content-dependent tasks fail WO-02 → WO-03, so they block WO-04.

---

### For WO-05+ Testing (TRUTH Layer)

**✅ Test content-dependent tasks:**
```bash
# After WO-05+ implementation, these should start passing
python scripts/run_wo.py --wo WO-05 --subset ids_content_dependent.txt
```

**Expected:** Some/all of the 108 tasks should pass once TRUTH features are available.

**Also test full set:**
```bash
# Should now achieve higher coverage (892 + X out of 1000)
python scripts/run_wo.py --wo WO-05 --subset ids_all1000.txt
```

---

## Quick Reference

| File | Tasks | Content-Dep | WO-02 Pass | Use For |
|------|-------|-------------|------------|---------|
| `ids_sweep50.txt` | 50 | 16 | 34 | ❌ Not recommended |
| `ids_sweep100.txt` | 100 | 9 | 91 | ❌ Not recommended |
| `ids_all1000.txt` | 1000 | 108 | 892 | ❌ Not recommended |
| `ids_sweep50_wo02.txt` | 34 | 0 | 34 | ✅ WO-02, WO-03, WO-04 |
| `ids_sweep100_wo02.txt` | 91 | 0 | 91 | ✅ WO-02, WO-03, WO-04 |
| `ids_all1000_wo02.txt` | 892 | 0 | 892 | ✅ WO-02, WO-03, WO-04 |
| `ids_content_dependent.txt` | 108 | 108 | 0 | ✅ WO-05+ validation |

---

## Verification

**To verify file consistency:**
```bash
# Count tasks
wc -l ids_*.txt

# Verify no overlap between WO-02 and content-dependent
cat ids_sweep50_wo02.txt ids_content_dependent.txt | sort | uniq -d
# Should be empty (no duplicates)

# Verify union covers full set
cat ids_all1000_wo02.txt ids_content_dependent.txt | sort | uniq | wc -l
# Should be 1000
```

---

## History

**2025-11-01: WO-02 Battle Test**
- Ran WO-02 on 1000 tasks
- Found 892 pass, 108 fail
- Verified ALL 108 failures are content-dependent (algebraic proof)
- Created cleaned sweep files for WO-03/WO-04 testing
- Zero bugs found ✅

**Status:** WO-02 frozen at 100% coverage on its defined scope

---

## Related Documentation

- `WO02_CONTENT_DEPENDENT_ANALYSIS.md` - Detailed analysis of 108 content-dependent tasks
- `wo02_1000_failure_verification.json` - Algebraic verification data
- `docs/anchors/00_math_spec.md` - Mathematical specification
- `docs/anchors/01_engineering_spec.md` - Engineering specification
- `docs/anchors/02_determinism_addendum.md` - Frozen parameters and encodings

---

**Generated:** 2025-11-01
**Maintainer:** Reviewer/Tester
**Status:** FROZEN ✅
