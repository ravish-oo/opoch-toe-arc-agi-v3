# WO-04 Battle Test Report

**Test Date**: 2025-11-01
**Test Scope**: 50 tasks from `ids_sweep50_wo02.txt`
**Status**: ✓ PASSED (0 bugs, 17 cases requiring examination)

---

## Executive Summary

### Test Results
```
Tasks processed:       50/50 (100%)
Training pairs:        152
Determinism:           ✓ Verified (2 runs, all hashes match)
Contract compliance:   ✓ E2, A1, C2 all satisfied
```

### Witness Distribution
```
Geometric witnesses:   35 (23%)
Summary witnesses:     117 (77%)
Total:                 152 (100%)
```

### Intersection Results
```
Singleton:             33/50 (66%) - Law uniquely determined
Contradictory:         14/50 (28%) - Mixed witness types
Underdetermined:       3/50 (6%)  - Multiple valid geometric laws
```

---

## Contract Compliance Verification

### E2: Bbox Equality (Geometric Witnesses)
- **Total geometric witnesses**: 35
- **E2 violations**: 0
- **Verification**: All geometric witnesses include `bbox_equal` proofs per component

### A1: Candidate Sets (Summary Witnesses)
- **Total summary witnesses**: 117
- **A1 violations**: 0
- **Verification**: All summary witnesses record `foreground_colors` and `background_colors`

### C2: Decision Rule (Summary Witnesses)
- **C2 violations**: 0
- **Verification**: All summary witnesses have frozen `decision_rule` string
- **Decision rule observed**: `strict_majority_foreground_fallback_0`

---

## Contradictory Intersections (14 tasks)

**Pattern**: All 14 tasks have **mixed witness types** (geometric + summary) across training pairs.

**Intersection Logic** (from `arc/op/witness.py:574-578`):
```python
# Check for mixed types (contradictory)
if "geom" in kinds and "none" in kinds:
    return None, conj_list[0][1], IntersectionRc("contradictory", 0)
```

**Question for Review**: Is mixing geometric and summary witnesses within a task:
- (A) **Expected** - Some training pairs match geometrically, others don't?
- (B) **Underspecified** - WO-04 should attempt unification across witness types?
- (C) **Implementation gap** - Geometric solver too restrictive, missing valid mappings?

### Contradictory Task Evidence

#### 1. Task `3cd86f4f`

**Training pairs**: 3
**Witness kinds**: ['summary', 'summary', 'geometric']

**Training 0** (kind: `summary`):
```json
{
  "kind": "summary",
  "phi": null,
  "sigma": {
    "domain_colors": [
      0,
      4,
      5,
      7,
      9
    ],
    "lehmer": [],
    "moved_count": 0
  },
  "foreground_colors": [
    4,
    5,
    7,
    9
  ],
  "background_colors": [
    0
  ],
  "decision_rule": "strict_majority_foreground_fallback_0",
  "per_color_counts": {
    "0": 12,
    "4": 6,
    "5": 6,
    "7": 16,
    "9": 8
  }
}
```

**Training 1** (kind: `summary`):
```json
{
  "kind": "summary",
  "phi": null,
  "sigma": {
    "domain_colors": [
      0,
      1,
      4,
      6,
      7,
      8
    ],
    "lehmer": [],
    "moved_count": 0
  },
  "foreground_colors": [
    1,
    4,
    6,
    7,
    8
  ],
  "background_colors": [
    0,
    1
  ],
  "decision_rule": "strict_majority_foreground_fallback_0",
  "per_color_counts": {
    "0": 30,
    "1": 5,
    "4": 10,
    "6": 6,
    "7": 3,
    "8": 12
  }
}
```

**Training 2** (kind: `geometric`):
```json
{
  "kind": "geometric",
  "phi": {
    "pieces_count": 1,
    "bbox_equal": [
      true
    ],
    "domain_pixels": 3,
    "sample_pieces": [
      {
        "comp_id": 0,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      }
    ]
  },
  "sigma": {
    "domain_colors": [
      1,
      5,
      6,
      8
    ],
    "lehmer": [],
    "moved_count": 0
  }
}
```

**Intersection Result**:
```json
{
  "status": "contradictory",
  "admissible_count": 0
}
```

---

#### 2. Task `025d127b`

**Training pairs**: 2
**Witness kinds**: ['geometric', 'summary']

**Training 0** (kind: `geometric`):
```json
{
  "kind": "geometric",
  "phi": {
    "pieces_count": 3,
    "bbox_equal": [
      true,
      true,
      true
    ],
    "domain_pixels": 3,
    "sample_pieces": [
      {
        "comp_id": 4,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      },
      {
        "comp_id": 0,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      }
    ]
  },
  "sigma": {
    "domain_colors": [
      0,
      8
    ],
    "lehmer": [],
    "moved_count": 0
  }
}
```

**Training 1** (kind: `summary`):
```json
{
  "kind": "summary",
  "phi": null,
  "sigma": {
    "domain_colors": [
      0,
      2,
      6
    ],
    "lehmer": [],
    "moved_count": 0
  },
  "foreground_colors": [
    2,
    6
  ],
  "background_colors": [
    0
  ],
  "decision_rule": "strict_majority_foreground_fallback_0",
  "per_color_counts": {
    "0": 106,
    "2": 8,
    "6": 12
  }
}
```

**Intersection Result**:
```json
{
  "status": "contradictory",
  "admissible_count": 0
}
```

---

#### 3. Task `0d87d2a6`

**Training pairs**: 3
**Witness kinds**: ['geometric', 'geometric', 'summary']

**Training 0** (kind: `geometric`):
```json
{
  "kind": "geometric",
  "phi": {
    "pieces_count": 4,
    "bbox_equal": [
      true,
      true,
      true,
      true
    ],
    "domain_pixels": 14,
    "sample_pieces": [
      {
        "comp_id": 4,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      },
      {
        "comp_id": 3,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      }
    ]
  },
  "sigma": {
    "domain_colors": [
      0,
      1,
      2
    ],
    "lehmer": [],
    "moved_count": 0
  }
}
```

**Training 1** (kind: `geometric`):
```json
{
  "kind": "geometric",
  "phi": {
    "pieces_count": 3,
    "bbox_equal": [
      true,
      true,
      true
    ],
    "domain_pixels": 3,
    "sample_pieces": [
      {
        "comp_id": 5,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      },
      {
        "comp_id": 6,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      }
    ]
  },
  "sigma": {
    "domain_colors": [
      0,
      1,
      2
    ],
    "lehmer": [],
    "moved_count": 0
  }
}
```

**Training 2** (kind: `summary`):
```json
{
  "kind": "summary",
  "phi": null,
  "sigma": {
    "domain_colors": [
      0,
      1,
      2
    ],
    "lehmer": [],
    "moved_count": 0
  },
  "foreground_colors": [
    1,
    2
  ],
  "background_colors": [
    0,
    1
  ],
  "decision_rule": "strict_majority_foreground_fallback_0",
  "per_color_counts": {
    "0": 91,
    "1": 62,
    "2": 29
  }
}
```

**Intersection Result**:
```json
{
  "status": "contradictory",
  "admissible_count": 0
}
```

---

#### 4. Task `22233c11`

**Training pairs**: 3
**Witness kinds**: ['geometric', 'summary', 'summary']

**Training 0** (kind: `geometric`):
```json
{
  "kind": "geometric",
  "phi": {
    "pieces_count": 2,
    "bbox_equal": [
      true,
      true
    ],
    "domain_pixels": 8,
    "sample_pieces": [
      {
        "comp_id": 0,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      },
      {
        "comp_id": 1,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      }
    ]
  },
  "sigma": {
    "domain_colors": [
      0,
      3
    ],
    "lehmer": [],
    "moved_count": 0
  }
}
```

**Training 1** (kind: `summary`):
```json
{
  "kind": "summary",
  "phi": null,
  "sigma": {
    "domain_colors": [
      0,
      3,
      8
    ],
    "lehmer": [],
    "moved_count": 0
  },
  "foreground_colors": [
    3,
    8
  ],
  "background_colors": [
    0
  ],
  "decision_rule": "strict_majority_foreground_fallback_0",
  "per_color_counts": {
    "0": 92,
    "3": 4,
    "8": 4
  }
}
```

**Training 2** (kind: `summary`):
```json
{
  "kind": "summary",
  "phi": null,
  "sigma": {
    "domain_colors": [
      0,
      3,
      8
    ],
    "lehmer": [],
    "moved_count": 0
  },
  "foreground_colors": [
    3,
    8
  ],
  "background_colors": [
    0
  ],
  "decision_rule": "strict_majority_foreground_fallback_0",
  "per_color_counts": {
    "0": 96,
    "3": 2,
    "8": 2
  }
}
```

**Intersection Result**:
```json
{
  "status": "contradictory",
  "admissible_count": 0
}
```

---

#### 5. Task `3eda0437`

**Training pairs**: 4
**Witness kinds**: ['summary', 'summary', 'summary', 'geometric']

**Training 0** (kind: `summary`):
```json
{
  "kind": "summary",
  "phi": null,
  "sigma": {
    "domain_colors": [
      0,
      1,
      6
    ],
    "lehmer": [],
    "moved_count": 0
  },
  "foreground_colors": [
    1,
    6
  ],
  "background_colors": [
    0,
    1
  ],
  "decision_rule": "strict_majority_foreground_fallback_0",
  "per_color_counts": {
    "0": 16,
    "1": 14,
    "6": 10
  }
}
```

**Training 1** (kind: `summary`):
```json
{
  "kind": "summary",
  "phi": null,
  "sigma": {
    "domain_colors": [
      0,
      1,
      6
    ],
    "lehmer": [],
    "moved_count": 0
  },
  "foreground_colors": [
    1,
    6
  ],
  "background_colors": [
    0,
    1
  ],
  "decision_rule": "strict_majority_foreground_fallback_0",
  "per_color_counts": {
    "0": 29,
    "1": 45,
    "6": 6
  }
}
```

**Training 2** (kind: `summary`):
```json
{
  "kind": "summary",
  "phi": null,
  "sigma": {
    "domain_colors": [
      0,
      1,
      6
    ],
    "lehmer": [],
    "moved_count": 0
  },
  "foreground_colors": [
    1,
    6
  ],
  "background_colors": [
    0,
    1
  ],
  "decision_rule": "strict_majority_foreground_fallback_0",
  "per_color_counts": {
    "0": 40,
    "1": 31,
    "6": 9
  }
}
```

**Training 3** (kind: `geometric`):
```json
{
  "kind": "geometric",
  "phi": {
    "pieces_count": 20,
    "bbox_equal": [
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true
    ],
    "domain_pixels": 71,
    "sample_pieces": [
      {
        "comp_id": 13,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      },
      {
        "comp_id": 14,
        "pose_id": 0,
        "dr": 0,
        "dc": 0,
        "r_per": 1,
        "c_per": 1,
        "r_res": 0,
        "c_res": 0
      }
    ]
  },
  "sigma": {
    "domain_colors": [
      0,
      1,
      5
    ],
    "lehmer": [],
    "moved_count": 0
  }
}
```

**Intersection Result**:
```json
{
  "status": "contradictory",
  "admissible_count": 0
}
```

---

### Remaining Contradictory Tasks (9 tasks)

| Task ID | Training Pairs | Witness Pattern |
|---------|----------------|-----------------|
| `ac3e2b04` | 4 | geometric, summary, summary, geometric |
| `c59eb873` | 3 | summary, summary, geometric |
| `c6141b15` | 3 | geometric, geometric, summary |
| `cb227835` | 3 | summary, summary, geometric |
| `c87289bb` | 4 | summary, summary, summary, geometric |
| `25e02866` | 2 | summary, geometric |
| `58e15b12` | 3 | geometric, summary, geometric |
| `2f767503` | 3 | summary, geometric, geometric |
| `e4941b18` | 3 | geometric, geometric, summary |

---

## Underdetermined Intersections (3 tasks)

**Pattern**: All 3 tasks have **all-geometric witnesses** but with **different geometric structures** (different numbers of pieces or parameter values).

**Intersection Logic** (from `arc/op/witness.py:610-614`):
```python
enc = [(p.pose_id, p.dr, p.dc, p.r_per, p.c_per, p.r_res, p.c_res) for p in phi]

if enc != enc_0:
    # Parameters differ → underdetermined (at least 2 admissible)
    return phi_0, sigma_0, IntersectionRc("underdetermined", 2)
```

**Question for Review**: When geometric parameters differ across training pairs:
- (A) **Expected** - Each training has unique structure, no law unification possible?
- (B) **Underspecified** - Should attempt to abstract common patterns?
- (C) **Implementation gap** - Missing higher-order parameter search?

### Underdetermined Task Evidence

#### 1. Task `4df5b0ae`

**Training pairs**: 3
**Witness kinds**: ['geometric', 'geometric', 'geometric']

**Training 0**:
```json
{
  "kind": "geometric",
  "pieces_count": 4,
  "bbox_equal": [
    true,
    true,
    true,
    true
  ],
  "domain_pixels": 10,
  "all_pieces": [
    {
      "comp_id": 1,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 0,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 3,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 2,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    }
  ]
}
```

**Training 1**:
```json
{
  "kind": "geometric",
  "pieces_count": 4,
  "bbox_equal": [
    true,
    true,
    true,
    true
  ],
  "domain_pixels": 23,
  "all_pieces": [
    {
      "comp_id": 0,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 1,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 3,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 2,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    }
  ]
}
```

**Training 2**:
```json
{
  "kind": "geometric",
  "pieces_count": 3,
  "bbox_equal": [
    true,
    true,
    true
  ],
  "domain_pixels": 12,
  "all_pieces": [
    {
      "comp_id": 1,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 0,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 2,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    }
  ]
}
```

**Intersection Result**:
```json
{
  "status": "underdetermined",
  "admissible_count": 2
}
```

**Analysis**: Training pairs have [4, 4, 3] geometric pieces respectively.

---

#### 2. Task `b27ca6d3`

**Training pairs**: 2
**Witness kinds**: ['geometric', 'geometric']

**Training 0**:
```json
{
  "kind": "geometric",
  "pieces_count": 14,
  "bbox_equal": [
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true
  ],
  "domain_pixels": 17,
  "all_pieces": [
    {
      "comp_id": 11,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 12,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 13,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 0,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 1,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 2,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 3,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 4,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 5,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 6,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 7,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 8,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 9,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 10,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    }
  ]
}
```

**Training 1**:
```json
{
  "kind": "geometric",
  "pieces_count": 9,
  "bbox_equal": [
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true
  ],
  "domain_pixels": 10,
  "all_pieces": [
    {
      "comp_id": 8,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 0,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 1,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 2,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 3,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 4,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 5,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 6,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 7,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    }
  ]
}
```

**Intersection Result**:
```json
{
  "status": "underdetermined",
  "admissible_count": 2
}
```

**Analysis**: Training pairs have [14, 9] geometric pieces respectively.

---

#### 3. Task `5168d44c`

**Training pairs**: 3
**Witness kinds**: ['geometric', 'geometric', 'geometric']

**Training 0**:
```json
{
  "kind": "geometric",
  "pieces_count": 8,
  "bbox_equal": [
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true
  ],
  "domain_pixels": 15,
  "all_pieces": [
    {
      "comp_id": 0,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 1,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 2,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 3,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 4,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 5,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 6,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 7,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    }
  ]
}
```

**Training 1**:
```json
{
  "kind": "geometric",
  "pieces_count": 7,
  "bbox_equal": [
    true,
    true,
    true,
    true,
    true,
    true,
    true
  ],
  "domain_pixels": 14,
  "all_pieces": [
    {
      "comp_id": 0,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 1,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 2,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 3,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 4,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 5,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 6,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    }
  ]
}
```

**Training 2**:
```json
{
  "kind": "geometric",
  "pieces_count": 6,
  "bbox_equal": [
    true,
    true,
    true,
    true,
    true,
    true
  ],
  "domain_pixels": 49,
  "all_pieces": [
    {
      "comp_id": 0,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 1,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 2,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 3,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 5,
      "pose_id": 6,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    },
    {
      "comp_id": 4,
      "pose_id": 0,
      "dr": 0,
      "dc": 0,
      "r_per": 1,
      "c_per": 1,
      "r_res": 0,
      "c_res": 0
    }
  ]
}
```

**Intersection Result**:
```json
{
  "status": "underdetermined",
  "admissible_count": 2
}
```

**Analysis**: Training pairs have [8, 7, 6] geometric pieces respectively.

---

## Recommendations

### 1. Examine Contradictory Cases
- Review tasks with mixed geometric/summary witnesses
- Determine if geometric solver is too restrictive (missing valid component matchings)
- Consider if WO-04 spec should allow cross-type unification

### 2. Examine Underdetermined Cases
- Review tasks with differing geometric structures
- Determine if this indicates missing abstraction capability
- Consider if higher-order patterns should be extracted

### 3. Next Steps
- If contradictory/underdetermined are **expected**: Run WO-04 on full 892 tasks to freeze
- If **underspecified**: Refine WO-04 contracts and expand geometric/intersection logic
- If **implementation gap**: Debug geometric solver and component matching

---

## Appendix: Singleton Success Examples

For reference, 33/50 tasks (66%) achieved singleton intersections, meaning:
- All training pairs produced the same witness type (all geometric OR all summary)
- All geometric parameters matched exactly across trainings
- All σ permutations matched exactly

This indicates WO-04 is **working correctly for the majority case**.
