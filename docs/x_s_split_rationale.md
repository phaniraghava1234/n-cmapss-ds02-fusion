# Why our variable split follows the target paper, not the dataset paper

**Phase 1 deliverable. Written 2026-11-05, verified against the release 2026-08-19.**

The two papers behind this project describe DS02's variables differently. This document
records which one we follow, why, and what the arithmetic proves.

---

## The disagreement

| Group | Target paper (2022) | Dataset paper (2021) | Released file |
|---|---|---|---|
| `w` scenario descriptors | 4 | 4 (Table 4) | 4 — `W_var` |
| `x_s` measured signals | **16** (Table 2 minus `w`) | 14 (Table 5) | 14 — `X_s_var` |
| `x_v` virtual sensors | **11** (Table 3) | 14 (Table 6) | 14 — `X_v_var` |
| `θ` health parameters | **3** (Table 4) | 10 (Table 7) | 10 — `T_var` |

Three separate differences, not one.

### 1. T40 and P30 move from virtual to measured

The release places T40 (burner outlet temperature) and P30 (HPC outlet pressure) in
`X_v_var`. Target paper Table 2 lists both among the condition monitoring signals.

This is a judgement about instrumentation — whether a real engine has sensors at those
stations. The target paper says yes; the dataset paper says no. Since the paper's framework
treats `x_s` as "what you can measure" and `x_v` as "what the surrogate must predict," the
choice changes which quantities D has to infer.

### 2. `phi` is dropped entirely

Removing T40 and P30 from the release's 14 virtual sensors leaves 12, but target paper
Table 3 lists **11**. The missing one is `phi`.

Dataset paper Table 6 row 14 defines `phi` as the ratio of fuel flow to Ps30 — a derived
quantity, computed from two values already present in the input set, not an independent
measurement. The target paper lists it in neither Table 2 nor Table 3.

### 3. Only 3 of 10 health parameters are active

The release carries efficiency and flow modifiers for all five rotating components. Target
paper Table 4 names three: `HPT_eff_mod`, `LPT_eff_mod`, `LPT_flow_mod`. DS02 has two
failure modes — HPT efficiency degradation, and HPT efficiency combined with LPT efficiency
and flow — so the other seven modifiers stay at nominal throughout.

---

## Decision: follow the target paper

Locked as decision **V1** in `PROJECT_PLAN.md` §1; recorded as Deviation 2 in
`REPLICATION_NOTES.md`.

**We replicate the target paper, not the dataset paper.** Its split is the input
specification for the prognostics network G. Using any other grouping changes G's input
dimensionality and makes our results incomparable with its Table 5 — which decision D8
depends on.

### The arithmetic settles it

Target paper §4.3 (p. 7) states two input dimensions explicitly. Both close exactly under
this reading and under no other:

```
baseline  [w, x_s]                 =  4 + 16                 = 20   ✓ "20 inputs"
hybrid    [w, x_s, x̂_s, x̂_v, θ̂]   =  4 + 16 + 16 + 11 + 3    = 50   ✓ "50 inputs"
```

Had `phi` been retained, the hybrid total would be 51. Had T40 and P30 stayed virtual,
the baseline would be 18 and the hybrid still 50 — but the baseline figure would then
contradict the paper. Only the reading above satisfies both stated numbers simultaneously.

It also fixes the surrogate D interface (§4.2, Appendix B):

```
D input   = len(w) + len(x_s) + len(θ)   =  4 + 16 + 3   = 23
D output  = len(x_s) + len(x_v)          =      16 + 11  = 27
```

---

## Verified against the file

`tests/test_data_loader.py` asserts this split. Checked 2026-08-19 against
`N-CMAPSS_DS02-006.h5`:

```
promoted x_v -> x_s per decision V1 : ['P30', 'T40']
in file but NOT used by the paper   : ['phi']
required by paper but NOT in file   : []
```

Every signal the paper requires exists in the release, and `phi` is the only released
column the paper uses nowhere.

---

## How this is implemented

`src/data_loader.py` stores **every released column** — all 14 `x_s`, all 14 `x_v`
including `phi`, all 10 `θ`. The paper's selection is applied at model-input time through
named constants:

```python
W_VARS       # 4   target paper Table 2 rows 1-3, 7
X_S_PAPER    # 16  Table 2 minus w, T40 and P30 promoted
X_V_PAPER    # 11  Table 3, phi excluded
THETA_PAPER  # 3   Table 4
```

Keeping storage faithful to the release has three benefits: the EDA notebooks can plot
anything, the v3 sampling-rate ablation needs no separate pipeline, and the deviation
is visible in code rather than baked irreversibly into a file.

---

## A note on terminology

Our loader's column names come from the release, so `phi`, T40 and P30 appear in the
Parquet even though the paper's model never sees `phi`. Anyone comparing our schema against
dataset paper Tables 5–6 will find them consistent; anyone comparing our *model inputs*
against target paper Tables 2–4 will also find them consistent. The two views differ, and
that is intentional.

---

## Sources

- Arias Chao et al. (2022), *Reliability Engineering & System Safety* 217, 107961 —
  Tables 2, 3, 4 (p. 8), §4.3 (p. 7), Appendix B (p. 16)
- Arias Chao et al. (2021), *Data* 6(1):5 — Tables 4, 5, 6, 7 (pp. 6–7)
