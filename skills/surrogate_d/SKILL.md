# SKILL: Surrogate Model D

**Status:** STUB — to be filled in during Phase 2 implementation.

**When to load this skill:** any session touching Phase 2 (surrogate D training) or Phase 3 (UKF using D as forward model).

---

## Scope of this skill

Methodology for building and training D, the DNN surrogate that approximates the CMAPSS thermodynamic model F. D is used inside the UKF loop as a fast replacement for F.

Reference: Arias Chao et al. 2022 §4.2 and Appendix B.

---

## What we know before Phase 2 starts

**Architecture (paper §4.2, Appendix B):**
- MLP with 4 layers (L = 4)
- Hidden units: n¹ = n² = n³ = 100
- Output layer: n_L = n (matches sensor vector dimension)
- Activation: ReLU throughout hidden layers
- Output activation: identity (linear)

**Input signature (paper §3.1, Eq. 5):**
```
[x̂_s^(t), x̂_v^(t)] = D(w^(t), x_s^(t-1), θ^(t))
```

**Training data source (this project, per REPLICATION_NOTES Deviation 1):**
- DS02 healthy segments per unit (rows where t < t_s from Table 1)
- θ labels approximated from dataset paper §3.1 degradation model
- 5 training units for D fitting, 1 held out for D validation

---

## To be filled in during Phase 2

### Concrete input/output dimensioning

- Input dim after concatenation
- Output dim (x_s + x_v)
- Any normalization scheme applied

### Training loop specifics

- Optimizer choice (Adam vs SGD)
- Learning rate schedule
- Batch size
- Early stopping criterion
- Weight initialization

### Validation gates that must pass

- Held-out unit MSE threshold
- Numerical stability check
- Physical plausibility check (D should not produce negative pressures, etc.)

### Gotchas encountered

- (To be recorded as they surface)

### Interface for Phase 3 consumers

- Function signature D expects the UKF to call
- Batch vs single-sample inference
- CPU vs GPU inference in the UKF hot loop

### Known limitations

- Fidelity gap per REPLICATION_NOTES Deviation 1
- Any additional gaps discovered

---

## Rules while filling this in (per replication_discipline SKILL)

- Cite paper section for every architectural choice traceable to the paper
- Flag every choice not specified by the paper in REPLICATION_NOTES
- No PyTorch defaults silently applied
- Test coverage: at minimum `tests/test_surrogate_d.py` verifies architecture matches paper Appendix B
