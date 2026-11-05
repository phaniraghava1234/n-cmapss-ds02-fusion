# SKILL: Health Parameter Estimation (UKF)

**Status:** STUB — to be filled in during Phase 3 implementation.

**When to load this skill:** any session touching Phase 3 (UKF-based θ estimation) or downstream phases that consume θ̂ trajectories.

---

## Scope of this skill

Methodology for the Unscented Kalman Filter that estimates health parameters θ̂(t) from measured sensor readings x_s using the surrogate D as the forward model.

Reference: Arias Chao et al. 2022 §3.1, Algorithm 2, Appendix C.

---

## What we know before Phase 3 starts

**State-space formulation (paper Eqs. 3, 4):**
```
θ^(t) = θ^(t-1) + ξ^(t),         ξ ~ N(0, Q)
x_s^(t) = F(w^(t), θ^(t)) + ε^(t), ε ~ N(0, R)
```

Where F is replaced by our trained D from Phase 2.

**UKF algorithm (paper Algorithm 2):**
Standard UKF update sequence:
1. Predict covariance: P̄ = P + Q
2. Sigma points: S = [θ̂, θ̂ + γ√P̄, θ̂ − γ√P̄]
3. Propagate: Y_i = F(w, S_i) for i = 0, ..., 2d
4. Predicted mean: x̂_s = Σ W_i^m · Y_i
5. Residual: r = x_s − x̂_s
6. Innovation covariance: P_y = Σ W_i^c · (Y_i − x̂_s)(Y_i − x̂_s)ᵀ + R
7. Cross-covariance: P_θy = Σ W_i^c · (S_i − θ̂)(Y_i − x̂_s)ᵀ
8. Kalman gain: K = P_θy · P_y⁻¹
9. Update: θ̂ ← θ̂ − K · r; P ← P̄ − K · P_y · Kᵀ

**Hyperparameters (paper Appendix C, Table C.14):**
- Q = q² · I_n with q = 0.01
- R = r² · I_d with r = 0.01
- α = 1e-3
- β = 2
- κ = 0
- λ = α²(d + κ) − d
- γ = √(d + λ)
- W₀^m = λ / (d + λ)
- W₀^c = W₀^m + 1 − α² + β
- W_i^c = W_i^m = 1 / (2(d + λ)) for i = 1, ..., 2d

Where d is the dimension of θ (3 for DS02: HPT_eff, LPT_eff, LPT_flow).

**Initial conditions:**
- θ̂(0) = θ_ref (nominal, zero degradation)
- P(0) = Q

---

## To be filled in during Phase 3

### Implementation choices

- Which linear algebra library for √P̄ (SciPy cholesky vs custom)
- How to handle non-positive-definite P (regularization scheme)
- Whether to normalize θ inputs before feeding D
- Whether to run D in eval mode with `torch.no_grad()` inside the UKF loop

### Numerical stability observations

- Any observed covariance blowup
- Any observed θ̂ jumps outside physical range
- Any needed clamping

### Performance characteristics

- Wall time per UKF step
- Total wall time per unit
- Bottleneck (D inference vs matrix ops)

### Sanity check outputs

- θ̂ trajectories per unit (fleet plot)
- Should show clear transition at t_s per unit
- Should trend toward paper Figure 6 patterns
- HPT-only units (2, 5, 10) should show HPT_eff degradation dominant
- HPT+LPT units should show all three θ moving

### Interface for Phase 4 consumers

- Output storage format (Parquet schema)
- Per-unit trajectory file naming
- What to include: θ̂, P (covariance), calibrated [x̂_s, x̂_v], residual r

---

## Rules while filling this in (per replication_discipline SKILL)

- Every algorithmic step maps to a line in paper Algorithm 2
- Every constant traces to paper Table C.14
- No numerical stability tricks silently added — document in REPLICATION_NOTES if needed
- Test coverage: `tests/test_theta_estimator.py` verifies at least the sigma point computation matches Table C.14
