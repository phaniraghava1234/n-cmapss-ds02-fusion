# N-CMAPSS DS02 Fusion Replication — Project Plan

## 0. Metadata

**Repo name:** `n-cmapss-ds02-fusion`
**Working title:** Replication of Arias Chao et al. (2022) on N-CMAPSS DS02
**Author:** Phani (Toulouse, France)
**Version:** 1.0 (initial plan, locked)
**Status:** Phase 0 complete; Phase 1 ready to start

**Target paper (replication anchor):**
Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2022).
"Fusing physics-based and deep learning models for prognostics."
*Reliability Engineering & System Safety*, 217, 107961.
DOI: 10.1016/j.ress.2021.107961

**Target dataset:**
NASA PCoE Dataset #17, Turbofan Engine Degradation Simulation Data Set 2 (N-CMAPSS DS02).
Citation: Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2021).
"Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics."
*Data*, 6(1), 5. DOI: 10.3390/data6010005
Download: https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip

**Scope statement:**
This project has three tracks layered on top of each other:

- **v1 (Phases 0-6):** Pure replication of Arias Chao et al. (2022) on N-CMAPSS DS02. Reproduce the paper's reported RUL prediction performance within a documented tolerance.
- **v1.5 (Phases 1d, 3d, 5d, 6d):** Gradio dashboard on Hugging Face Spaces, built incrementally alongside v1. Tabs added as underlying phases produce content. Shareable URL for recruiters.
- **v2 (Phases V2-1 through V2-6):** MLOps track on the same DS02 model. Full pipeline from training to serverless deployment to monitoring. Reuses the trained model from v1; adds W&B experiment tracking, Docker containerization, GitHub Actions CI/CD, AWS Lambda + API Gateway inference, DynamoDB prediction storage, Evidently AI drift monitoring, and A/B testing framework.

v1 must complete before v2 starts. v1.5 is built alongside v1 phase-by-phase.

**Non-goals (explicitly out of scope for v1):**
- Alternative fusion architectures (Xiong 2023, Liao 2023, others)
- Additional datasets (original C-MAPSS FD001–FD004, PHM08, other N-CMAPSS subsets DS01/DS03–DS08)
- MCP / agentic tool-calling layer
- Any physics-informed extension beyond what the paper does
- Building a standalone thermodynamic cycle solver
- Reusing LEAP-1A code — that project stays separate
- Any v2 work (MLOps track) — do not start until v1 complete

**Phase 7 (GenAI explanation layer) is deferred indefinitely — not in v1, not in v2.** May become v3 if user revisits later.

---

## 1. Locked decisions

The following are project requirements. Do not change without explicit user override.

| # | Item | Value |
|---|---|---|
| — | Target paper | Arias Chao et al. 2022, RESS 217 |
| — | Dataset | N-CMAPSS DS02 |
| — | Replication discipline | Strict — if paper does X, we do X |
| D1 | Repo name | `n-cmapss-ds02-fusion` |
| D2 | Folder naming | snake_case |
| D3 | DL framework | PyTorch 2.x |
| D5 | θ storage format | Parquet |
| D5b | SQL demonstration | DuckDB (local, file-based, reads Parquet directly) |
| D6/D7 | Phase 7 (GenAI) | Deferred indefinitely (not in v2 either); may become v3 |
| DASH | Dashboard framework | Gradio, deployed on Hugging Face Spaces free tier |
| DASH-SEQ | Dashboard build sequencing | Incremental, tab per phase (see §14) |
| V2-EXP | Experiment tracking for v2 | Weights & Biases free tier |
| V2-AB | A/B testing framework in v2 | Yes, included as V2-Phase 5 |
| V2-DATA | Dataset for v2 | Same DS02 model from v1 (do not switch datasets) |
| D8 | Phase 5 tolerance | CNN hybrid RMSE within paper's 4.14 ± 3σ (≈ 3.9 to 4.4 cycles) |
| D9 | Compute location | **Resolved 2026-11-05:** local (RTX 2060 6 GB) viable; Colab as overflow. Code stays CPU/GPU/Colab agnostic |
| D10 | GitLab code (arimanue) | Treated as unavailable; not blocking |
| V1 | x_s split | 16-sensor per target paper Table 2 (T40 and P30 promoted from x_v to x_s); x_v = 11 with `phi` dropped per Table 3 |
| Q2 | Surrogate D training data | S1 primary (train D on DS02 healthy segments) + S2 parallel (email M. Arias Chao) |
| D3b | Python version | **3.12** (amended 2026-11-05 from 3.11; env `gnn_surrogate` is 3.12.11, paper specifies none) |
| D11 | Sampling rate | Decimate the released 1 Hz to the paper's 0.1 Hz — every 10th sample within each (unit, cycle), offset 0. REPLICATION_NOTES Deviation 4 |
| D12 | Data path | `data/N-CMAPSS_DS02-006.h5`, flat in `data/`. `N_CMAPSS_DATA` env var overrides |
| D13 | Parquet layout | Partitioned by unit: `data/processed/ds02/unit=N/part.parquet`. All released columns retained; the paper's selection applies at model-input time |
| D14 | Repo split | This repo = replication + dashboard. `turbofan-ehm-platform` = MLOps + agents/MCP/GenAI. See §18 |

---

## 2. Architecture we replicate

Three distinct model objects (paper §3, §4.2):

- **F** — real CMAPSS thermodynamic simulator. NASA-restricted MATLAB/Simulink. We do NOT touch this.
- **D** — DNN surrogate approximating F. MLP, 4 hidden layers × 100 units, ReLU, linear output (paper §4.2, Appendix B). We build this.
- **G** — RUL prognostics network. Three architectures: FNN, 1D CNN, LSTM. Consumes enriched features [w, x_s, x̂_s, x̂_v, θ̂]. We build all three.

**Data flow:**
```
DS02 HDF5 → data_loader → (w, x_s, x_v, θ_ref, RUL)
                                    ↓
                          [Phase 2: train D on healthy segments]
                                    ↓
                   D: (w, x_s^(t-1), θ) → (x̂_s, x̂_v)
                                    ↓
                          [Phase 3: UKF loop using D]
                                    ↓
                   θ̂(t) trajectories per unit + calibrated (x̂_s, x̂_v)
                                    ↓
                          [Phase 4: train G on [w, x_s, x̂_s, x̂_v, θ̂]]
                                    ↓
                   RUL predictions on test units 11, 14, 15
                                    ↓
                          [Phase 5: RMSE, s-score, prediction horizon]
                                    ↓
                          [Compare against paper Table 5]
```

---

## 3. Phase 0 — Paper reading and specification extraction

**Status:** ✅ Complete.

**Objective:** convert both papers into an unambiguous specification before writing code.

**Deliverables (completed):**
- Both papers read end-to-end
- Extracted facts: UKF Q = R = diag(0.01²), MLP surrogate arch, three DL arches with hyperparameters, sliding window N_tw = 50, unit splits, headline metrics
- Identified S1+S2 strategy for surrogate D training data
- Identified V1 as the x_s split resolution
- All facts documented in this plan and `REPLICATION_NOTES.md`

**References:**
- Arias Chao et al. 2022 (target paper)
- Arias Chao et al. 2021 (dataset paper)

---

## 4. Phase 1 — Data ingestion and exploratory analysis

**Objective:** load DS02 into memory-efficient Python structures matching paper notation. Verify correctness by reproducing dataset paper statistics. Complete memory check (D9).

**Activities:**
1. ~~Download DS02~~ ✅ **Done.** Present as `data/N-CMAPSS_DS02-006.h5` (2.28 GB), flat in `data/`, alongside the full N-CMAPSS collection (~27 GB). v1 touches only DS02.
2. ✅ **Done.** `src/data_loader.py`:
   - `load_split(split)` / `build_processed()` / `load_processed(split)`, returning paper notation
   - x_s per V1: 16 signals (T40, P30 promoted); x_v = 11 (`phi` dropped); θ = 3. Cited in the module docstring
   - Split by unit ID: dev = {2, 5, 10, 16, 18, 20}, test = {11, 14, 15} — taken from the file's own `_dev`/`_test` partition
   - **Decimation 1 Hz → 0.1 Hz** per D11, within each (unit, cycle) group
3. ✅ **Memory check closed.** Full in-memory footprint is **2.28 GB**, not the ~24 GB the original plan guarded against. Chunked loading via Polars or h5py chunking is **not required**. The loader still reads in 500k-row sequential chunks, for I/O behaviour and progress reporting rather than memory necessity. D9 resolved to local.
4. Partition the Parquet output by unit (D13) before anything depends on the layout.
5. EDA notebook `notebooks/01_data_eda.ipynb`:
   - Reproduce dataset paper Table 1 (unit counts, transition times t_s, EOL cycles)
   - Reproduce Figure 3 (kernel density plots of flight envelopes)
   - Reproduce Figure 6 (degradation traces)
   - Any mismatch is a blocker — investigate before proceeding
6. ✅ **Done.** x_s split documented in `docs/x_s_split_rationale.md`

**Deliverables:**
- ✅ `src/data_loader.py`
- ✅ `tests/test_data_loader.py` — 38 tests asserting target paper Table 1
- ⬜ `notebooks/01_data_eda.ipynb`
- ⬜ `notebooks/02_data_prep_experiments.ipynb`
- ✅ `docs/x_s_split_rationale.md`
- ✅ Memory footprint report in `PROGRESS.md` (Session 3)
- ✅ D9 resolved — local, Colab as overflow

**Success criteria:**
- ✅ **Met 2026-08-19.** All 9 units reproduce target paper Table 1: `t_s` and `t_EOL` exact, `m_i` within ±440 (inside the ±500 implied by the paper's rounding). Split totals within 0.4% of Table 1 sums
- ✅ Unit ID splits match exactly
- ✅ Memory footprint documented (2.28 GB) and local path chosen
- ⬜ EDA reproduction of dataset paper Figures 3 and 6

**References:**
- Arias Chao et al. 2021 (dataset paper) §2.3, Tables 3–7

**Estimated effort:** 3–5 focused days

---

## 5. Phase 2 — Surrogate model D

**Objective:** build and train the DNN surrogate D that approximates CMAPSS dynamics, using the S1 strategy (train on DS02 healthy segments).

**Activities:**
1. Implement `src/surrogate_d.py`:
   - MLP: input → 100 → 100 → 100 → n_out, ReLU throughout, linear output layer
   - Input dimension: len(w) + len(x_s) + len(θ_active) = 4 + 16 + 3 = 23 (verify from paper's exact formulation)
   - Output dimension: len(x_s) + len(x_v) = 16 + 11 = 27 (verify)
2. Prepare S1 training data:
   - Extract healthy segments per unit: rows where t < t_s (transition time; per Table 1 of target paper)
   - For each row: (w^(t), x_s^(t-1), θ_approx^(t)) → (x_s^(t), x_v^(t))
   - θ_approx from dataset paper §3.1: δ_n(t) = a_n · t + δ_0 with a_n ≈ -0.001, δ_0 per-unit initial degradation (uniform in [0, 0.01])
   - Split by unit: 5 training units for D fitting, 1 held out for D validation
3. Train D:
   - MSE loss on (x_s, x_v) output
   - Adam optimizer, LR 1e-3, batch size 512, epochs until validation loss plateaus, early stopping patience 10
   - Log training curves
4. Validation gates:
   - D validation MSE on held-out unit's healthy segment
   - Qualitative check: D predictions vs actual sensor traces on a single flight cycle
   - Reasonableness check: D should NOT produce wildly different outputs for small (w, θ) perturbations — sanity check on numerical stability
5. Write `skills/surrogate_d/SKILL.md` capturing methodology and gotchas discovered during implementation
6. Send S2 email to Arias Chao in parallel; if response arrives with training data before Phase 2 completes, evaluate whether to switch or continue with S1

**Deliverables:**
- `src/surrogate_d.py`
- `configs/surrogate_d.yaml`
- Trained D checkpoint in `results/checkpoints/surrogate_d.pt`
- `notebooks/02_surrogate_d_validation.ipynb`
- `skills/surrogate_d/SKILL.md` filled in
- Fidelity note entry in `REPLICATION_NOTES.md`

**Success criteria:**
- D trains without divergence
- Validation MSE stable and reported in Parquet log
- Manual inspection of predictions vs measurements on one flight cycle shows no obvious pathologies

**References:**
- Arias Chao et al. 2022 §4.2, Appendix B
- Arias Chao et al. 2021 §3.1 (degradation model for θ_approx labels)

**Estimated effort:** 4–6 days

---

## 6. Phase 3 — Health parameter estimation via UKF

**Objective:** implement the Unscented Kalman Filter (paper Algorithm 2, Appendix C) using D as the forward model, and estimate θ̂(t) trajectories for every unit.

**Activities:**
1. Implement `src/theta_estimator.py`:
   - Follow Algorithm 2 verbatim
   - Q = q² · I_n, R = r² · I_d with q = r = 0.01 per Appendix C
   - Sigma point computation, weight coefficients per Table C.14 (γ = √(d+λ), λ = α²(d+κ)−d, α = 1e-3, β = 2, κ = 0)
   - State vector: θ (dimension = number of active health parameters, 3 for DS02)
   - Measurement equation via trained D from Phase 2
2. Run UKF per unit across all flight cycles:
   - Initial θ̂ = θ_ref (nominal)
   - Initial covariance P⁰ = Q
   - Log θ̂(t), P(t), residual r(t), calibrated [x̂_s, x̂_v] at each timestep
   - Store per-unit trajectories as Parquet in `results/theta_trajectories/`
3. Sanity checks:
   - θ̂ trajectories trend toward paper's Figure 6 patterns (declining efficiency/flow with degradation)
   - θ̂ for HPT_eff, LPT_eff, LPT_flow shows clear transition around each unit's t_s
   - Fleet-level plot comparable to paper Figure 11 (which is for FD002 but shape should match)
4. Write `skills/theta_estimation/SKILL.md`

**Deliverables:**
- `src/theta_estimator.py`
- Per-unit θ trajectories in `results/theta_trajectories/unit_*.parquet`
- `notebooks/03_theta_trajectories.ipynb`
- `skills/theta_estimation/SKILL.md` filled in

**Success criteria:**
- UKF runs without numerical instability (no NaN, no covariance blowup)
- θ̂ trajectories show expected degradation patterns qualitatively
- If paper publishes θ̂ figures for specific units, ours match qualitatively

**References:**
- Arias Chao et al. 2022 §3.1, Algorithm 2, Appendix C, Table C.14

**Estimated effort:** 5–7 days

---

## 7. Phase 4 — Deep learning RUL prediction

**Objective:** train the three prognostics networks (FNN, 1D CNN, LSTM) on the enriched feature vector [w, x_s, x̂_s, x̂_v, θ̂], matching paper §4.4 architectures and §4.5 training setup exactly.

**Activities:**
1. Implement `src/models.py`:
   - **FNN:** [n, 200, 200, 200, 50, 1], ReLU, linear output. 95k params.
   - **1D CNN:** three 1D conv layers (filter size 10; channels 10, 10, 1), zero padding, flatten, 50-way FC, linear output. ReLU throughout. 24k params.
   - **LSTM:** three LSTM layers (20 cells each), 50-way FC, linear output. ReLU. 24k params.
2. Implement `src/train.py`:
   - Optimizer: mini-batch SGD with AMSgrad (paper §4.5)
   - Batch size: 1024
   - Learning rate: 0.001
   - Xavier weight init
   - Max epochs: FNN 60, CNN 30, LSTM (paper doesn't specify for main study — use 30 as default, document in REPLICATION_NOTES)
   - Early stopping patience 5
   - Validation set: 10% of D (development data)
   - Sliding window N_tw = 50, stride 1 (for CNN and LSTM only)
   - Input normalization: min/max to [-1, 1] using training data statistics
3. Implement two feature configurations:
   - **Baseline:** x = [w, x_s], n = 20 (target paper Table 2 dimension check)
   - **Hybrid:** x = [w, x_s, x̂_s, x̂_v, θ̂], n = 50
4. Train each of 3 architectures × 2 configurations = 6 models × 5 seeds = 30 runs total. Log everything.
5. Reproducibility metadata per run: seed, git commit hash, config hash, wall time

**Deliverables:**
- `src/models.py`
- `src/train.py`
- `configs/train_fnn.yaml`, `configs/train_cnn.yaml`, `configs/train_lstm.yaml`
- 30 trained checkpoints in `results/checkpoints/`
- Training logs per run in `results/logs/`
- `notebooks/04_training_curves.ipynb`

**Success criteria:**
- All 30 runs complete without divergence
- Training curves stable, comparable to paper's implied behavior
- Baseline runs reproduce paper Table 5 data-driven numbers within ±σ

**References:**
- Arias Chao et al. 2022 §4.3, §4.4, §4.5

**Estimated effort:** 5–7 days (compute time dominated)

---

## 8. Phase 5 — Evaluation and paper comparison

**Objective:** evaluate all 30 trained models on the paper's test split, compute all metrics the paper reports, and compare against the paper's Table 5, 6, and Figure 8.

**Activities:**
1. Implement `src/evaluate.py`:
   - RMSE per unit and fleet aggregate (paper Eq. 8)
   - NASA scoring function s (paper Eq. 7)
   - Prediction horizon H_|Δy|≤5 = t_EOL − t_|Δy|≤5 (paper Eq. 11)
   - Per-cycle averaged RUL estimate ŷ_u^[c] (paper Eq. 9)
   - Inference time per sample
2. Ablation reproductions:
   - **Ablation I** (paper §5.2.1, Table 7): CNN trained on subset (only units 16, 18, 20). Compare 4.95 → 4.14 RMSE swing.
   - **Ablation II** (paper §5.2.2, Table 8): feature ablation [w,x_s] → [w,x_s,x̂_s] → [w,x_s,x̂_s,x̂_v] → full hybrid
   - **Ablation III** (paper §5.2.3, Table 9): calibration noise/bias sensitivity
3. Generate comparison table: our results vs paper Table 5 (mean ± σ over 5 seeds, side-by-side)
4. Reproduce paper Figure 8 (RUL error curves per test unit with ±5-cycle bands)
5. DuckDB analytics notebook `notebooks/05a_sql_analytics.ipynb`:
   - Load engineered features and results from Parquet via DuckDB
   - Demonstrate SQL: window functions over flight cycles, CTE-based degradation-mode ranking, unit-level RUL distribution aggregation, anomaly detection query
6. Write final entry in `REPLICATION_NOTES.md` for every observed deviation and its quantitative fidelity impact

**Deliverables:**
- `src/evaluate.py`
- `notebooks/05_evaluation.ipynb`
- `notebooks/05a_sql_analytics.ipynb` (DuckDB)
- `results/comparison_table.parquet` and Markdown-rendered version in `docs/replication_comparison.md`
- Reproduced Figure 8 in `results/figures/`
- Complete `REPLICATION_NOTES.md`

**Success criteria (replication success threshold D8):**
- CNN hybrid RMSE within 4.14 ± 3σ (~3.9 to ~4.4 cycles) over 5 runs
- Prediction horizon fleet average within paper's reported 37 cycles ± reasonable margin
- Ablation trends qualitatively match paper (dataset-size reduction impact, feature ablation ordering, calibration robustness)

**References:**
- Arias Chao et al. 2022 §4.6, §5.1, §5.2, Tables 5–9, Figure 8

**Estimated effort:** 4–6 days

---

## 9. Phase 6 — Documentation and closeout

**Objective:** produce public-facing artifacts of the replication.

**Activities:**
1. Finalize `README.md`:
   - What this project replicates and doesn't
   - Headline results table (ours vs paper)
   - Honest limitations section (S1 fidelity gap, x_s split note, any other observed deviations)
   - How to reproduce (environment setup, DS02 download, run order)
2. Finalize `REPLICATION_NOTES.md` with all deviations catalogued
3. Finalize `REFERENCES.md`
4. Add `environment.yml` for conda reproducibility
5. Add background section to README citing Volponi 2014 as field context (gas path analysis / EHM tradition)

**Deliverables:**
- Polished `README.md`
- Final `REPLICATION_NOTES.md`
- Final `REFERENCES.md`
- `environment.yml`
- Optional: short LinkedIn/portfolio post writeup

**References cited in README background section:**
- Volponi, A. J. (2014). "Gas Turbine Engine Health Management: Past, Present, and Future Trends." *ASME Journal of Engineering for Gas Turbines and Power*, 136(5), 051201. (Field context only, not used as method reference.)

**Estimated effort:** 2–3 days

---

## 10. Phase 7 — GenAI diagnostic explanation layer (DEFERRED indefinitely)

Not part of v1. Not part of v2. Documented in earlier planning conversation. May become v3 if user revisits. Do not start.

---

## 11. Reference list (v1)

**Required (methods, replication core):**
1. Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2022). "Fusing physics-based and deep learning models for prognostics." *Reliability Engineering & System Safety*, 217, 107961.
2. Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2021). "Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics." *Data*, 6(1), 5.
3. Frederick, D. K., DeCastro, J. A., & Litt, J. S. (2007). "User's Guide for the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)." NASA/TM-2007-215026. (Referenced by target paper; may not be needed directly since we do not touch CMAPSS.)

**Background (README only, field context):**
4. Volponi, A. J. (2014). "Gas Turbine Engine Health Management: Past, Present, and Future Trends." *ASME J. Eng. Gas Turbines Power*, 136(5), 051201.

**Parked for future work (NOT v1):**
5. Xiong, J., Fink, O., Zhou, J., & Ma, Y. (2023). "Controlled physics-informed data generation for deep learning-based remaining useful life prediction under unseen operation conditions." *Mechanical Systems and Signal Processing*, 197, 110359.
6. Liao, X., Chen, S., Wen, P., & Zhao, S. (2023). "Remaining useful life with self-attention assisted physics-informed neural network." *Advanced Engineering Informatics*, 58, 102195.

---

## 12. Repository structure

```
n-cmapss-ds02-fusion/
├── CLAUDE.md                        # session handoff
├── PROGRESS.md                      # append-only session log
├── PROJECT_PLAN.md                  # this file
├── README.md                        # public-facing
├── REPLICATION_NOTES.md             # deviation ledger
├── REFERENCES.md                    # bib list
├── environment.yml                  # conda env spec (Phase 6)
├── .gitignore
├── docs/
│   ├── x_s_split_rationale.md       # Phase 1
│   └── replication_comparison.md    # Phase 5
├── skills/
│   ├── replication_discipline/SKILL.md
│   ├── surrogate_d/SKILL.md         # filled Phase 2
│   └── theta_estimation/SKILL.md    # filled Phase 3
├── data/                            # gitignored
│   ├── N-CMAPSS_DS02-006.h5         # flat, not data/DS02/ (D12)
│   └── processed/ds02/unit=N/       # partitioned Parquet (D13)
├── src/
│   ├── data_loader.py               # Phase 1  ✅
│   ├── data_prep.py                 # Phase 1
│   ├── utils.py                     # device-agnostic helpers (CPU / GPU / Colab)
│   ├── surrogate_d.py               # Phase 2
│   ├── theta_estimator.py           # Phase 3
│   ├── models.py                    # Phase 4
│   ├── train.py                     # Phase 4
│   └── evaluate.py                  # Phase 5
├── notebooks/                       # pattern: explore → script → inspect
│   ├── 01_data_eda.ipynb            # P1  full EDA
│   ├── 02_data_prep_experiments.ipynb  # P1  experiments only
│   ├── 03_surrogate_d.ipynb         # P2  inspect D predictions
│   ├── 04_theta_trajectories.ipynb  # P3  inspect θ̂
│   ├── 05_training_curves.ipynb     # P4  inspect training
│   ├── 06_predictions.ipynb         # P5  predictions vs actual
│   └── 07_sql_analytics.ipynb       # P5  DuckDB
├── configs/
│   ├── surrogate_d.yaml
│   ├── train_fnn.yaml
│   ├── train_cnn.yaml
│   └── train_lstm.yaml
├── tests/
│   ├── test_data_loader.py
│   ├── test_surrogate_d.py
│   └── test_metrics.py
└── results/
    ├── checkpoints/
    ├── logs/
    ├── theta_trajectories/
    └── figures/
```

---

## 13. Timeline

### v1 (replication core)

| Phase | Nominal duration | Cumulative |
|---|---|---|
| 0. Paper reading | ✅ Done | 0 days |
| 1. Data ingestion & EDA | 3–5 days | 3–5 days |
| 2. Surrogate D | 4–6 days | 7–11 days |
| 3. θ estimation (UKF) | 5–7 days | 12–18 days |
| 4. DL training | 5–7 days | 17–25 days |
| 5. Evaluation | 4–6 days | 21–31 days |
| 6. Documentation | 2–3 days | 23–34 days |
| **v1 subtotal** | **3–5 focused weeks** | |

### v1.5 (dashboard, incremental)

Dashboard sub-deliverables built alongside v1 phases (see §14). No standalone duration.

Additional ~1 day per relevant phase (Phases 1, 3, 5, 6) + 2 days final polish and deploy = ~6 days spread across v1.

### v2 (MLOps track, after v1 complete)

| Phase | Nominal duration | Cumulative from v1 end |
|---|---|---|
| V2-1. W&B tracking retrofit | 2 days | 2 days |
| V2-2. Docker containerization | 2 days | 4 days |
| V2-3. GitHub Actions CI/CD | 3 days | 7 days |
| V2-4. AWS Lambda + API Gateway inference | 4 days | 11 days |
| V2-5. A/B testing framework | 3 days | 14 days |
| V2-6. Monitoring + drift detection | 3 days | 17 days |
| V2-7. Documentation | 2 days | 19 days |
| **v2 subtotal** | **~3 focused weeks** | |

### Overall

- **v1 + v1.5:** ~4-6 focused weeks
- **v1 + v1.5 + v2:** ~7-9 focused weeks

Calendar time depends on focused-hours-per-week budget alongside job hunt and other projects.

---

## 14. v1.5 Dashboard — incremental build schedule

Framework: Gradio, deployed on Hugging Face Spaces free tier.
Repo path: `dashboard/` (added at Phase 1d).
URL pattern: `huggingface.co/spaces/<username>/n-cmapss-ds02-fusion`

### Phase 1d — Tabs 1 (Problem) + 2 (Data drill-down)

**When:** after Phase 1 (Data ingestion & EDA) completes.
**Tab 1 (Problem):** brief prose intro, fleet overview plot (paper Figure 6 style), stated data volume.
**Tab 2 (Data drill-down):** unit selector, cycle selector, sensor multi-select, interactive plot, summary stats panel.
**Data source:** pre-computed aggregates + on-demand loads of specific cycles from a subset of DS02 shipped with the app.
**Effort:** ~1 day.
**Deliverables:** `dashboard/app.py` (initial), `dashboard/data/precomputed.parquet`, first deploy to HF Spaces (URL exists).

### Phase 3d — Tab 3 (Physics / θ trajectories)

**When:** after Phase 3 (θ estimation) completes.
**Tab 3 (Physics):** engine schematic, explanation of θ (HPT_eff, LPT_eff, LPT_flow), per-unit θ̂ trajectory plot with t_s marked.
**Data source:** θ̂ trajectories Parquet files from Phase 3.
**Effort:** ~1 day.
**Deliverables:** updated `dashboard/app.py`, θ̂ trajectories bundled in dashboard data.

### Phase 5d — Tab 4 (Model predictions)

**When:** after Phase 5 (Evaluation) completes.
**Tab 4 (Model):** architecture summary, per-test-unit predicted vs true RUL curves with ±5-cycle bands (paper Figure 8 style), metrics table with paper comparison, baseline-vs-hybrid toggle.
**Data source:** predictions and metrics Parquet from Phase 5.
**Effort:** ~1.5 days.
**Deliverables:** updated `dashboard/app.py`, prediction Parquets bundled.

### Phase 6d — Tab 5 (About) + polish + final deploy

**When:** during Phase 6 (Documentation).
**Tab 5 (About):** links to GitHub, paper, bio, contact, honest limitations section.
**Polish:** consistent theming, mobile responsiveness check, loading states, error handling for edge cases.
**Effort:** ~2 days.
**Deliverables:** finalized dashboard, README badge with HF Spaces URL, screenshot in main repo README.

### Dashboard tech stack

- Gradio 4.x (Python-native, HF Spaces first-class support)
- pandas, numpy, matplotlib for data manipulation and plots
- HF Spaces free tier (16 GB storage, 2 vCPU, 16 GB RAM, no sleep)
- Model artifacts loaded from repo (no external hosting needed)

### Constraints for the dashboard

- **Pre-computed only.** No live inference on user-selected DS02 slices beyond what the app can compute in <2 seconds.
- **Bundle size <500 MB.** Keep only what's needed for the guided narrative, not the full 14 GB DS02.
- **No sensitive data.** Everything shipped is public (dataset is public, model is public).
- **Independent of v1 replication code.** Dashboard imports precomputed Parquets, does not depend on `src/` runtime — this decouples the deployed artifact from Python environment drift.

---

## 15. v2 MLOps Track — plan

**Prerequisite:** v1 must be complete (Phases 0-6 done, CNN hybrid model trained and evaluated, meets D8 tolerance).

**Objective:** demonstrate an end-to-end production ML pipeline on the same DS02 model. Signal to recruiters: "aerospace ML + can ship it." Reuses v1's trained model — does not train a new one.

**Detailed phase-by-phase plan:** `docs/v2_mlops_plan.md` (companion document to this file).

**Summary of v2 phases:**

### V2-Phase 1: W&B experiment tracking retrofit
- Retroactively log v1 training runs to Weights & Biases free tier
- Register the trained CNN hybrid model in W&B Model Registry
- Deliverable: public W&B project with all v1 runs visible, model versioned with commit hash and metrics
- References: W&B docs (https://docs.wandb.ai)
- Effort: ~2 days

### V2-Phase 2: Docker containerization
- `Dockerfile` for inference container (PyTorch + trained model + surrogate D + minimal Python deps)
- `Dockerfile` for training container (full v1 stack, reproducible training)
- `docker-compose.yml` for local end-to-end run
- Deliverable: reproducible builds, `docker-compose up` runs a working prediction API locally
- Effort: ~2 days

### V2-Phase 3: GitHub Actions CI/CD
- CI: on push, run pytest, lint, verify model loads
- CD: on tag, build Docker image, push to public registry (GHCR)
- Optional: nightly retraining job that runs on a subsampled DS02, verifies metrics still meet threshold
- Deliverable: green CI badge on README, automated model-artifact publishing
- Effort: ~3 days

### V2-Phase 4: AWS Lambda + API Gateway inference
- Deploy CNN hybrid model as Lambda function (container image, since model + PyTorch >250 MB)
- API Gateway REST endpoint for RUL predictions
- Input: flight cycle sensor data. Output: RUL prediction with uncertainty band.
- DynamoDB table for prediction logs (unit_id, timestamp, prediction, latency)
- S3 bucket for model artifacts
- Cost: within always-free tier for portfolio-scale usage (1M Lambda invocations/month, 25 GB DynamoDB, 5 GB S3)
- References: AWS Lambda docs, AWS API Gateway docs
- Effort: ~4 days

### V2-Phase 5: A/B testing framework
- Deploy two model versions to Lambda (e.g., CNN hybrid vs FNN hybrid, or two random seeds of CNN hybrid)
- Route requests via API Gateway with weighted routing (or Lambda function URL with in-function routing)
- Log per-model predictions and latencies to DynamoDB
- Analysis notebook comparing model behavior on production traffic (or simulated traffic)
- Recruiter signal: "can design and reason about production experiments"
- Effort: ~3 days

### V2-Phase 6: Monitoring + drift detection
- CloudWatch dashboards: invocation count, error rate, p95 latency
- Evidently AI (open-source) for data drift detection: compare inbound feature distributions vs training distribution
- Alerting: SNS notification if error rate or drift score exceeds threshold
- Cost: within always-free CloudWatch (10 custom metrics, 5 GB log ingestion) and SNS (1M requests)
- References: CloudWatch docs, Evidently AI docs (https://docs.evidentlyai.com)
- Effort: ~3 days

### V2-Phase 7: Documentation
- Architecture diagram (data → training → registry → containerize → deploy → monitor)
- Runbook: how to deploy, how to rollback, how to interpret monitoring
- README v2 section with cost estimate and always-free-tier accounting
- Effort: ~2 days

### v2 total effort: ~3 focused weeks

### v2 tech stack additions

- Weights & Biases (free tier)
- Docker + Docker Compose
- GitHub Actions
- AWS: Lambda, API Gateway, DynamoDB, S3, CloudWatch, SNS
- Evidently AI (Python package, open source)
- AWS SAM or Serverless Framework for infrastructure-as-code (choice deferred to V2-Phase 4)

### v2 constraints

- **Stay within always-free tier.** Every AWS component chosen because it fits under always-free limits at portfolio scale. Any component that would consume the $200 credit pool needs explicit justification.
- **Same DS02 model.** Do not retrain, do not switch datasets. v2 signal is "can operationalize the v1 model," not "can build another model."
- **No Kubernetes, no SageMaker, no Airflow.** Simpler stack; portfolio project not production system.
- **Cost transparency.** Documentation includes explicit accounting of what would cost money at what scale.

---

## 16. Parked for later (potential v3+)

Not committed. Do not start without explicit decision.

- **v3 candidate: GenAI diagnostic explanation layer** (originally deferred from v1). LLM consumes model output, produces engineer-facing explanation. Would use Anthropic Claude or Groq Llama free tier. Scope: single function `explain(model_output_dict) -> str`, JSON-schema-bound, no chat, no agentic behavior.
- **v3 candidate: Separate small "full pipeline" project on NYC TLC dataset.** Discussed but not committed. Decision deferred. If v2 on DS02 delivers the full-pipeline signal, this becomes redundant.
- **v3 candidate: Physics-informed extensions.** Xiong 2023 (data augmentation) or Liao 2023 (PINN with attention). Would extend beyond replication into method contribution.
- **v3 candidate: Additional N-CMAPSS subsets** (DS01, DS03-DS08). Would broaden the replication to more failure modes.

### v3 candidate: sampling-rate ablation (designed 2026-11-05)

**Question:** does training on the full 1 Hz data (5.26M development samples) improve RUL
prediction over the 0.1 Hz replication baseline (0.53M)?

**Design:**
- **FNN only.** The paper applies its sliding window to the CNN and LSTM alone (§4.3, p. 8),
  so for those two a sampling-rate change forces a second change — either the window spans
  50 s instead of 500 s, or `N_tw` goes to 500 and the input layer grows tenfold. Either way
  the comparison confounds sampling rate with architecture. The FNN takes single timesteps,
  so its input dimension and parameter count are identical under both rates. It is the only
  clean arm.
- **Evaluate both arms on the 0.1 Hz test set**, so RMSE is computed over the same number of
  test samples and stays comparable to each other and to paper Table 5.
- `src/data_loader.py` already supports this: `load_split(..., decimate=False)` returns the
  full 1 Hz data, and the Parquet retains every released column.

**Prerequisite:** a D8-passing baseline. Without a validated 0.1 Hz arm, neither number is
interpretable — a difference could equally be a real effect or an implementation bug.

**Prior expectation:** small or no gain. Consecutive 1-second readings are highly
autocorrelated, so 10× the rows is far less than 10× the information, and the label space is
unchanged (unit 2 has 75 cycles and 75 RUL values at either rate). Worth measuring rather
than asserting; nothing published addresses it for DS02.

---

## 17. Success criteria

### v1 success

1. CNN hybrid RMSE within D8 threshold (3.9 to 4.4 cycles) on paper's test split
2. `REPLICATION_NOTES.md` documents every deviation with honest fidelity assessment
3. Full pipeline runs end-to-end reproducibly from a clean checkout with only DS02 as external input
4. README makes replication scope and limitations clear enough that a careful reader can trust the claims
5. DuckDB SQL notebook demonstrates the analytical query bullet

### v1.5 success

1. HF Spaces URL is live and shareable to recruiters
2. All 5 tabs render correctly with data from Phases 1, 3, 5
3. Dashboard loads in <15 seconds cold start on HF Spaces free tier
4. No live inference beyond what completes in <2 seconds per interaction
5. Bundle size <500 MB

### v2 success

1. All v1 experiments visible in public W&B project
2. Docker Compose runs the full local pipeline reproducibly
3. GitHub Actions CI passes on every commit, CD publishes model artifacts on every tag
4. AWS Lambda inference API responds in <2 seconds p95
5. A/B testing framework routes traffic between two model versions and produces comparative metrics
6. CloudWatch dashboards + Evidently drift reports visible
7. Total ongoing AWS cost = $0 at portfolio scale (documented in v2 README)

---

## 18. Repository split — `turbofan-ehm-platform` (decided 2026-11-05)

### Why

The MLOps track (formerly v2, §15) and a new agentic/GenAI track both operate **on top of**
this repository's outputs rather than inside its pipeline. `CLAUDE.md` already required that
MCP and agentic tool-calling live in "a dedicated agentic project, not here." Giving that
project a name satisfies the constraint rather than reversing it.

The separation is one-way: Repo 2 reads Repo 1's published Parquet through a documented
schema. Nothing flows back into `src/`. This is the same relationship §14 already specifies
for the Gradio dashboard.

For portfolio purposes it also splits one sprawling repository into two legible claims:
*replicated a published prognostics framework* and *built an agentic analysis platform on
top of it*.

### Repo 1 — `n-cmapss-ds02-fusion` (this repo)

v1 replication (Phases 0–6) + v1.5 Gradio dashboard. Scope unchanged.

**Deliverable that Repo 2 consumes:** partitioned Parquet under a documented schema —
`data/processed/ds02/unit=N/part.parquet`, plus θ̂ trajectories from Phase 3, training logs
from Phase 4, and prediction/metric tables from Phase 5.

### Repo 2 — `turbofan-ehm-platform` (created when Repo 1 reaches Phase 3)

Two tracks. The agent track leads because it needs only a local checkpoint; the MLOps track
needs the final evaluated model.

#### Track B — agents, MCP, GenAI

Three layers, one source of truth per capability. Tool implementations are plain Python
functions; agents call them directly and the MCP server wraps them.

**Agent A — data layer (4 tools).** Scope: load, compare, detect.

| Tool | Depends on |
|---|---|
| `get_unit_data(unit_id, cycles=None, sensors=None)` | Repo 1 Phase 1 ✅ |
| `get_theta_trajectory(unit_id)` | Repo 1 Phase 3 |
| `compare_units(unit_ids, metric)` | metric vocabulary grows per phase |
| `detect_anomalies(unit_id, threshold=None)` | Repo 1 Phase 3 (UKF residuals) |

**Agent B — decision layer (3 tools + 1 internal).** Scope: interpret, diagnose, rank, plan.

| Tool | Depends on |
|---|---|
| `predict_rul(unit_id, backend="local"\|"remote")` | Repo 1 Phase 4 |
| `query_knowledge_base(question)` | RAG corpus, independent |
| `rank_fleet(criteria)` | Repo 1 Phase 5 |
| `ask_data_agent(query)` *(internal)* | delegates to Agent A |

Agent B holds no data-loading tools; it delegates. `predict_rul` defaults to a **local
checkpoint**, with the deployed endpoint as an optional backend — the agent layer must not
depend on AWS being up.

**MCP server — `turbofan-ehm-mcp` (7 tools).** Thin wrapper over the same Python functions,
exposing all seven capabilities to external clients (Claude Desktop, Cursor, VS Code).
`ask_data_agent` is **not** exposed: it is a composition primitive, not a capability.

**RAG corpus:** this project's own documentation (`CLAUDE.md`, `PROJECT_PLAN.md`,
`REPLICATION_NOTES.md`, the SKILL files) plus public-domain NASA technical reports. The
Elsevier PDF is excluded pending a licence check; its content is already distilled into our
notes. Project docs give an agent that answers "why did we decimate by 10?" from the ledger
— self-maintaining, no licensing question.

**Models:** Claude first. Then migrate Agent A to a local ~3B model (comfortable at 6 GB
VRAM, roughly 2 GB at 4-bit) while Agent B stays on Claude. The split is not cosmetic —
Agent A does structured retrieval over four well-named tools, which small models handle
reliably; Agent B does multi-step reasoning and delegation, which they do not.

#### Track A — MLOps

The seven phases of §15, unchanged: W&B retrofit → Docker → GitHub Actions → Lambda +
API Gateway → A/B testing → monitoring + Evidently → documentation. `predict_rul`'s remote
backend lands during the Lambda phase as a swap-in, not a rewrite.

### Sequencing

| Phase | Work | Needs | Est. |
|---|---|---|---|
| P2-0 | Scaffold, data contract, shared tool layer | R1 P3 | 2 d |
| P2-1 | Agent A, 4 tools, Claude-backed | R1 P3 | 3 d |
| P2-2 | MCP server, tested from an external client | P2-1 | 2 d |
| P2-3 | `predict_rul`, `rank_fleet`, Agent B + delegation; MCP to 7 tools | R1 P5 | 4 d |
| P2-4 | `query_knowledge_base` — RAG | P2-3 | 3 d |
| P2-5 | Local-model migration + benchmark | P2-4 | 3 d |
| P2-6 … P2-12 | MLOps track (§15) | R1 P5 | 19 d |

```
Repo 1:  P1 ─ P2 ─ P3 ─────── P4 ─ P5 ─ P6
                     └─────┐        │
Repo 2:                 P2-0..P2-2  └─ P2-3 ─ P2-4 ─ P2-5 ─ MLOps
```

Estimates for Repo 2 carry no benchmark behind them — treat them as ordering, not schedule.
