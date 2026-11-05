# REPLICATION_NOTES.md — Deviation Ledger

Every departure from the target paper (Arias Chao et al. 2022, RESS 217) is documented here.
Format per entry:

```
## Deviation N: <short title>

**Paper does:** <what the paper does>
**We do:** <what we do>
**Reason:** <why we deviate>
**Fidelity impact:** <quantitative or qualitative assessment>
**Phase introduced:** <which phase this deviation surfaces in>
**Status:** open | closed | monitoring
```

---

## Deviation 1: Surrogate D trained on DS02 healthy segments, not on independent CMAPSS-generated data

**Paper does:** Trains the surrogate D on "an independent dataset generated with the CMAPSS simulator" (§4.2). Development of this training dataset is stated as "beyond the scope of this case study." The dataset is not released with the paper.

**We do:** Extract healthy segments of DS02 (rows where t < t_s, the transition time per Table 1) from the 6 training units. Use these as training data for D, with θ labels approximated from the dataset paper's degradation model (§3.1: δ_n(t) = a_n · t + δ_0, a_n ≈ -0.001). Split by unit for D training vs D validation.

**Reason:**
1. The paper's independent CMAPSS training dataset is not publicly released.
2. CMAPSS itself is NASA-restricted MATLAB/Simulink — obtaining and running it to regenerate the training set would require weeks of paperwork plus MATLAB licensing, disproportionate to a portfolio replication.
3. DS02 healthy segments are the closest publicly available proxy — they are CMAPSS outputs at healthy conditions, which is exactly what D needs to learn.
4. The paper's Ablation Study III (§5.2.3, Table 9) shows the framework is robust to calibration noise (SNR 15 dB) and bias (±50%), suggesting some tolerance to lower-fidelity surrogate models.

**Fidelity impact:**
- **Expected direction:** D will be trained on a narrower θ range than the paper's D (which saw full-envelope CMAPSS runs with varied θ). This may reduce UKF accuracy in the abnormal-degradation regime, precisely where θ deviates most from nominal.
- **Bounded by:** paper's own ablation showing framework robustness to calibration noise.
- **Quantifiable in:** Phase 5, by comparing our CNN hybrid RMSE against paper's 4.14 ± 0.09.
- **Mitigation attempted:** S2 email sent to M. Arias Chao (manuel.arias@ethz.ch) requesting the original training data. If received before Phase 2 completes, evaluate switching.

**Phase introduced:** Phase 2

**Status:** monitoring (until Phase 5 quantifies the fidelity gap)

---

## Deviation 2: x_s / x_v split follows target paper Tables 2–3, not dataset paper Tables 5–6

*Amended 2026-11-05: `phi` exclusion added after direct verification against the release. The original entry said x_v was "accordingly reduced" to 12, which was wrong — the paper's count is 11.*

**Paper does:** Target paper Table 2 (p. 8) lists 20 condition monitoring signals `[w, x_s]`; removing the four scenario descriptors `w` leaves **x_s = 16**, including T40 and P30. Target paper Table 3 (p. 8) lists **x_v = 11** virtual sensors.

Dataset paper Tables 5 and 6 (p. 7) describe the released file differently: **x_s = 14** (no T40, no P30) and **x_v = 14** (T40, P30, the paper's 11, plus `phi`).

**We do:** Follow the target paper. Three concrete consequences:

1. **T40 and P30 are promoted** out of the release's `X_v_var` into x_s, giving 16.
2. **`phi` is dropped entirely.** Dataset paper Table 6 row 14 defines it as the ratio of fuel flow to Ps30 — derived from two values already in the input set, not an independent sensor. The target paper lists it in neither table.
3. **θ = 3** (HPT_eff_mod, LPT_eff_mod, LPT_flow_mod) per target paper Table 4, out of the 10 modifiers the release carries. DS02's two failure modes move only these three.

Verified against the file on 2026-08-19: every signal the paper requires is present, and `phi` is the only released column the paper uses nowhere.

**Reason:** We replicate the target paper, not the dataset paper. Its split is the input specification for the prognostics network G. The arithmetic confirms the reading — `4 + 16 = 20` baseline inputs and `4 + 16 + 16 + 11 + 3 = 50` hybrid inputs, both stated explicitly in §4.3 (p. 7). No other split reproduces those two numbers.

**Fidelity impact:** Zero for the replication target — this *is* the paper's specification. The inconsistency lies between the two papers, not between us and the target. Terminology differs from the dataset paper's schema; documented in `docs/x_s_split_rationale.md`.

**Phase introduced:** Phase 1

**Status:** closed (documented and verified against the data)

---

## Deviation 3: LSTM max-epochs not specified in target paper for main N-CMAPSS case study

**Paper does:** Specifies FNN max epochs = 60, CNN max epochs = 30 for the main case study (§4.5). Does not explicitly state LSTM max epochs.

**We do:** Default to LSTM max epochs = 30, matching CNN. Document choice.

**Reason:** Paper specifies patience = 5 for early stopping regardless of architecture, so the exact max cap matters less than the early-stopping criterion. Defaulting to CNN's value is a defensible convention.

**Fidelity impact:** Likely negligible. Early stopping typically triggers before the max cap on the paper's architectures.

**Phase introduced:** Phase 4

**Status:** monitoring (revisit if LSTM training behavior looks atypical)

---

## Deviation 4: Sampling rate reduced from the released 1 Hz to the paper's 0.1 Hz by decimation

**Paper does:** States the sampling rate is 0.1 Hz (§4.1, p. 6), giving 0.53M development and 0.12M test samples. It never states how that series was produced.

**We do:** Keep every 10th sample **within each (unit, cycle) group**, at offset 0. Implemented as `decimation_index()` in `src/data_loader.py`.

**Evidence the release is 1 Hz** (all verified directly, 2026-08-19):

| Evidence | Source | Result |
|---|---|---|
| Stated sampling rate | Target paper §4.1 | 0.1 Hz |
| Unit 14 flight time | Target paper §4.1 | 0.16M **seconds** |
| Unit 14 sample count | Target paper Table 1 | 0.016M **samples** → 10 s/sample |
| Unit 14 rows in release | measured | 156,778 → 1 row = 1 second |
| Window N_tw = 50 spans | Target paper §4.3 | "500 s" → 10 s/sample |
| Total release size | Dataset paper Table 2 | 6.5M, matches our file |
| Row-count ratio | measured, all 9 units | mean 9.995 (min 9.799, max 10.081) |

**Reason:**
1. `N_tw = 50` spans 500 seconds at the paper's sampling and 50 seconds at the release's. Keeping 1 Hz would change the model's receptive field tenfold, or force a 10× larger input layer — either way it stops being the paper's architecture.
2. RMSE (Eq. 8) and the per-cycle averaged estimate (Eq. 9) are both averaged over sample counts. A 10× difference makes our numbers incomparable to Table 5, which decision D8 depends on.

**Why decimation and not block averaging:** the paper is silent, so this is an inference. Unit ids, cycle counts, transition cycles, flight classes and failure modes all match the paper exactly (see Deviation 5), so the two releases describe the same simulation run. The lower-rate series is therefore a subsample of that run — values the simulator actually produced at 10-second marks. Block averaging would yield smoothed values that never occurred in the simulation, handing our models cleaner inputs than the paper had. That could improve results while destroying comparability, which is the worst possible outcome for a replication.

**Measured consequence:** the two methods are **not** interchangeable on this data. On unit 2 cycle 1, `max|decimated − averaged|` reaches 26–44% of signal standard deviation for fast-responding channels (Nf 43.7%, T50 41.5%, T48 40.8%, Nc 39.7%, T30 38.9%, Wf 31.4%, TRA 30.5%), while altitude, P2 and T2 stay under 3%. Real signal content exists above the 0.05 Hz Nyquist limit, so **decimation aliases**. If the paper's dataset was produced the same way, its models saw identically aliased data and matching it is correct.

*Caveat on that measurement: it is a maximum, so it is driven by the single worst instant in the cycle and overstates typical divergence. It establishes that the methods can differ sharply, not how often they do.*

**Residual unknown:** the original release's decimation phase. We take offset 0 (seconds 0, 10, 20, …); it may have been offset 5. Unknowable without the old file.

**Fidelity impact:** Reproduces the paper's per-unit sample counts to within ±440, inside the ±500 implied by the paper's own rounding to three decimals in millions. Split totals land within 0.4% of Table 1's sums. A different phase or method in the old release would change individual sample values but not distributions.

**Phase introduced:** Phase 1

**Status:** monitoring (until Phase 5 shows whether RMSE lands in the D8 window)

---

## Deviation 5: We replicate against a newer DS02 release than the paper used

**Paper does:** Target paper §4.1 (p. 6) refers the reader to reference [23] for "a more recent version of this dataset (i.e., DS02)". Reference [23] (p. 17) is the dataset paper, Arias Chao et al. (2021), *Data* 6(1):5. The paper therefore used an **earlier release** than the one the dataset paper documents.

**We do:** Use `N-CMAPSS_DS02-006.h5`, 6.5M samples — which dataset paper Table 2 (p. 6) confirms is the release it describes, i.e. the newer one. The paper's own data totals 0.65M.

**Reason:** The older release is not publicly available. The NASA PCoE repository serves the current version only, and we have no documentation of what the `-006` suffix denotes.

**What is demonstrably unchanged** (measured, all 9 units, 2026-08-19): unit ids; cycle counts `t_EOL` = 75, 89, 82, 63, 71, 66, 59, 76, 67 (exact); transition cycles `t_s` = 17, 17, 17, 16, 17, 17, 19, 36, 24 (exact); flight classes; failure modes. The degradation model (dataset paper §3.1) is stochastic in both initial wear and fault onset, so two independent simulation runs could not produce identical end-of-life cycles across nine engines. The underlying simulation is the same.

**What we cannot determine:** *what* the version difference consists of. Two possibilities survive and we cannot distinguish them:

- **(i)** the older release was already 0.1 Hz and the newer one added finer sampling;
- **(ii)** both releases are 1 Hz, the paper decimated as preprocessing, and the version difference is something unobservable to us.

Under both, the required action is identical and the mechanism is decimation rather than averaging, so Deviation 4 holds either way.

**Fidelity impact:** Unquantifiable in principle, since the older file is unavailable. Bounded in practice by exact agreement on every structural quantity the paper reports. Any remaining difference lies in sample values, not in fleet composition or failure timing.

**Phase introduced:** Phase 1

**Status:** open (cannot be closed without the original release; to be stated plainly in the README limitations section)

---

<!--
Add new deviations below as they are discovered. Do not silently modify code without adding an entry here.
-->
