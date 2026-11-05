# PROGRESS.md — Append-Only Session Log

Format per entry:

```
## Session N — YYYY-MM-DD

**Duration:** X hours
**Phase:** N. Phase Name

### Done
- [what was completed, with commit hashes]

### Decisions
- [decisions made this session, with rationale]

### Blockers
- [anything blocked, with proposed resolution]

### Next action
- [concrete next step for the next session]

### Notes
- [observations, gotchas, things future-you should remember]
```

---

## Session 1 — 2026-08-13

**Duration:** N/A (planning conversation)
**Phase:** 0 — Paper reading and specification extraction

### Done
- Read Arias Chao et al. 2022 (target paper) end-to-end
- Read Arias Chao et al. 2021 (dataset paper) end-to-end
- Extracted paper-verified facts (see PROJECT_PLAN.md §2 and REPLICATION_NOTES.md)
- Identified S1+S2 strategy for surrogate D training data
- Identified V1 resolution for x_s split discrepancy between target paper and dataset paper
- Locked all decisions in PROJECT_PLAN.md §1
- Scaffolded workspace: PROJECT_PLAN.md, CLAUDE.md, PROGRESS.md, REPLICATION_NOTES.md, REFERENCES.md, skills/, src/, notebooks/, configs/, tests/, data/, results/

### Decisions
- **Repo name:** `n-cmapss-ds02-fusion`
- **DL framework:** PyTorch 2.x
- **Storage:** Parquet (via pyarrow)
- **SQL demonstration:** DuckDB local
- **Surrogate D training:** S1 (DS02 healthy segments) as primary path + S2 (email M. Arias Chao) as parallel low-cost attempt
- **x_s split (V1):** follow target paper Table 2 (16 sensors including T40, P30 promoted from x_v)
- **Phase 5 tolerance (D8):** CNN hybrid RMSE within 4.14 ± 3σ (~3.9 to ~4.4)
- **Phase 7 (GenAI):** deferred to v2
- **No thermodynamic solver.** DNN surrogate D only.
- **No LEAP-1A reuse.** Separate portfolio project.

### Blockers
- None

### Next action
- Session 2: `git init`, initial commit of scaffolding, then start Phase 1
- Download DS02 from https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip
- Extract to `data/DS02/`
- Implement `src/data_loader.py` skeleton
- Run memory profiling to close D9

### Notes
- GitLab repo (arimanue/Fusing-physics-based-and-deep-learning-models-for-prognostics) inaccessible to both user and Claude. Treated as unavailable. Not blocking.
- S2 email to send: address manuel.arias@ethz.ch, request the CMAPSS-generated training dataset for the surrogate D used in the DS02 case study. Do not block on response.
- The x_s split discrepancy is a real inconsistency between the two papers. See REPLICATION_NOTES.md for full analysis. Choice of V1 (target paper) means x_s has 16 sensors, not 14.

---

## Session 2 — 2026-08-13 (dashboard + v2 scoping)

**Duration:** N/A (planning conversation continuation)
**Phase:** Still Phase 0 planning. No code yet.

### Done
- Discussed dashboard requirements: guided narrative dashboard (5 tabs), not raw-data explorer
- Chose Gradio over Streamlit and Power BI (Gradio native to HF Spaces, ML-idiomatic)
- Chose incremental phase-by-phase dashboard build (tabs added as underlying phases complete)
- Verified current AWS free tier for post-July-2025 accounts:
  - $200 credits (12-month expiry)
  - Always-free: Lambda (1M req/mo), DynamoDB (25 GB), S3 (5 GB), CloudWatch, SNS, Aurora PostgreSQL Serverless (new March 2026)
  - 100 GB/month free data transfer out (up from 1 GB in old model)
- Scoped v2 as MLOps track on same DS02 model, ~3 focused weeks after v1 done
- Confirmed v2 includes: W&B, Docker, GitHub Actions, AWS Lambda + API Gateway, DynamoDB, S3, CloudWatch + Evidently AI, A/B testing framework
- Chose W&B free tier over local MLflow (better recruiter recognition, zero setup)
- Included A/B testing framework as V2-Phase 5 (valued across data science / ML roles broadly, not just Amazon)
- Kept NYC TLC separate pipeline project as parked "decide later" item
- Confirmed dataset for v2 = same DS02 model from v1 (no dataset switch)
- Explicitly deferred GenAI (Phase 7) indefinitely (not in v1, not in v2, may become v3)

### Decisions
- **DASH framework:** Gradio, HF Spaces free tier
- **DASH-SEQ:** incremental build, tabs 1-2 after Phase 1, tab 3 after Phase 3, tab 4 after Phase 5, tab 5 during Phase 6
- **V2-EXP:** Weights & Biases free tier for experiment tracking
- **V2-AB:** A/B testing framework included as V2-Phase 5
- **V2-DATA:** same DS02 model from v1, no dataset switch
- **AWS scope:** always-free tier only (no credit-draining services)
- **IaC:** AWS SAM (chose over Serverless Framework, decision noted for revisit if issues surface in V2-Phase 4)
- **v2 sequencing rule:** do not start any v2 work during v1 phases (hard constraint added to CLAUDE.md)

### Blockers
- None

### Next action
- Session 3: `git init`, initial commit of all planning docs
- Send S2 email to Arias Chao (do not block on response)
- Begin Phase 1: download DS02, implement `src/data_loader.py`

### Notes
- v2 A/B testing framework: two implementation paths documented (single-Lambda in-function split vs multi-Lambda with router). Choice deferred to V2-Phase 5.
- v2 total cost projection: $0/month within always-free tier at portfolio scale (~100 requests/day). Explicit cost accounting to be produced in V2-Phase 7 documentation.
- Recruiter signal analysis: for aerospace propulsion roles, MLOps is nice-to-have but not core; for data analyst / applied scientist roles, MLOps is high-value. v2 addresses the second track more than the first, but coherent narrative ("aerospace ML + can ship it") benefits both.
- Files modified this session: PROJECT_PLAN.md (major additions §14, §15, §16, §17), CLAUDE.md (structure + hard constraints + locked decisions + out-of-scope), REFERENCES.md (v2 tool references), REPLICATION_NOTES.md (unchanged), new doc: docs/v2_mlops_plan.md

---

## Session 3 — 2026-08-13 to 2026-11-05 (Phase 1 data ingestion)

**Duration:** several working sessions, logged together
**Phase:** 1 — Data ingestion and exploratory analysis

### Done

**Environment verified**
- conda env `gnn_surrogate`: Python **3.12.11** (not 3.11), torch 2.8.0+cu128, CUDA available, RTX 2060 6 GB
- numpy 2.5.1, pandas 2.3.3, h5py 3.16.0, pyarrow 23.0.1, pytest 9.0.3. `duckdb` still missing
- **D9 resolved:** local training viable. DS02 is 2.28 GB in memory; models are 95k / 24k parameters. Colab as overflow

**Data located and identified**
- DS02 was already downloaded, together with the whole N-CMAPSS collection (~27 GB, DS01–DS08), flat in `data/` rather than `data/DS02/`
- DS02 = `N-CMAPSS_DS02-006.h5`, 2.28 GB, uncompressed contiguous float64, 6.5M rows
- PROJECT_PLAN's chunked-loading contingency is unnecessary — the file fits in RAM with room to spare

**Sampling rate discrepancy found and resolved** (the session's main finding)
- The release is **1 Hz**; the target paper worked at **0.1 Hz**
- Four independent proofs, the cleanest being unit 14: the paper gives its flight time as 0.16M *seconds* (§4.1) and its size as 0.016M *samples* (Table 1); our file has 156,778 rows — matching the seconds. One row is one second
- Resolution: decimate every 10th sample **within each (unit, cycle) group**
- Measured that decimation and block averaging are not interchangeable (26–44% of signal σ on fast channels), so the choice is substantive, not cosmetic

**Dataset version question settled**
- Target paper §4.1 points to ref [23] for "a more recent version" of DS02; ref [23] (p. 17) is the dataset paper. The paper used an older release
- Dataset paper Table 2 gives DS02 as 6.5M — matching our file. Ours is the newer release
- Degradation trajectories are identical (all 9 `t_EOL` and `t_s` exact), so the simulation is the same. What the version difference actually consists of is undetermined

**Variable split verified against the file**
- x_s = 16 (T40, P30 promoted), x_v = 11, θ = 3
- `phi` is the only released column the paper never uses — it is a derived ratio (Wf/Ps30), which is why target paper Table 3 lists 11 and not 12
- The paper's stated 20 baseline and 50 hybrid input dimensions both close exactly

**Built**
- `src/data_loader.py` — chunked sequential HDF5 reader, per-cycle decimation, Parquet output, `N_CMAPSS_DATA` override
- `tests/test_data_loader.py` — **38 tests, all passing**, asserting paper Table 1
- `data/processed/ds02_dev.parquet` (526,558 rows) and `ds02_test.parquet` (125,479 rows)
- `.gitignore` extended to exclude third-party PDFs and the NASA example notebook

**Acceptance gate — target paper Table 1**

| Unit | t_s | t_EOL | m_i paper | m_i ours | Δ |
|---|---|---|---|---|---|
| 2 | 17 ✓ | 75 ✓ | 85,000 | 85,351 | +351 |
| 5 | 17 ✓ | 89 ✓ | 103,000 | 103,384 | +384 |
| 10 | 17 ✓ | 82 ✓ | 95,000 | 95,313 | +313 |
| 16 | 16 ✓ | 63 ✓ | 77,000 | 76,560 | −440 |
| 18 | 17 ✓ | 71 ✓ | 89,000 | 89,103 | +103 |
| 20 | 17 ✓ | 66 ✓ | 77,000 | 76,847 | −153 |
| 11 | 19 ✓ | 59 ✓ | 66,000 | 66,381 | +381 |
| 14 | 36 ✓ | 76 ✓ | 16,000 | 15,714 | −286 |
| 15 | 24 ✓ | 67 ✓ | 43,000 | 43,384 | +384 |

All `t_s` and `t_EOL` exact. All `m_i` inside the ±500 implied by the paper's rounding.

### Decisions

- **Downsampling:** decimation (every 10th sample, offset 0, per cycle), not block averaging. Reason: the two releases come from one simulation run, so the 0.1 Hz series is a subsample of it. Averaging would give us cleaner data than the paper had
- **Python 3.12** accepted; the CLAUDE.md constraint amended from 3.11. Not a replication deviation — the paper specifies no Python version
- **Data path:** loader points at `data/`, not `data/DS02/`. PROJECT_PLAN §4 amended
- **Parquet keeps every released column** (all 14 x_s, all 14 x_v including `phi`, all 10 θ). The paper's selection happens at model-input time through named constants, so storage stays a faithful decimated copy of DS02
- **`t_s` convention:** first cycle with `hs == 0`. Deriving it as the last healthy cycle gives values one lower across all nine units
- **Two-repo split:** this repo stays replication + dashboard. A new `turbofan-ehm-platform` takes the MLOps track plus a new agent/MCP/GenAI track. This satisfies the existing "dedicated agentic project" constraint rather than reversing it. See PROJECT_PLAN §18
- **Repo 2 starts after Phase 3**, when three Agent A tools have real data behind them
- **Sampling-rate ablation** (1 Hz vs 0.1 Hz, FNN only) parked as a designed v3 experiment in §16

### Blockers

- None technical. Documentation debt was the blocker and is cleared by this entry

### Next action

1. Partition `data/processed/` Parquet by unit — before anything depends on the layout
2. `notebooks/01_data_eda.ipynb` — dataset paper Table 1, Fig. 3 envelopes, Fig. 6 degradation traces. Closes Phase 1
3. `git init` + first commit, push to GitHub
4. `pip install duckdb`; delete stray `debug.log`

### Notes

- **Windows pagefile:** 20 MB initial against 32 GB RAM, never corrected despite several attempts. Caused repeated system-wide stalls and a `WinError 1455` when loading CUDA libraries. The user is migrating to Linux, which makes this moot — but if any Windows work resumes, fix it before Phase 4's 30 training runs
- **OneDrive:** the 27 GB of HDF5 sits inside the OneDrive tree as on-demand files. Plausibly contributed to several I/O stalls. Only `N-CMAPSS_DS02-006.h5` is needed for v1; `N_CMAPSS_DATA` exists to relocate it
- **Flight durations** are roughly half the documented flight-class ranges (Fc3 units show ~3.2 h against a stated 5–7 h). Dataset paper Figure 3's own single-flight trace for unit 5 runs to ~11,000 s, matching our data, so this is a property of the dataset, not an error. Recorded window covers climb/cruise/descent above 10,000 ft
- **Migration bundle** created at `E:\OneDrive\Github_Projects\claude-migration.tar.gz` — three session transcripts plus settings, with instructions for restoring them under Linux
- Files modified: REPLICATION_NOTES.md (Deviation 2 amended, 4 and 5 added), CLAUDE.md (state, Python 3.12, data path, D9, Repo 2 pointers), PROJECT_PLAN.md (§1, §4, §12, §16, new §18), PROGRESS.md (this entry). New: src/data_loader.py, tests/test_data_loader.py, docs/x_s_split_rationale.md, docs/colab_guide.md
