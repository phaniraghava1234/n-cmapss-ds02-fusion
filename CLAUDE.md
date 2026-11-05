# CLAUDE.md — Session Context

**Project:** n-cmapss-ds02-fusion
**One-liner:** Pure replication of Arias Chao et al. (2022) hybrid physics + DL prognostics framework on N-CMAPSS DS02.

## Read this every session

1. This file (current state)
2. `PROGRESS.md` (session history, most recent entries)
3. `PROJECT_PLAN.md` (§ for the current phase)
4. `REPLICATION_NOTES.md` (deviation ledger — respect existing deviations, don't silently add new ones)
5. `skills/replication_discipline/SKILL.md` (rules of engagement)

## Current state

*Updated 2026-11-05.*

**Phase:** 1 in progress — data ingestion complete and verified; EDA not started
**Last completed:** `src/data_loader.py` + `tests/test_data_loader.py` (38 passing). Loaded data reproduces target paper Table 1: all 9 units match on `t_s` and `t_EOL` exactly, `m_i` within ±440.
**Last commit:** [repo not yet initialised — `git init` is the next infrastructure step]

**Key Phase 1 findings:**
- The released DS02 is **1 Hz**; the paper used **0.1 Hz**. We decimate every 10th sample within each flight cycle. See REPLICATION_NOTES Deviation 4.
- The paper used an **older DS02 release** (§4.1, ref [23]). Same simulation, finer sampling in ours. Deviation 5.
- x_s = 16, x_v = 11 (`phi` dropped), θ = 3. Verified against the file. Deviation 2.

**Next actions, in order:**
1. Partition `data/processed/` Parquet by unit (`unit=N/part.parquet`) — needed before anything depends on the layout
2. `notebooks/01_data_eda.ipynb` — closes Phase 1
3. `git init` + first commit
4. `pip install duckdb` (needed from Phase 5); delete stray `debug.log`

## Data location

DS02 lives at `data/N-CMAPSS_DS02-006.h5` (2.28 GB), flat in `data/` — **not** `data/DS02/`.
The full N-CMAPSS collection (~27 GB, DS01–DS08) sits alongside it; v1 touches only DS02.

`src/data_loader.py` reads the `N_CMAPSS_DATA` environment variable first and falls back
to `<repo>/data`. Set it to relocate the data without code changes (Colab, Linux, a
non-OneDrive disk).

## Environment

Verified 2026-11-05: conda env `gnn_surrogate`, **Python 3.12.11**, torch 2.8.0+cu128,
CUDA available, NVIDIA RTX 2060 6 GB. numpy 2.5.1, pandas 2.3.3, h5py 3.16.0,
pyarrow 23.0.1, pytest 9.0.3. `duckdb` not yet installed.

**D9 resolved:** local training is viable — DS02 is 2.28 GB in memory, and the paper's
models are 95k (FNN) and 24k (CNN/LSTM) parameters. Colab is the overflow path; code
stays CPU/GPU/Colab agnostic.

## Project structure: v1 + v1.5 + v2

- **v1 (Phases 0-6):** paper replication on DS02. Currently active track.
- **v1.5 (Phases 1d, 3d, 5d, 6d):** Gradio dashboard on HF Spaces, built alongside v1 phases.
- **v2 (Phases V2-1 through V2-7):** MLOps track on the same DS02 model. **Do not start until v1 complete.**

Detailed v2 plan: `docs/v2_mlops_plan.md`.

## Hard constraints (do not violate)

- **Python 3.12** (amended 2026-11-05 from 3.11 — the `gnn_surrogate` env is 3.12.11 and
  nothing in the stack requires 3.11; the paper specifies no Python version, so this is
  not a replication deviation), PyTorch 2.x, snake_case naming
  - Sensor names keep the paper's own symbols verbatim (`T24`, `Ps30`, `alt`). snake_case
    applies to everything else.
- **v1 = replication only.** No new methods, no architectures the paper didn't use.
- **No thermodynamic solver.** We build a DNN surrogate D, not a cycle solver.
- **No LEAP-1A code reuse.** That project stays separate.
- **No PostgreSQL.** DuckDB only (local, file-based, reads Parquet).
- **Parquet for all intermediate storage.**
- **Every deviation from the paper goes into `REPLICATION_NOTES.md` with fidelity impact assessment.**
- **No v2 work during v1.** MLOps track starts only after Phase 6 complete. If you catch me starting v2 tooling (AWS, W&B, Docker, GitHub Actions workflows) during v1 phases, stop me — that's scope creep.
- **v1.5 dashboard tabs built only when their underlying data exists.** Tab 4 (Model predictions) cannot start before Phase 5. Do not build placeholder tabs.

## Rules of engagement (project-specific replication discipline)

Full detail in `skills/replication_discipline/SKILL.md`. Summary:

1. Never assume. Ask before inventing missing information.
2. State uncertainty explicitly. Separate facts from assumptions.
3. Don't hallucinate. If you don't know, say so. Never fabricate references, APIs, URLs, or file names.
4. Ask before making architectural / library / algorithm / naming / hyperparameter decisions.
5. Explain reasoning. Show why, mention alternatives.
6. Preserve context. Never silently change earlier decisions. Flag conflicts.
7. Be precise. No "maybe" or "should work" — explain exactly why.
8. Cite sources. Distinguish paper facts from personal experience.
9. No hidden changes. Name what changed and why.
10. Think before answering. Verify → identify missing info → ask if needed → answer.
11. Correctness over speed.
12. Challenge me when appropriate. Point out logical errors, unsafe assumptions.
13. Don't over-engineer. Simplest solution that meets requirements.
14. Label explicit assumptions clearly. Never treat as confirmed.
15. Previously locked decisions are project requirements unless I explicitly change them.

**Response workflow:** Restate request → list missing info → ask clarifying questions if needed → explain reasoning → give answer → list assumptions → state confidence (High/Medium/Low with why).

## Out-of-scope list

### Out of scope for v1 (may come in v2 or later)
- Cloud deployment (AWS Lambda, DynamoDB, S3) — comes in v2
- MLOps tooling (W&B, Docker, GitHub Actions CI/CD) — comes in v2
- A/B testing framework — comes in v2
- Model monitoring, drift detection — comes in v2

### Out of scope for this repo — moved to `turbofan-ehm-platform` (decided 2026-11-05)
- **GenAI insight layer** (originally Phase 7). Now Repo 2 Track B.
- **Agents A and B + custom MCP server.** Now Repo 2 Track B.
- **MLOps track** (the former v2 plan — W&B, Docker, GitHub Actions, AWS, monitoring, A/B).
  Now Repo 2 Track A. `docs/v2_mlops_plan.md` stays here as the reference until Repo 2 exists.

This is **not** a reversal of the constraints below — it is the dedicated project they
called for. See §18 of `PROJECT_PLAN.md`. Repo 2 reads this repo's published Parquet
through a documented schema; nothing flows back into `src/`.

### Out of scope for v1 and v2 (parked indefinitely, may become v3)
- Physics-informed extensions (Xiong 2023, Liao 2023)
- Additional datasets (FD001–FD004, other N-CMAPSS subsets)
- Separate NYC TLC full-pipeline project — decision deferred
- Sampling-rate ablation: train the FNN on the full 1 Hz data and compare against the
  0.1 Hz replication baseline. Designed experiment, see `PROJECT_PLAN.md` §16. Requires
  a D8-passing baseline first.

### Out of scope forever (for THIS repo)
- MCP / agentic tool-calling — belongs in a dedicated agentic project. That project is
  now named: `turbofan-ehm-platform`. The constraint stands; the work has a home.
- Streamlit dashboard — replaced by Gradio (locked)
- PostgreSQL — replaced by DuckDB for v1 (locked); Aurora Serverless considered for v2 if a use case emerges
- Thermodynamic cycle solver — paper uses DNN surrogate D instead
- LEAP-1A code reuse — separate portfolio project
- Kubernetes, SageMaker, Airflow — over-engineered for this project's scale

## Locked decisions (do not re-litigate)

See `PROJECT_PLAN.md` §1 for the full table. Highlights:

**v1:**
- Repo: `n-cmapss-ds02-fusion`
- DL framework: PyTorch
- Storage: Parquet
- SQL: DuckDB (local)
- Surrogate D training: S1 (DS02 healthy segments) + S2 (email author in parallel)
- Phase 5 tolerance: CNN hybrid RMSE 4.14 ± 3σ (~3.9 to ~4.4)

**v1.5:**
- Dashboard framework: Gradio (not Streamlit, not Power BI)
- Hosting: Hugging Face Spaces free tier
- Sequencing: incremental, tab per phase

**v2:**
- Experiment tracking: Weights & Biases free tier
- Containerization: Docker + Docker Compose
- CI/CD: GitHub Actions
- Cloud: AWS always-free tier only (Lambda + API Gateway + DynamoDB + S3 + CloudWatch + SNS)
- IaC: AWS SAM
- Drift detection: Evidently AI (open source)
- Dataset: same DS02 model from v1 (do not switch datasets)
- A/B testing framework: included as V2-Phase 5

## Pointers

- Full plan: `PROJECT_PLAN.md`
- Session log: `PROGRESS.md`
- Deviation ledger: `REPLICATION_NOTES.md`
- Rules of engagement (detailed): `skills/replication_discipline/SKILL.md`
- Phase 2 methodology (once written): `skills/surrogate_d/SKILL.md`
- Phase 3 methodology (once written): `skills/theta_estimation/SKILL.md`
- References: `REFERENCES.md`

## Notes for Claude (behavioral)

- Do not generate placeholder code unless asked.
- Do not add features, refactor, or introduce abstractions beyond what the task requires.
- Do not add error handling for scenarios that cannot happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external files).
- Do not simplify algorithms without telling me.
- Do not preserve backward compatibility unless instructed.
- When requirements are incomplete, produce a checklist of missing information instead of guessing.
- If you detect a better approach than what I requested, explain it and wait for approval before changing direction.
- When I describe a problem or think out loud, deliver an assessment. Don't apply a fix until I ask.
