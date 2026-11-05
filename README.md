# n-cmapss-ds02-fusion

**Status:** In development (Phase 1 pending)

A pure replication of the hybrid physics + deep learning prognostics framework proposed by Arias Chao et al. (2022) on NASA's N-CMAPSS DS02 turbofan engine run-to-failure dataset.

## What this replicates

Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2022). "Fusing physics-based and deep learning models for prognostics." *Reliability Engineering & System Safety*, 217, 107961.

The framework combines:
1. A DNN surrogate (**D**) that approximates the CMAPSS thermodynamic engine model
2. An Unscented Kalman Filter that uses D to estimate unobservable engine health parameters θ̂ (efficiency and flow modifiers for HPT, LPT)
3. A deep learning prognostics network (**G**) that consumes the enriched feature space [w, x_s, x̂_s, x̂_v, θ̂] to predict Remaining Useful Life

Three G architectures are evaluated: feed-forward (FNN), 1D convolutional (CNN), and LSTM — matching the paper.

## Dataset

N-CMAPSS DS02 from NASA Prognostics Data Repository.

Citation: Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2021). "Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics." *Data*, 6(1), 5.

Download: https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip

## Background

This project sits within the gas path analysis and engine health management (EHM) tradition, comprehensively reviewed by Volponi (2014, ASME JEGTP). The fusion of a calibrated physics-based performance model with a deep learning residual correction is one instance of the hybrid architectures Volponi identifies as the field's direction of travel.

## What is NOT in this project

Explicitly out of scope for v1:
- Alternative fusion architectures (e.g., Xiong et al. 2023, Liao et al. 2023)
- Additional datasets (original C-MAPSS FD001–FD004, other N-CMAPSS subsets)
- Cloud deployment (AWS Lambda, DynamoDB, S3)
- LLM-based explanation layer (deferred to v2)
- A standalone thermodynamic cycle solver

## Repository structure

See `PROJECT_PLAN.md` §12 for the full layout.

## Documentation

- `PROJECT_PLAN.md` — full plan, phases, deliverables, references
- `REPLICATION_NOTES.md` — every deviation from the target paper, with fidelity impact
- `REFERENCES.md` — full bibliography
- `CLAUDE.md` — session context for Claude Code
- `PROGRESS.md` — session log
- `skills/*/SKILL.md` — methodology anchors

## Reproducibility

Once Phase 6 completes, this section will contain:
- Environment setup instructions (`environment.yml`)
- DS02 download and extraction steps
- Run order for full pipeline reproduction

## Results

_To be filled in on completion of Phase 5._

Replication target: CNN hybrid RMSE = 4.14 ± 0.09 cycles on paper's test units (11, 14, 15). Success threshold: within 4.14 ± 3σ.

## License

_TBD._

## Contact

Phani — Toulouse, France
