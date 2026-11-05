# v2 MLOps Track — Detailed Plan

**Prerequisite:** v1 (Phases 0-6) must be complete. Do not start v2 until CNN hybrid model is trained, evaluated, and meets D8 tolerance.

**Companion document to:** `PROJECT_PLAN.md` §15.

**Objective:** operationalize the v1 trained model. Full ML lifecycle from experiment tracking to production serverless inference to monitoring. Zero ongoing cost via AWS always-free tier.

---

## V2-Phase 1: Weights & Biases experiment tracking retrofit

**Duration:** ~2 days

### Objective
Retroactively log v1 training runs to Weights & Biases free tier. Register the trained CNN hybrid model in W&B Model Registry. Result: public W&B project link that recruiters can browse.

### Activities
1. Create W&B account, initialize public project `n-cmapss-ds02-fusion`
2. Modify `src/train.py` to log to W&B (metrics per epoch, hyperparameters, model artifacts, run configs)
3. Re-run the 30 v1 experiments (3 architectures × 2 configurations × 5 seeds) with W&B logging enabled
4. Register the best-performing CNN hybrid model in W&B Model Registry with tags: `production`, `v1.0`, git commit hash
5. Verify public W&B project link works from an incognito browser

### Deliverables
- `src/wandb_config.py` with project name, entity, tags
- Updated `src/train.py` (backward-compatible; W&B logging can be disabled via env var)
- Public W&B project URL added to README
- Model artifact URI logged in `REPLICATION_NOTES.md`

### Success criteria
- All 30 runs visible in W&B, with metrics, curves, and configs
- Model registry shows a versioned model with lineage back to git commit
- Anyone with the URL can view experiments without login

### References
- W&B Python SDK: https://docs.wandb.ai/quickstart
- W&B Model Registry: https://docs.wandb.ai/guides/model_registry

---

## V2-Phase 2: Docker containerization

**Duration:** ~2 days

### Objective
Reproducible builds for both training and inference. `docker-compose up` runs a working local prediction API.

### Activities
1. `Dockerfile.train`: base image `python:3.11-slim`, install v1 dependencies, copy `src/`, entrypoint `python -m src.train`
2. `Dockerfile.inference`: base image `python:3.11-slim`, install inference-only deps (torch, numpy, pyarrow, fastapi, uvicorn), copy trained model artifacts, entrypoint `uvicorn src.api:app`
3. `docker-compose.yml`: services for `training` (optional), `inference` (default), volume mounts for data and results
4. Multi-stage build for inference container to keep image size <2 GB (Lambda limit)
5. Verify inference container responds to a curl request with a RUL prediction

### Deliverables
- `Dockerfile.train`
- `Dockerfile.inference`
- `docker-compose.yml`
- `.dockerignore`
- `src/api.py` — minimal FastAPI wrapper around the model
- Documentation in `docs/docker_setup.md`

### Success criteria
- Both containers build without errors
- Inference container image <2 GB (Lambda compatible)
- `docker-compose up inference` starts a working REST endpoint on localhost
- `curl -X POST localhost:8000/predict -d '{...}'` returns a RUL prediction

### Constraints
- No `RUN pip install ...` beyond what's actually needed
- Pin versions (no `torch` — use `torch==2.x.y`)
- Multi-stage build for inference to strip build-time deps

---

## V2-Phase 3: GitHub Actions CI/CD

**Duration:** ~3 days

### Objective
Automated testing on every commit, automated model artifact publishing on every tag.

### Activities
1. `.github/workflows/ci.yml`:
   - Trigger: pull_request and push to main
   - Steps: checkout, setup-python 3.11, install deps, run pytest, run ruff, verify model loads
2. `.github/workflows/cd.yml`:
   - Trigger: git tag matching `v*`
   - Steps: build Docker inference image, push to GitHub Container Registry (GHCR), upload model artifact to release assets
3. Optional: `.github/workflows/nightly.yml`:
   - Trigger: cron nightly
   - Steps: run training on subsampled DS02, verify metrics still within tolerance, alert on regression
4. README badges for CI status, model version, W&B project link

### Deliverables
- `.github/workflows/ci.yml`
- `.github/workflows/cd.yml`
- Optional: `.github/workflows/nightly.yml`
- README badges section

### Success criteria
- Green CI badge on repo
- Every tag produces a Docker image on GHCR and a GitHub release
- Nightly job (if enabled) does not consume >100 GitHub Actions minutes/month
- Public repo has under 2,000 minutes of Actions usage per month (free tier)

### Constraints
- Public repo (2,000 free minutes/month; private is 500)
- No Actions secrets that would cost money (no AWS credentials that trigger paid services unbounded)

---

## V2-Phase 4: AWS Lambda + API Gateway inference

**Duration:** ~4 days

### Objective
Public REST API serving RUL predictions. Zero ongoing cost within always-free limits.

### Activities
1. Create AWS account resources via infrastructure-as-code:
   - Choice: AWS SAM (simpler, aws-native) vs Serverless Framework (multi-cloud). Decision: SAM.
   - `template.yaml`: Lambda function, API Gateway REST API, DynamoDB table, S3 bucket, IAM roles
2. Package inference container from V2-Phase 2 for Lambda (Lambda supports container images up to 10 GB)
3. Push container to Amazon Elastic Container Registry (ECR); note this may consume a small amount of credits
4. Deploy Lambda function using the container image; configure memory (2 GB) and timeout (30 s)
5. API Gateway REST endpoint: POST `/predict` with schema:
   - Input: unit_id, flight cycle sensor arrays (matching v1's DL feature vector shape)
   - Output: RUL prediction (mean), uncertainty band (from MC dropout at inference time), latency
6. DynamoDB table `n_cmapss_predictions`: partition key `unit_id`, sort key `timestamp`, attributes: prediction, uncertainty, latency, request_id
7. S3 bucket `n-cmapss-models`: stores versioned model artifacts (linked to W&B Model Registry entries)
8. Test end-to-end: curl the public API, verify DynamoDB row appears

### Deliverables
- `infra/template.yaml` (SAM template)
- `infra/deploy.sh` (deployment script)
- `docs/aws_deployment.md` (how-to guide)
- Public API endpoint URL added to README (with usage examples)
- Cost accounting in `docs/aws_cost_analysis.md`

### Success criteria
- API responds in <2 seconds p95
- Predictions stored in DynamoDB with correct schema
- Total AWS cost = $0 within always-free tier for portfolio-scale traffic (<100 requests/day)
- No NAT Gateway (Lambda VPC-less by default; keep it that way)

### Constraints
- **Stay always-free.** Every choice validated against always-free limits.
- No cross-region calls (data transfer costs)
- No provisioned concurrency (drains credits)
- No warm-up mechanism beyond what's free (accept ~1s cold start for portfolio project)

### Cost accounting (portfolio scale, ~100 requests/day)
- Lambda: 3,000 invocations/month × 2 GB × 2 s = 12,000 GB-seconds → well under 400,000 free
- API Gateway: 3,000 requests/month → well under 1M free
- DynamoDB: 3,000 writes + 3,000 reads/month → well under free tier
- CloudWatch Logs: <100 MB/month → well under 5 GB free
- S3: model artifact ~100 MB → well under 5 GB free
- **Estimated monthly cost: $0**

### References
- AWS Lambda container images: https://docs.aws.amazon.com/lambda/latest/dg/images-create.html
- AWS SAM: https://docs.aws.amazon.com/serverless-application-model/
- API Gateway REST APIs: https://docs.aws.amazon.com/apigateway/latest/developerguide/

---

## V2-Phase 5: A/B testing framework

**Duration:** ~3 days

### Objective
Deploy two model versions, route traffic between them, collect comparative metrics. Signal to recruiters: "understands production experimentation."

### Activities
1. Deploy a second Lambda function with a different model version (e.g., FNN hybrid instead of CNN hybrid, or a different random seed)
2. Routing strategy: API Gateway does not natively support weighted routing to different Lambdas. Options:
   - **Option A (simpler):** single Lambda with in-function A/B split (env var or DynamoDB feature flag for weights)
   - **Option B (more real):** Lambda function URL for each version, plus a thin router Lambda that flips a weighted coin per request
   - Choice: Option A for simplicity; Option B documented as "how you'd do it at scale"
3. Log per-request: which model served (`model_version`), prediction, latency, request_id
4. Analytics notebook `notebooks/v2_ab_analysis.ipynb`:
   - Query DynamoDB for last N days of predictions
   - Compare distributions of predictions between model versions
   - Compare p50/p95 latency
   - Statistical test: are the two models' predictions significantly different?
5. Documentation: how the framework works, how to add a third model, how to change split ratios

### Deliverables
- Updated `src/api.py` with model version routing
- `notebooks/v2_ab_analysis.ipynb`
- `docs/ab_testing.md`

### Success criteria
- Traffic splits between two models at configurable ratio (default 50/50)
- Analytics notebook produces a comparative report
- Documentation includes both current simple approach and how-to-scale approach

### Constraints
- No paid feature flag services (LaunchDarkly, etc.); use DynamoDB or environment variables
- Both model versions fit within same Lambda container OR use two Lambdas (both always-free at portfolio scale)

---

## V2-Phase 6: Monitoring + drift detection

**Duration:** ~3 days

### Objective
Observability for the deployed model. Data drift detection using Evidently AI (open source).

### Activities
1. CloudWatch dashboards:
   - Invocation count per hour
   - Error rate (4xx, 5xx)
   - p50, p95, p99 latency
   - Model version distribution (from V2-Phase 5)
2. Custom CloudWatch metrics from Lambda: emit prediction distribution stats per hour
3. Evidently AI drift monitoring:
   - Reference dataset: training features from v1 (fixed reference)
   - Current dataset: rolling 24-hour window of inference features from DynamoDB
   - Metrics: PSI (population stability index) per feature, Kolmogorov-Smirnov test
   - Runs as a scheduled Lambda (daily) or GitHub Actions cron
4. Alerting: SNS topic subscribed to your email; alarm on error rate > 5% or drift score > 0.25
5. Drift report dashboard (Evidently generates HTML reports; host in S3 static hosting)

### Deliverables
- `infra/cloudwatch_dashboard.json`
- `src/drift_monitor.py`
- `notebooks/v2_drift_report.ipynb`
- SNS topic + subscription in SAM template
- `docs/monitoring.md`

### Success criteria
- CloudWatch dashboard accessible from AWS console
- Drift report generated daily, viewable in browser via S3 static URL
- SNS alert fires when triggered by a manual test event
- All within always-free CloudWatch (10 custom metrics, 5 GB logs) and SNS (1M requests)

### References
- Evidently AI: https://docs.evidentlyai.com
- CloudWatch custom metrics: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/publishingMetrics.html
- SNS: https://docs.aws.amazon.com/sns/

### Constraints
- **10 custom metrics/month limit.** Use them wisely (5-10 metrics max)
- Drift report HTML files in S3: keep under 5 GB total
- No third-party APM (Datadog, New Relic) — costs money

---

## V2-Phase 7: Documentation

**Duration:** ~2 days

### Activities
1. Architecture diagram: data → training → W&B → containerize → GHCR → Lambda → API Gateway → DynamoDB → CloudWatch/Evidently → SNS
2. Runbook `docs/v2_runbook.md`:
   - How to deploy from scratch
   - How to roll back a model version
   - How to interpret CloudWatch dashboards
   - How to respond to a drift alert
   - How to modify the A/B split
3. Cost analysis `docs/aws_cost_analysis.md`:
   - Explicit accounting of what runs on always-free
   - Scale projections: at what request volume does each component start costing money
4. Update main README with v2 section
5. Add v2 badges to README (W&B, GHCR image, live API status via UptimeRobot free tier)

### Deliverables
- `docs/architecture.md` with diagram (mermaid or PNG)
- `docs/v2_runbook.md`
- `docs/aws_cost_analysis.md`
- Updated `README.md`

### Success criteria
- Anyone reading `docs/v2_runbook.md` can redeploy from scratch without asking questions
- Cost analysis is honest and specific
- Main README explains what v2 is and how to browse the deployed artifacts

---

## v2 rules of engagement (in addition to project-wide replication_discipline SKILL)

1. **Every AWS resource has a cost projection** before being created. If it's not in always-free tier, it needs explicit justification.
2. **Infrastructure-as-code only.** No manually-clicked resources in the AWS console except for one-time setup (account creation, IAM setup). Everything else is SAM/CloudFormation.
3. **Tear-down script exists.** Any deployment can be undone with one command. Portfolio project should be reversible.
4. **The v1 model is immutable.** v2 does not retrain, refactor, or "improve" the v1 model. Any behavioral change is an explicit v2.x feature, not a silent tweak.
5. **Public repo, public W&B, public API endpoint.** Recruiter accessibility is the point. If something can't be public (secrets), it goes in GitHub Actions secrets or AWS Secrets Manager, not in the repo.

---

## v2 non-goals (out of scope)

- Kubernetes / EKS (over-engineered for a 24k-param model)
- SageMaker (drains credits, no incremental signal over Lambda for portfolio)
- Airflow / Dagster / Prefect (GitHub Actions cron is enough)
- Real-time streaming (Kinesis, MSK — DS02 is batch)
- Feature store (Feast, Tecton — meaningless at this scale)
- Custom Grafana (CloudWatch built-in is enough)
- Multi-region deployment (single region eu-west-3 Paris for low latency to you)
- Blue/green deployment beyond the A/B framework in V2-Phase 5
