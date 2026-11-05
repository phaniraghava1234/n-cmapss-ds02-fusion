# REFERENCES.md

## Required (methods, replication core)

**[1]** Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2022).
"Fusing physics-based and deep learning models for prognostics."
*Reliability Engineering & System Safety*, 217, 107961.
DOI: 10.1016/j.ress.2021.107961
Open access.

**[2]** Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2021).
"Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics."
*Data*, 6(1), 5.
DOI: 10.3390/data6010005
Open access.
Dataset URL: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
Direct download: https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip

**[3]** Frederick, D. K., DeCastro, J. A., & Litt, J. S. (2007).
"User's Guide for the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)."
NASA/TM-2007-215026.
Referenced by [1] and [2] for CMAPSS documentation. We do not access CMAPSS directly in this project.

## Background (README field context only, not used as method reference)

**[4]** Volponi, A. J. (2014).
"Gas Turbine Engine Health Management: Past, Present, and Future Trends."
*ASME Journal of Engineering for Gas Turbines and Power*, 136(5), 051201.
DOI: 10.1115/1.4026126
Cited for framing this project within the gas path analysis / engine health management tradition. Not used as a method reference.

## Parked for future work (NOT v1)

**[5]** Xiong, J., Fink, O., Zhou, J., & Ma, Y. (2023).
"Controlled physics-informed data generation for deep learning-based remaining useful life prediction under unseen operation conditions."
*Mechanical Systems and Signal Processing*, 197, 110359.
DOI: 10.1016/j.ymssp.2023.110359

**[6]** Liao, X., Chen, S., Wen, P., & Zhao, S. (2023).
"Remaining useful life with self-attention assisted physics-informed neural network."
*Advanced Engineering Informatics*, 58, 102195.
DOI: 10.1016/j.aei.2023.102195

## Attempted but inaccessible

**[gitlab]** Public code repository referenced by [1] §5.3 footnote 2:
`https://gitlab.ethz.ch/arimanue/Fusing-physics-based-and-deep-learning-models-for-prognostics`
Per paper text, scope is the CMAPSS FD002 validation subsection, not the main N-CMAPSS DS02 case study.
Neither user nor Claude could access the repo at project start. Treated as unavailable.
Revisit if access becomes possible.

---

## v1.5 dashboard tooling

**[gradio]** Gradio Python SDK for ML dashboards. https://www.gradio.app/docs

**[hf-spaces]** Hugging Face Spaces documentation. https://huggingface.co/docs/hub/spaces

## v2 MLOps tooling

**[wandb]** Weights & Biases documentation. https://docs.wandb.ai

**[wandb-registry]** W&B Model Registry guide. https://docs.wandb.ai/guides/model_registry

**[docker]** Docker documentation. https://docs.docker.com

**[gh-actions]** GitHub Actions documentation. https://docs.github.com/en/actions

**[aws-lambda]** AWS Lambda Developer Guide. https://docs.aws.amazon.com/lambda/latest/dg/

**[aws-lambda-container]** AWS Lambda container images. https://docs.aws.amazon.com/lambda/latest/dg/images-create.html

**[aws-apigw]** AWS API Gateway REST APIs. https://docs.aws.amazon.com/apigateway/latest/developerguide/

**[aws-sam]** AWS SAM (Serverless Application Model). https://docs.aws.amazon.com/serverless-application-model/

**[aws-dynamodb]** AWS DynamoDB Developer Guide. https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/

**[aws-free]** AWS Free Tier (current model, post-July 2025). https://aws.amazon.com/free/

**[evidently]** Evidently AI documentation (data drift and model monitoring). https://docs.evidentlyai.com

**[aws-cloudwatch]** AWS CloudWatch documentation. https://docs.aws.amazon.com/cloudwatch/
