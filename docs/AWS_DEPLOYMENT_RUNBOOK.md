# AWS Deployment Runbook: Document Orientation API

This guide provides step-by-step instructions for deploying the **Document Orientation API** to Amazon Web Services (AWS). It covers two deployment models:
1. **AWS App Runner**: Recommended for fast, fully managed demo hosting (5-minute deployment, minimal setup).
2. **AWS ECS Fargate + ALB**: Recommended for production enterprise infrastructure with Infrastructure-as-Code (CloudFormation).

---

## 1. Architecture & Resource Sizing

### Sizing Rationale
- **Compute**: **2 vCPU, 4 GiB RAM** (Minimum recommended).
  - RapidOCR + PyMuPDF rasterization can consume 1.2–2.0 GiB during concurrent multi-page PDF processing.
  - ONNX Runtime letterboxing and inference allocate contiguous tensors in memory.
- **Worker Concurrency Strategy**:
  - The API uses a single-worker Uvicorn process (`--workers 1`) backed by `DemoProcessor` with a dedicated single-threaded executor.
  - This design intentionally protects underlying native C++ libraries (PyMuPDF, RapidOCR, ONNX Runtime) from cross-thread state corruption and heap fragmentation.
  - Concurrency scaling is achieved **horizontally** via AWS ALB target auto-scaling, rather than vertical multi-threading.
- **Health Checks**:
  - **Liveness (`/live`)**: Returns HTTP 200 immediately without loading models or requiring authentication. Used by ALB target group health checks.
  - **Readiness (`/ready`)**: Verifies that the required ONNX models and configured external APIs are warmed up and ready to serve traffic.

---

## 2. Secrets Management (AWS Systems Manager Parameter Store)

Store sensitive keys as `SecureString` parameters in AWS Systems Manager (SSM) Parameter Store:

```bash
# 1. Set the live demo password (must be at least 16 characters)
aws ssm put-parameter \
  --name "/doc-orientation/demo-password" \
  --type "SecureString" \
  --value "SeniorInterview2026!SecureKey" \
  --overwrite

# 2. Set the Google Gemini API key (optional, required only for GenAI mode)
aws ssm put-parameter \
  --name "/doc-orientation/gemini-api-key" \
  --type "SecureString" \
  --value "YOUR_GEMINI_API_KEY" \
  --overwrite
```

---

## 3. Container Image Build & ECR Push

Authenticate Docker to your Amazon ECR registry:

```bash
# Set your AWS variables
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO_NAME="doc-orientation-api"

# Create ECR repository (if not already created)
aws ecr create-repository --repository-name ${REPO_NAME} --region ${AWS_REGION} || true

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build the hardened multi-stage image (note: .dockerignore excludes .venv, data, output to keep context < 40MB)
docker build -t ${REPO_NAME}:latest .

# Tag and push
docker tag ${REPO_NAME}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest
```

---

## 4. Deployment Option A: AWS App Runner (Fast Demo)

AWS App Runner provides a serverless container runner with automatic TLS, health checks, and autoscaling.

1. Navigate to **AWS App Runner** in the AWS Console.
2. Click **Create service**:
   - **Source**: Container registry -> Amazon ECR.
   - Select `${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/doc-orientation-api:latest`.
   - **Deployment trigger**: Manual or Automatic.
   - **ECR access role**: Create new or use existing `AppRunnerECRAccessRole`.
3. Configure service:
   - **Service name**: `doc-orientation-api`
   - **CPU**: `2 vCPU`, **Memory**: `4 GB`.
   - **Port**: `8000`.
   - **Health check path**: `/live` (Interval: 10s, Timeout: 5s, Healthy threshold: 1, Unhealthy: 3).
4. Add Environment Variables:
   - `APP_INFERENCE_MODE` = `pure`
   - `APP_REQUIRE_AUTH` = `true`
   - `APP_DEMO_USERNAME` = `demo`
   - Reference SSM parameters for `APP_DEMO_PASSWORD` and `APP_GEMINI_API_KEY`.
5. Click **Create & Deploy**. In ~3 minutes, App Runner provides a live HTTPS URL (e.g. `https://xyz123.us-east-1.awsapprunner.com`).

---

## 5. Deployment Option B: AWS ECS Fargate + ALB (Production CloudFormation)

For enterprise infrastructure with isolated subnets and ALB path routing, use the bundled template:

```bash
aws cloudformation deploy \
  --template-file aws/cloudformation-template.yml \
  --stack-name doc-orientation-prod \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      ImageUri="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/doc-orientation-api:latest" \
      DemoPasswordParameterArn="arn:aws:ssm:${AWS_REGION}:${AWS_ACCOUNT_ID}:parameter/doc-orientation/demo-password" \
      GeminiApiKeyParameterArn="arn:aws:ssm:${AWS_REGION}:${AWS_ACCOUNT_ID}:parameter/doc-orientation/gemini-api-key"
```

Once deployment completes, retrieve the public URL:
```bash
aws cloudformation describe-stacks \
  --stack-name doc-orientation-prod \
  --query "Stacks[0].Outputs[?OutputKey=='ServiceUrl'].OutputValue" \
  --output text
```

---

## 6. Verification Checklist

1. **Liveness Check**:
   ```bash
   curl -i http://<ALB-OR-APPRUNNER-URL>/live
   # HTTP/1.1 200 OK
   # {"status":"alive"}
   ```
2. **Readiness Check**:
   ```bash
   curl -i http://<ALB-OR-APPRUNNER-URL>/ready
   # HTTP/1.1 200 OK
   # {"status":"ready"}
   ```
3. **Demo UI & Authentication**:
   - Open `http://<ALB-OR-APPRUNNER-URL>/` in your browser.
   - Enter username `demo` and password from SSM.
   - Click "Sample Invoice (90°)" and verify automatic correction in ~30ms.
4. **Header Validation**:
   - Check response headers for `X-Content-Type-Options: nosniff`, `Content-Security-Policy`, and `X-Request-Id`.

---

## 7. Cost Estimation (Monthly)

| Service | Configuration | Estimated Monthly Cost |
| :--- | :--- | :--- |
| **AWS App Runner** | 2 vCPU / 4 GB, active during business hours (paused when idle) | ~$15 – $25 / month |
| **AWS ECS Fargate** | 1 task (2 vCPU, 4 GB 24/7) + Application Load Balancer | ~$35 – $48 / month |
| **Amazon ECR** | 1 repository (< 1 GB compressed image) | ~$0.10 / month |
| **CloudWatch Logs** | 30 days retention (~5 GB logs) | ~$2.50 / month |

---

## 8. Teardown / Cleanup

To avoid ongoing AWS charges after the demonstration:

```bash
# Option A (App Runner):
aws apprunner delete-service --service-arn <SERVICE_ARN>

# Option B (CloudFormation / ECS):
aws cloudformation delete-stack --stack-name doc-orientation-prod
```

