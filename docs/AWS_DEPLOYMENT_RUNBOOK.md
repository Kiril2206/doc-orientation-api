# AWS Deployment Runbook: Document Orientation API

This runbook provides step-by-step instructions for deploying the **Document Orientation API** to Amazon Web Services (AWS). It covers three distinct deployment options:

1. **Option 1: AWS EC2 Free Tier (100% Free - $0.00/mo)**: Recommended for personal testing, portfolios, and zero-cost hosting (750 hours/month free).
2. **Option 2: AWS App Runner**: Recommended for managed serverless container hosting (automatic TLS, managed auto-scaling).
3. **Option 3: AWS ECS Fargate + ALB**: Recommended for enterprise production with full Infrastructure-as-Code (CloudFormation).

---

## 1. Architecture & Resource Sizing

### Sizing Rationale
- **Compute Recommendation**:
  - **EC2 Free Tier**: `t2.micro` (1 vCPU, 1 GiB RAM) augmented with a **2 GiB swapfile** (`/swapfile`) to ensure memory safety during PyMuPDF rasterization and ONNX inference.
  - **ECS / App Runner**: **2 vCPU, 4 GiB RAM** (recommended for concurrent multi-page workloads).
- **Single-Worker Execution Strategy**:
  - The API uses a single-worker process (`--workers 1`) backed by `DocumentProcessor` with a dedicated single-threaded executor.
  - This design intentionally isolates native C++ libraries (PyMuPDF, RapidOCR, ONNX Runtime) from cross-thread memory corruption and heap fragmentation.
  - Scale out **horizontally** (more container tasks or instances behind an ALB), rather than vertically adding threads.
- **Health Checks**:
  - **Liveness (`/live`)**: Returns HTTP 200 immediately without loading models or requiring authentication. Used by ALB/target group health monitors.
  - **Readiness (`/ready`)**: Verifies that the required ONNX models and configured pipelines are loaded and ready to serve traffic.

---

## 2. Option 1: AWS EC2 Free Tier ($0.00/mo)

Deploy directly to an Amazon EC2 `t2.micro` or `t3.micro` instance within the 750 hrs/month AWS Free Tier.

### Step 1: Launch EC2 Instance
1. Open the **AWS EC2 Console** and click **Launch instance**.
2. **Name**: `doc-orientation-api`
3. **OS**: **Amazon Linux 2023** (AMI: default AL2023 HVM).
4. **Instance Type**: `t2.micro` (Free Tier eligible) or `t3.micro`.
5. **Key Pair**: Select your SSH key pair or create a new one.
6. **Security Group**: Allow incoming traffic on:
   - **SSH** (port 22) from your IP.
   - **HTTP** (port 80) from anywhere (`0.0.0.0/0`).
7. **Storage**: 8 GiB gp3 (default, Free Tier eligible).
8. **Advanced Details -> User Data**:
   Paste the contents of [`aws/ec2-userdata.sh`](../aws/ec2-userdata.sh):

```bash
#!/bin/bash
set -e
exec > /var/log/user-data.log 2>&1

# 1. 2 GB swap for memory safety on 1 GB RAM
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "/swapfile swap swap defaults 0 0" >> /etc/fstab

# 2. System dependencies
dnf update -y
dnf install -y git python3.12 python3.12-pip mesa-libGL glib2

# 3. Clone repository
mkdir -p /opt/app
cd /opt/app
git clone https://github.com/Kiril2206/doc-orientation-api.git .

# 4. Virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/generate_sample_files.py

# 5. systemd service on port 80
cat << 'EOF' > /etc/systemd/system/doc-orientation.service
[Unit]
Description=Document Orientation API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/app
ExecStart=/opt/app/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 80 --workers 1
Restart=always
RestartSec=5
EnvironmentFile=-/opt/app/.env
Environment=APP_HOST=0.0.0.0
Environment=APP_PORT=80
Environment=APP_INFERENCE_MODE=pure
Environment=APP_REQUIRE_AUTH=false

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now doc-orientation.service
```

9. Click **Launch Instance**. Within 3–4 minutes, the API and web UI are live at `http://<YOUR_EC2_PUBLIC_IP>/`.

### Configuring the Gemini API Key on EC2
To enable the `genai` multimodal route on your running EC2 instance:

```bash
# SSH into EC2
ssh -i your-key.pem ec2-user@<YOUR_EC2_PUBLIC_IP>

# Append your API key and preferred model to /opt/app/.env
echo "APP_GEMINI_API_KEY=your_actual_api_key" | sudo tee -a /opt/app/.env
echo "APP_GEMINI_MODEL=gemini-3.5-flash-lite" | sudo tee -a /opt/app/.env

# Restart the systemd service
sudo systemctl restart doc-orientation
```

---

## 3. Option 2: AWS App Runner (Serverless Containers)

1. Build and push the Docker image to Amazon ECR:
   ```bash
   export AWS_REGION="us-east-1"
   export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
   export REPO_NAME="doc-orientation-api"

   aws ecr create-repository --repository-name ${REPO_NAME} --region ${AWS_REGION} || true
   aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

   docker build -t ${REPO_NAME}:latest .
   docker tag ${REPO_NAME}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest
   docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest
   ```

2. Open the **AWS App Runner Console** -> **Create service**:
   - **Source**: ECR image `${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/doc-orientation-api:latest`.
   - **Compute**: 2 vCPU, 4 GiB RAM.
   - **Port**: 8000.
   - **Health check path**: `/live` (Interval: 10s, Timeout: 5s).
   - **Environment Variables**:
     - `APP_INFERENCE_MODE` = `pure`
     - `APP_REQUIRE_AUTH` = `false`
     - `APP_GEMINI_API_KEY` = (optional, via Secrets Manager or direct)
3. Click **Create & Deploy**. App Runner provides an HTTPS URL within minutes.

---

## 4. Option 3: AWS ECS Fargate + ALB (CloudFormation)

For enterprise infrastructure with isolated subnets and ALB routing, use the bundled template:

```bash
# 1. Store secrets in SSM Parameter Store
aws ssm put-parameter \
  --name "/doc-orientation/demo-password" \
  --type "SecureString" \
  --value "ProductionSecurePassword2026!" \
  --overwrite

aws ssm put-parameter \
  --name "/doc-orientation/gemini-api-key" \
  --type "SecureString" \
  --value "YOUR_GEMINI_API_KEY" \
  --overwrite

# 2. Deploy CloudFormation stack
aws cloudformation deploy \
  --template-file aws/cloudformation-template.yml \
  --stack-name doc-orientation-prod \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      ImageUri="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/doc-orientation-api:latest" \
      DemoPasswordParameterArn="arn:aws:ssm:${AWS_REGION}:${AWS_ACCOUNT_ID}:parameter/doc-orientation/demo-password" \
      GeminiApiKeyParameterArn="arn:aws:ssm:${AWS_REGION}:${AWS_ACCOUNT_ID}:parameter/doc-orientation/gemini-api-key"

# 3. Retrieve service URL
aws cloudformation describe-stacks \
  --stack-name doc-orientation-prod \
  --query "Stacks[0].Outputs[?OutputKey=='ServiceUrl'].OutputValue" \
  --output text
```

---

## 5. Verification & Health Monitoring

Verify endpoints against your deployed host:

```bash
# 1. Liveness check (ALB target group)
curl -i http://<HOST>/live
# HTTP/1.1 200 OK
# {"status":"alive"}

# 2. Readiness check (Model warmup validation)
curl -i http://<HOST>/ready
# HTTP/1.1 200 OK
# {"status":"ready"}

# 3. Full diagnostic health check
curl -i http://<HOST>/health
```

---

## 6. Cost Estimation Comparison

| Deployment Model | Compute Specs | Estimated Monthly Cost |
| :--- | :--- | :--- |
| **AWS EC2 Free Tier** | `t2.micro` (1 vCPU, 1 GB RAM + 2 GB swap, 8 GB gp3) | **$0.00 / month** (within 750 hrs free tier) |
| **AWS App Runner** | 2 vCPU / 4 GB RAM (active hours, auto-paused when idle) | ~$15 – $25 / month |
| **AWS ECS Fargate** | 1 task (2 vCPU, 4 GB 24/7) + Application Load Balancer | ~$35 – $48 / month |
| **CloudWatch Logs** | 30-day retention (~5 GB logs) | ~$2.50 / month |
