#!/bin/bash
# ==============================================================================
# AWS EC2 Free Tier Automated Deployment Script (Amazon Linux 2023)
# Completely Free: Runs 24/7 on t2.micro within the 750 hrs/mo AWS Free Tier
# ==============================================================================

set -e
exec > /var/log/user-data.log 2>&1

echo "--- 1. Setting up 2 GB swapfile for memory safety on t2.micro ---"
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "/swapfile swap swap defaults 0 0" >> /etc/fstab

echo "--- 2. Installing system dependencies ---"
dnf update -y
dnf install -y git python3.12 python3.12-pip mesa-libGL glib2

echo "--- 3. Cloning repository ---"
mkdir -p /opt/app
cd /opt/app
git clone https://github.com/Kiril2206/doc-orientation-api.git .

echo "--- 4. Creating virtual environment and installing dependencies ---"
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "--- 5. Generating synthetic demonstration sample files ---"
python scripts/generate_sample_files.py

echo "--- 6. Setting up systemd service on port 80 ---"
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
Environment=APP_HOST=0.0.0.0
Environment=APP_PORT=80
Environment=APP_INFERENCE_MODE=pure
Environment=APP_REQUIRE_AUTH=false

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now doc-orientation.service

echo "--- Deployment Complete! Service is running on port 80 ---"
