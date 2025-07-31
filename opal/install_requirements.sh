#!/bin/bash
set -e  # Exit on error

ENV_NAME="opal_env"
LOGFILE="setup_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee -a $LOGFILE
echo "🔹 Starting environment setup..." | tee -a $LOGFILE
echo " Mode: ${1:-cpu}" | tee -a $LOGFILE
echo " Log file: $LOGFILE" | tee -a $LOGFILE
echo "==========================================" | tee -a $LOGFILE

# ✅ Check Python version
echo "➡ Using Python version:" | tee -a $LOGFILE
python3 --version | tee -a $LOGFILE

# ✅ Create virtual environment if not exists
if [ ! -d "$ENV_NAME" ]; then
    echo "➡ Creating virtual environment: $ENV_NAME" | tee -a $LOGFILE
    python3 -m venv $ENV_NAME
else
    echo "✅ Virtual environment $ENV_NAME already exists" | tee -a $LOGFILE
fi

# ✅ Activate virtual environment
source $ENV_NAME/bin/activate
echo "✅ Virtual environment activated: $(pwd)/$ENV_NAME" | tee -a $LOGFILE

# ✅ Upgrade pip
pip install --upgrade pip 2>&1 | tee -a $LOGFILE

if [ "$1" == "gpu" ]; then
    echo "➡ Installing GPU requirements..." | tee -a $LOGFILE
    pip install -r requirements_gpu.txt --no-cache-dir 2>&1 | tee -a $LOGFILE

    echo "➡ Installing GPU-enabled PyTorch (CUDA 12.1)..." | tee -a $LOGFILE
    pip install torch==2.7.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.7.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121 2>&1 | tee -a $LOGFILE

    echo "➡ Verifying GPU availability..." | tee -a $LOGFILE
    python3 - <<'EOF' 2>&1 | tee -a $LOGFILE
import torch
if torch.cuda.is_available():
    print("✅ GPU is available! Using:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU is NOT available. Check drivers or CUDA installation.")
EOF

else
    echo "➡ Installing CPU requirements..." | tee -a $LOGFILE
    pip install -r requirements_cpu.txt 2>&1 | tee -a $LOGFILE
fi

echo "==========================================" | tee -a $LOGFILE
echo "✅ Environment setup completed successfully!" | tee -a $LOGFILE
echo " Virtual environment path: $(pwd)/$ENV_NAME" | tee -a $LOGFILE
echo "==========================================" | tee -a $LOGFILE
