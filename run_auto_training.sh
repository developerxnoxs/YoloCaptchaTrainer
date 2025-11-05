#!/bin/bash

# ============================================================================
# Run Auto Training - One-Click Training Script
# ============================================================================
# Script wrapper untuk menjalankan full training pipeline
# ============================================================================

set -e

echo "========================================================================="
echo "🚀 YOLOv8 CAPTCHA Solver - Auto Training"
echo "========================================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment tidak ditemukan!"
    echo "Jalankan setup terlebih dahulu: ./setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check ROBOFLOW_API_KEY
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo "⚠️  ROBOFLOW_API_KEY tidak di-set"
    echo ""
    echo "Dapatkan API key dari: https://app.roboflow.com/settings/api"
    echo ""
    read -p "Masukkan API key (atau tekan Enter untuk skip download): " api_key
    
    if [ -n "$api_key" ]; then
        export ROBOFLOW_API_KEY="$api_key"
        echo "✅ API Key set"
    else
        echo "⚠️  Download akan di-skip"
        SKIP_DOWNLOAD="--skip-download"
    fi
else
    echo "✅ ROBOFLOW_API_KEY ditemukan: ${ROBOFLOW_API_KEY:0:8}..."
    SKIP_DOWNLOAD=""
fi

echo ""
echo "========================================================================="
echo "Starting training pipeline..."
echo "========================================================================="
echo ""

# Run auto training
python auto_train.py $SKIP_DOWNLOAD "$@"

# Save completion status
echo ""
echo "========================================================================="
echo "✅ Script selesai!"
echo "========================================================================="
echo ""
echo "Untuk melihat hasil training:"
echo "  ls -lah runs/train/"
echo ""
echo "Untuk menjalankan inference:"
echo "  python scripts/inference.py --model runs/train/captcha_*/weights/best.pt --image path/to/image.jpg"
echo ""
