#!/bin/bash

# ============================================================================
# Setup Script - YOLOv8 CAPTCHA Solver untuk GPU A100 (80GB)
# ============================================================================
# Script ini akan otomatis:
# 1. Check sistem requirements
# 2. Install semua dependencies
# 3. Setup environment
# ============================================================================

set -e  # Exit on error

echo "========================================================================="
echo "🚀 YOLOv8 CAPTCHA Solver - Automated Setup"
echo "========================================================================="
echo ""

# Warna untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function untuk print dengan warna
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# ============================================================================
# 1. Check Python
# ============================================================================
echo "📋 Checking system requirements..."
echo ""

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 tidak ditemukan!"
    echo "Install Python 3.8+ terlebih dahulu"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION ditemukan"

# ============================================================================
# 2. Check CUDA & GPU
# ============================================================================
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU ditemukan"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    print_error "NVIDIA GPU tidak terdeteksi!"
    print_info "Script akan tetap berjalan tapi training akan lambat tanpa GPU"
    echo ""
fi

# ============================================================================
# 3. Check/Create Virtual Environment
# ============================================================================
print_info "Setting up Python virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment dibuat"
else
    print_info "Virtual environment sudah ada"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

# ============================================================================
# 4. Install Dependencies
# ============================================================================
print_info "Installing dependencies..."
echo "Ini mungkin memakan waktu beberapa menit..."
echo ""

# Install PyTorch dengan CUDA support
print_info "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

# Install requirements
if [ -f "requirements.txt" ]; then
    print_info "Installing from requirements.txt..."
    pip install -r requirements.txt -q
fi

# Install tambahan yang diperlukan
print_info "Installing additional packages..."
pip install roboflow -q

print_success "Semua dependencies berhasil diinstall"
echo ""

# ============================================================================
# 5. Verify Installation
# ============================================================================
print_info "Verifying installation..."

python3 << 'END_PYTHON'
import sys
import torch
from ultralytics import YOLO

print(f"✅ Python: {sys.version}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  WARNING: CUDA tidak tersedia, training akan sangat lambat!")

print(f"✅ Ultralytics YOLOv8 tersedia")
END_PYTHON

echo ""
print_success "Setup completed!"
echo ""

# ============================================================================
# 6. Create directories
# ============================================================================
print_info "Creating project directories..."

mkdir -p datasets/downloaded
mkdir -p datasets/merged
mkdir -p models
mkdir -p runs/train
mkdir -p logs
mkdir -p output

print_success "Directories created"
echo ""

# ============================================================================
# 7. Final Instructions
# ============================================================================
echo "========================================================================="
echo "✅ SETUP SELESAI!"
echo "========================================================================="
echo ""
echo "Langkah selanjutnya:"
echo ""
echo "1. Set Roboflow API Key:"
echo "   export ROBOFLOW_API_KEY='your_api_key_here'"
echo "   (Dapatkan dari: https://app.roboflow.com/settings/api)"
echo ""
echo "2. Jalankan training otomatis:"
echo "   ./run_auto_training.sh"
echo ""
echo "   ATAU jalankan manual step-by-step:"
echo "   source venv/bin/activate"
echo "   python auto_train.py"
echo ""
echo "========================================================================="
