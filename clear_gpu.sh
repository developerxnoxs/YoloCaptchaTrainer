#!/bin/bash

# ============================================================================
# Clear GPU Memory & Kill Stuck Processes
# ============================================================================

echo "========================================================================="
echo "🔧 Clearing GPU Memory"
echo "========================================================================="
echo ""

# Kill all Python processes (training yang stuck)
echo "1. Killing stuck Python processes..."
pkill -9 python
pkill -9 python3
sleep 2

# Kill any CUDA processes
echo "2. Killing CUDA processes..."
sudo fuser -k /dev/nvidia*
sleep 2

# Clear PyTorch cache (via Python)
echo "3. Clearing PyTorch cache..."
python3 -c "import torch; torch.cuda.empty_cache(); print('✅ CUDA cache cleared')"

# Check GPU status
echo ""
echo "========================================================================="
echo "📊 GPU Status After Cleanup"
echo "========================================================================="
nvidia-smi

echo ""
echo "========================================================================="
echo "✅ GPU Memory Cleared!"
echo "========================================================================="
echo ""
echo "Sekarang Anda bisa run training lagi dengan batch size lebih kecil:"
echo "  python scripts/train_yolov8.py --data datasets/merged_all/dataset.yaml"
echo ""
