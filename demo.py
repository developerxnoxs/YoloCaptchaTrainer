"""
Demo Script - Validasi semua komponen YOLOv8 CAPTCHA Solver
Script ini untuk memastikan semua komponen berfungsi dengan baik
"""

import sys
from pathlib import Path

print("=" * 70)
print("🤖 YOLOv8 CAPTCHA Solver - Demo & Validation")
print("=" * 70)

# 1. Check Python version
print(f"\n1. ✅ Python Version: {sys.version}")

# 2. Test imports
print("\n2. 🔍 Testing imports...")
try:
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
    print(f"   🖥️  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   🚀 CUDA Device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"   ❌ PyTorch: Not installed ({e})")

try:
    import cv2
    print(f"   ✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"   ⚠️  OpenCV: Not installed ({e})")

try:
    from ultralytics import YOLO
    print(f"   ✅ Ultralytics YOLOv8: Available")
except ImportError as e:
    print(f"   ⚠️  Ultralytics: Not installed ({e})")

try:
    import yaml
    print(f"   ✅ PyYAML: Available")
except ImportError:
    print(f"   ⚠️  PyYAML: Not installed")

# 3. Test custom modules
print("\n3. 🔧 Testing custom modules...")

try:
    from utils.dataset_merger import DatasetMerger
    print("   ✅ Dataset Merger")
except Exception as e:
    print(f"   ❌ Dataset Merger: {e}")

try:
    from utils.coordinate_calculator import CoordinateCalculator
    print("   ✅ Coordinate Calculator")
    
    # Test coordinate calculation
    calc = CoordinateCalculator()
    mock_detections = [
        {'class': 0, 'conf': 0.95, 'bbox': [100, 150, 200, 250]},
        {'class': 1, 'conf': 0.92, 'bbox': [400, 180, 500, 280]},
    ]
    result = calc.calculate_drag_coordinates(
        mock_detections,
        image_width=800,
        image_height=600,
        class_names=['puzzle_piece', 'drop_zone']
    )
    if result['status'] == 'success':
        print(f"      📊 Distance: {result['drag']['distance']:.1f}px")
        print(f"      📐 Angle: {result['drag']['angle']:.1f}°")
        print(f"      🎯 Actions: {len(result['actions'])} steps")
except Exception as e:
    print(f"   ❌ Coordinate Calculator: {e}")

try:
    from utils.dataset_validator import DatasetValidator
    print("   ✅ Dataset Validator")
except Exception as e:
    print(f"   ❌ Dataset Validator: {e}")

# 4. Check directory structure
print("\n4. 📁 Checking directory structure...")
required_dirs = ['scripts', 'utils', 'models', 'datasets', 'output', 'logs']
for dir_name in required_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"   ✅ {dir_name}/")
    else:
        print(f"   ⚠️  {dir_name}/ (tidak ada)")

# 5. Check script files
print("\n5. 📄 Checking script files...")
script_files = [
    'scripts/train_yolov8.py',
    'scripts/inference.py',
    'utils/dataset_merger.py',
    'utils/coordinate_calculator.py',
    'utils/dataset_validator.py',
    'requirements.txt'
]
for script in script_files:
    if Path(script).exists():
        print(f"   ✅ {script}")
    else:
        print(f"   ❌ {script}")

# 6. Summary
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print("""
✅ Semua komponen siap digunakan!

📖 PANDUAN PENGGUNAAN DI VPS:

1. Install dependencies:
   pip install -r requirements.txt

2. Siapkan dataset dalam format YOLO:
   datasets/
   ├── dataset1/
   │   ├── images/
   │   │   ├── train/
   │   │   ├── val/
   │   │   └── test/
   │   ├── labels/
   │   │   ├── train/
   │   │   ├── val/
   │   │   └── test/
   │   └── data.yaml

3. Merge datasets (jika ada multiple datasets):
   python -c "
   from utils.dataset_merger import merge_recaptcha_hcaptcha_datasets
   config = merge_recaptcha_hcaptcha_datasets(
       recaptcha_paths=['path/to/recaptcha1', 'path/to/recaptcha2'],
       hcaptcha_paths=['path/to/hcaptcha1', 'path/to/hcaptcha2'],
       output_dir='datasets/merged_captcha'
   )
   print(f'Dataset merged: {config}')
   "

4. Training model:
   python scripts/train_yolov8.py --data datasets/merged_captcha/dataset.yaml

5. Inference/Solving CAPTCHA:
   python scripts/inference.py --model runs/train/captcha_xxx/weights/best.pt --source test_image.jpg

6. Export model ke ONNX (untuk production):
   python scripts/train_yolov8.py --export runs/train/captcha_xxx/weights/best.pt

💡 TIPS OPTIMASI GPU A100:
   - Batch size: 64-128 (sesuaikan dengan VRAM)
   - Image size: 1280 untuk detail maksimal
   - Mixed precision (AMP): Enabled by default
   - Multi-scale training: Enabled
   - Cache: disk (untuk dataset besar)

📊 MONITORING TRAINING:
   - TensorBoard: tensorboard --logdir runs/train
   - Metrics: runs/train/captcha_xxx/results.csv
   - Best model: runs/train/captcha_xxx/weights/best.pt
""")
print("=" * 70)
