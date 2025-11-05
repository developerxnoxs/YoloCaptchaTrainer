"""
Example Usage - Panduan lengkap penggunaan YOLOv8 CAPTCHA Solver
Script ini menunjukkan workflow lengkap dari download dataset sampai inference
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   YOLOv8 CAPTCHA Solver - Example Usage                     ║
║                  reCAPTCHA v2, hCaptcha, Drag Puzzle Detection              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📖 PANDUAN LENGKAP PENGGUNAAN DI VPS GPU A100
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: INSTALL DEPENDENCIES                                                 │
└──────────────────────────────────────────────────────────────────────────────┘

pip install -r requirements.txt

Dependencies yang akan diinstall:
- ultralytics (YOLOv8 framework)
- torch + torchvision (Deep Learning)
- opencv-python (Image processing)
- roboflow (Dataset download)
- Dan lainnya...

Waktu install: ~5-10 menit (tergantung koneksi)
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: DOWNLOAD DATASETS                                                    │
└──────────────────────────────────────────────────────────────────────────────┘

A. Cara Otomatis (Recommended):

# 1. Dapatkan Roboflow API Key
#    - Buat akun di: https://roboflow.com
#    - Ambil key di: https://app.roboflow.com/settings/api
#    - Free tier: 10,000 API calls/month

# 2. Set environment variable
export ROBOFLOW_API_KEY='your_api_key_here'

# 3. Download & merge semua dataset
python scripts/download_datasets.py --merge

Dataset yang akan didownload:
├── reCAPTCHA v2 (Roboflow) - 1,828 images
├── hCaptcha Challenger - 712 images
├── hCaptcha Images - multiple classes
├── Slide/Drag Puzzle - 2,778 images
└── Captcha Detection - puzzle pieces, targets

Total: ~5,300+ annotated images
Output: datasets/downloaded/merged_all/dataset.yaml

B. Cara Manual (Jika tidak ada Roboflow API):

# Download specific dataset
python scripts/download_datasets.py \\
    --dataset slide_captcha \\
    --api-key your_key

# Atau download manual dari browser:
# 1. https://universe.roboflow.com/my-workspace-4p8ud/recaptcha-v2
# 2. https://universe.roboflow.com/qin2dim/hcaptcha-challenger
# 3. https://universe.roboflow.com/captcha-lwpyk/slide_captcha
# Extract ke folder datasets/
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: VALIDASI DATASET                                                     │
└──────────────────────────────────────────────────────────────────────────────┘

# Validasi struktur dan format
python utils/dataset_validator.py datasets/downloaded/merged_all/dataset.yaml

Output akan menunjukkan:
✅ Dataset structure valid
✅ All annotations in correct format
✅ Train/Val/Test split information
⚠️  Warnings (jika ada missing labels)
❌ Errors (jika ada format issues)
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: TRAINING MODEL (GPU A100 80GB)                                       │
└──────────────────────────────────────────────────────────────────────────────┘

# Basic training (dengan default config optimal untuk A100)
python scripts/train_yolov8.py \\
    --data datasets/downloaded/merged_all/dataset.yaml

Konfigurasi Default:
├── Model: YOLOv8x (largest, highest accuracy)
├── Batch Size: 64 (optimal untuk A100 80GB)
├── Image Size: 1280 (high resolution)
├── Epochs: 300
├── Mixed Precision: Enabled (AMP)
├── Multi-scale: Enabled
└── Cache: disk

Training Time Estimate:
- Dataset 5,000 images: ~3-5 hours
- Dataset 10,000 images: ~6-10 hours
- Dataset 50,000 images: ~1-2 days

Real-time Monitoring:
# Terminal 1: Training
python scripts/train_yolov8.py --data dataset.yaml

# Terminal 2: TensorBoard
tensorboard --logdir runs/train

# Buka browser: http://localhost:6006

Output files:
runs/train/captcha_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt        ← Best model (highest mAP)
│   └── last.pt        ← Latest epoch
├── results.csv        ← Metrics (mAP, precision, recall)
├── results.png        ← Training curves
├── confusion_matrix.png
└── val_batch*.jpg     ← Validation predictions
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: EVALUASI MODEL                                                       │
└──────────────────────────────────────────────────────────────────────────────┘

# Check training results
cat runs/train/captcha_xxx/results.csv | tail -10

Expected Good Metrics (after 300 epochs):
├── mAP50: 0.85 - 0.95
├── mAP50-95: 0.70 - 0.85
├── Precision: 0.80 - 0.90
└── Recall: 0.75 - 0.90

If metrics are low:
→ Train longer: --epochs 500
→ Add more data
→ Check data quality (annotations)
→ Tune hyperparameters
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: INFERENCE / SOLVING CAPTCHA                                          │
└──────────────────────────────────────────────────────────────────────────────┘

A. Single Image Test:

python scripts/inference.py \\
    --model runs/train/captcha_xxx/weights/best.pt \\
    --source test_captcha.jpg \\
    --output output/results

Output:
├── test_captcha_solved.jpg  ← Visualized with boxes & arrows
└── Console output:
    🔍 Processing: test_captcha.jpg
       Detections: 2
       Drag Status: success
       Distance: 245.3px
       Angle: 12.5°

B. Batch Processing:

python scripts/inference.py \\
    --model runs/train/captcha_xxx/weights/best.pt \\
    --source test_images/ \\
    --output output/batch_results \\
    --save-json

Output:
output/batch_results/
├── image1_solved.jpg
├── image2_solved.jpg
├── ...
└── results.json  ← All predictions in JSON

C. Programmatic Usage:

python -c "
from scripts.inference import CaptchaSolver

solver = CaptchaSolver('runs/train/captcha_xxx/weights/best.pt')
result = solver.solve_captcha('test.jpg')

if result['solved']:
    print('✅ CAPTCHA Solved!')
    for action in result['actions']:
        print(f\"{action['action']}: ({action['x']}, {action['y']}) - {action['delay_ms']}ms\")
else:
    print(f'❌ Failed: {result[\"message\"]}')
"

Actions output (untuk automation):
mousedown: (150, 200) - 100ms
mousemove: (180, 205) - 50ms
mousemove: (210, 210) - 50ms
...
mousemove: (450, 230) - 50ms
mouseup: (450, 230) - 100ms
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: EXPORT MODEL UNTUK PRODUCTION                                        │
└──────────────────────────────────────────────────────────────────────────────┘

# Export ke ONNX (universal format)
python scripts/train_yolov8.py \\
    --export runs/train/captcha_xxx/weights/best.pt

Output:
runs/train/captcha_xxx/weights/
├── best.pt              ← PyTorch (training/inference)
├── best.onnx            ← ONNX (universal deployment)
└── best.torchscript     ← TorchScript (PyTorch production)

ONNX Benefits:
✅ Platform independent
✅ Faster inference
✅ Smaller file size
✅ Compatible dengan ONNX Runtime, TensorRT, etc.

Usage dengan ONNX:
python scripts/inference.py \\
    --model runs/train/captcha_xxx/weights/best.onnx \\
    --source test.jpg
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: INTEGRATION DENGAN AUTOMATION                                        │
└──────────────────────────────────────────────────────────────────────────────┘

Example: Selenium Integration

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from scripts.inference import CaptchaSolver
import time

# Setup
driver = webdriver.Chrome()
solver = CaptchaSolver('runs/train/captcha_xxx/weights/best.pt')

# Navigate to page with CAPTCHA
driver.get('https://example.com/captcha')

# Take screenshot
driver.save_screenshot('captcha.png')

# Solve CAPTCHA
result = solver.solve_captcha('captcha.png')

if result['solved']:
    # Execute drag actions
    element = driver.find_element_by_id('captcha-canvas')
    actions = ActionChains(driver)
    
    for action in result['actions']:
        if action['action'] == 'mousedown':
            actions.click_and_hold(element).perform()
        elif action['action'] == 'mousemove':
            actions.move_by_offset(
                action['x'] - prev_x, 
                action['y'] - prev_y
            ).perform()
        elif action['action'] == 'mouseup':
            actions.release().perform()
        
        time.sleep(action['delay_ms'] / 1000)
        prev_x, prev_y = action['x'], action['y']

Example: Playwright Integration

from playwright.sync_api import sync_playwright
from scripts.inference import CaptchaSolver

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    solver = CaptchaSolver('runs/train/captcha_xxx/weights/best.pt')
    
    page.goto('https://example.com/captcha')
    page.screenshot(path='captcha.png')
    
    result = solver.solve_captcha('captcha.png')
    
    if result['solved']:
        # Get drag coordinates
        start = result['drag_info']['from']
        end = result['drag_info']['to']
        
        # Execute drag
        page.mouse.move(start[0], start[1])
        page.mouse.down()
        page.mouse.move(end[0], end[1], steps=10)
        page.mouse.up()
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ ADVANCED: CUSTOM DATASET & FINE-TUNING                                       │
└──────────────────────────────────────────────────────────────────────────────┘

Jika Anda punya dataset sendiri:

1. Format dataset dalam YOLO format:
   datasets/my_dataset/
   ├── images/train/  (images)
   ├── images/val/
   ├── labels/train/  (annotations .txt)
   ├── labels/val/
   └── dataset.yaml

2. Merge dengan dataset existing:
   
   from utils.dataset_merger import merge_recaptcha_hcaptcha_datasets
   
   config = merge_recaptcha_hcaptcha_datasets(
       recaptcha_paths=['datasets/downloaded/recaptcha_roboflow'],
       hcaptcha_paths=[
           'datasets/downloaded/slide_captcha',
           'datasets/my_custom_dataset'  # Your dataset
       ],
       output_dir='datasets/merged_custom'
   )

3. Fine-tune model yang sudah trained:

   python scripts/train_yolov8.py \\
       --data datasets/merged_custom/dataset.yaml \\
       --weights runs/train/captcha_xxx/weights/best.pt \\
       --epochs 100

Fine-tuning benefits:
✅ Faster convergence (50-100 epochs vs 300)
✅ Better performance dengan less data
✅ Transfer learning dari pre-trained model
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ PERFORMANCE BENCHMARKS (GPU A100)                                            │
└──────────────────────────────────────────────────────────────────────────────┘

Training Speed:
├── YOLOv8n: ~0.5 hours (5K images, 300 epochs)
├── YOLOv8s: ~1 hour
├── YOLOv8m: ~2 hours
├── YOLOv8l: ~3 hours
└── YOLOv8x: ~5 hours

Inference Speed:
├── YOLOv8n: ~200 FPS
├── YOLOv8s: ~150 FPS
├── YOLOv8m: ~100 FPS
├── YOLOv8l: ~70 FPS
└── YOLOv8x: ~50 FPS (highest accuracy)

Memory Usage:
├── Batch 64, imgsz 1280: ~40-50 GB VRAM
├── Batch 32, imgsz 1280: ~20-25 GB VRAM
└── Batch 64, imgsz 640: ~15-20 GB VRAM

CAPTCHA Solving Success Rate (well-trained model):
├── reCAPTCHA v2: 85-95%
├── hCaptcha: 80-90%
└── Drag Puzzle: 75-90%
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ TROUBLESHOOTING                                                               │
└──────────────────────────────────────────────────────────────────────────────┘

❌ Out of Memory (OOM):
   → Reduce batch size: --batch 32
   → Reduce image size: --imgsz 640
   → Use gradient accumulation

❌ Low mAP (<0.5):
   → Train longer (--epochs 500)
   → Check data quality
   → Balance class distribution
   → Add more training data

❌ Overfitting (train mAP >> val mAP):
   → Add data augmentation (already enabled)
   → Reduce model size (yolov8l instead of yolov8x)
   → Add regularization (weight_decay)

❌ Slow training:
   → Check GPU utilization: nvidia-smi
   → Increase workers: --workers 32
   → Enable AMP (already default)
   → Use SSD for dataset cache

❌ CUDA out of memory:
   → Restart training with smaller batch
   → Close other GPU processes
   → Monitor: watch -n 1 nvidia-smi
""")

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ RESOURCES & LINKS                                                             │
└──────────────────────────────────────────────────────────────────────────────┘

Documentation:
├── YOLOv8: https://docs.ultralytics.com/
├── Roboflow: https://docs.roboflow.com/
└── PyTorch: https://pytorch.org/docs/

Datasets:
├── Roboflow Universe: https://universe.roboflow.com/
├── Kaggle Datasets: https://www.kaggle.com/datasets
└── GitHub Collections: https://github.com/topics/captcha-dataset

Research Papers:
├── Breaking reCAPTCHAv2: https://arxiv.org/abs/2409.08831
├── YOLO CAPTCHA Benchmark: https://arxiv.org/abs/2502.13740
└── YOLOv8 Paper: https://arxiv.org/abs/2305.09972

Community:
├── Ultralytics GitHub: https://github.com/ultralytics/ultralytics
├── Roboflow Forum: https://discuss.roboflow.com/
└── PyTorch Forum: https://discuss.pytorch.org/
""")

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              QUICK REFERENCE                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

# Download datasets
python scripts/download_datasets.py --merge

# Validate dataset
python utils/dataset_validator.py datasets/path/dataset.yaml

# Train model
python scripts/train_yolov8.py --data datasets/path/dataset.yaml

# Inference
python scripts/inference.py --model runs/train/xxx/weights/best.pt --source image.jpg

# Export to ONNX
python scripts/train_yolov8.py --export runs/train/xxx/weights/best.pt

# Monitor training
tensorboard --logdir runs/train

# Check GPU
nvidia-smi

# Resume training
python scripts/train_yolov8.py --resume runs/train/xxx/weights/last.pt

╔══════════════════════════════════════════════════════════════════════════════╗
║                        SELAMAT TRAINING! 🚀                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Run demo validation
print("\n🔧 Running validation checks...")
import subprocess
subprocess.run(['python', 'demo.py'])
