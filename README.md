# YOLOv8 CAPTCHA Solver - reCAPTCHA v2 & hCaptcha

Script Python lengkap untuk melatih model YOLOv8 pada GPU A100 80GB untuk mendeteksi dan menyelesaikan reCAPTCHA v2, hCaptcha, dan drag-based CAPTCHA challenges.

## 🎯 Fitur Utama

- ✅ **Training YOLOv8** dengan konfigurasi optimal untuk GPU A100 80GB
- ✅ **Dataset Merger** untuk menggabungkan multiple datasets (reCAPTCHA v2, hCaptcha, drag puzzle)
- ✅ **Coordinate Calculator** untuk menghitung posisi drag dari puzzle piece ke target zone
- ✅ **Inference Engine** untuk deteksi real-time dan solving CAPTCHA otomatis
- ✅ **Multi-class Detection** untuk berbagai objek CAPTCHA
- ✅ **Export Model** ke ONNX dan TorchScript untuk deployment
- ✅ **Visualisasi** hasil deteksi dengan bounding box dan drag paths

## 📁 Struktur Project

```
.
├── scripts/
│   ├── train_yolov8.py          # Script training utama
│   ├── inference.py              # Inference dan solving engine
│   └── download_datasets.py      # Download datasets otomatis
├── utils/
│   ├── dataset_merger.py         # Merge multiple datasets
│   ├── coordinate_calculator.py  # Hitung koordinat drag
│   └── dataset_validator.py      # Validasi dataset format
├── models/                       # Trained models (.pt, .onnx)
├── datasets/                     # Dataset storage
├── output/                       # Inference results
├── logs/                         # Training logs
├── requirements.txt              # Python dependencies
└── demo.py                       # Demo dan validasi
```

## 🚀 Quick Start di VPS

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies utama:
- ultralytics (YOLOv8)
- torch + torchvision
- opencv-python
- numpy, pandas, matplotlib
- roboflow (untuk download dataset)

### 2. Download Datasets

#### Option A: Otomatis dari Roboflow (Recommended)

```bash
# List available datasets
python scripts/download_datasets.py --list

# Download semua dataset (perlu API key)
export ROBOFLOW_API_KEY='your_api_key_here'
python scripts/download_datasets.py --merge

# Atau download spesifik dataset
python scripts/download_datasets.py --dataset recaptcha_roboflow --api-key your_key
```

**Cara dapat Roboflow API Key:**
1. Buat akun di https://roboflow.com
2. Ambil API key dari https://app.roboflow.com/settings/api
3. Free tier: 10,000 API calls/month

#### Option B: Manual Download

**Dataset yang tersedia:**

| Dataset | Images | Source | Classes |
|---------|--------|--------|---------|
| reCAPTCHA v2 (Roboflow) | 1,828 | [Link](https://universe.roboflow.com/my-workspace-4p8ud/recaptcha-v2) | reCAPTCHA objects |
| hCaptcha Challenger | 712 | [Link](https://universe.roboflow.com/qin2dim/hcaptcha-challenger) | Elephants, animals |
| hCaptcha Images | Multiple | [Link](https://universe.roboflow.com/stopthecap/hcaptcha-images) | Various objects |
| Slide/Drag Puzzle | 2,778 | [Link](https://universe.roboflow.com/captcha-lwpyk/slide_captcha) | Puzzle pieces |
| Captcha Detection | Multiple | [Link](https://universe.roboflow.com/captcha-detection/captcha-detection-smkks) | jigsaw-piece, puzzle, target |

Download manual dan extract ke folder `datasets/`.

### 3. Merge Datasets (Jika Punya Multiple Datasets)

```python
from utils.dataset_merger import merge_recaptcha_hcaptcha_datasets

config_path = merge_recaptcha_hcaptcha_datasets(
    recaptcha_paths=[
        'datasets/recaptcha_roboflow',
        'datasets/recaptcha_dataset2'
    ],
    hcaptcha_paths=[
        'datasets/hcaptcha_challenger',
        'datasets/slide_captcha',
        'datasets/captcha_detection'
    ],
    output_dir='datasets/merged_captcha'
)

print(f'Dataset merged: {config_path}')
# Output: datasets/merged_captcha/dataset.yaml
```

### 4. Validasi Dataset

```bash
python utils/dataset_validator.py datasets/merged_captcha/dataset.yaml
```

### 5. Training di GPU A100

```bash
# Training dengan konfigurasi default (optimal untuk A100 80GB)
python scripts/train_yolov8.py --data datasets/merged_captcha/dataset.yaml

# Custom configuration
python scripts/train_yolov8.py \
    --data datasets/merged_captcha/dataset.yaml \
    --project runs/train \
    --name captcha_v1
```

**Konfigurasi Training (Default untuk A100 80GB):**
- Model: YOLOv8x (largest)
- Batch size: 64
- Image size: 1280
- Epochs: 300
- Workers: 16
- Mixed Precision (AMP): Enabled
- Multi-scale training: Enabled
- Cache: disk

**Training akan menghasilkan:**
- `runs/train/captcha_xxx/weights/best.pt` - Best model
- `runs/train/captcha_xxx/weights/last.pt` - Last epoch
- `runs/train/captcha_xxx/results.csv` - Metrics
- `runs/train/captcha_xxx/` - Plots dan visualisasi

### 6. Resume Training

```bash
python scripts/train_yolov8.py --resume runs/train/captcha_xxx/weights/last.pt
```

### 7. Inference / Solving CAPTCHA

```bash
# Single image
python scripts/inference.py \
    --model runs/train/captcha_xxx/weights/best.pt \
    --source test_image.jpg \
    --output output/results

# Batch inference (directory)
python scripts/inference.py \
    --model runs/train/captcha_xxx/weights/best.pt \
    --source test_images/ \
    --output output/batch_results \
    --save-json

# Custom thresholds
python scripts/inference.py \
    --model runs/train/captcha_xxx/weights/best.pt \
    --source image.jpg \
    --conf 0.35 \
    --iou 0.5
```

**Output:**
- Annotated images dengan bounding boxes
- Drag path visualization (puzzle → target)
- JSON file dengan koordinat dan actions
- Drag actions untuk automation (mousedown, mousemove, mouseup)

### 8. Export Model untuk Production

```bash
# Export ke ONNX dan TorchScript
python scripts/train_yolov8.py --export runs/train/captcha_xxx/weights/best.pt
```

Format export:
- ONNX (`.onnx`) - Universal format
- TorchScript (`.torchscript`) - PyTorch deployment
- TensorRT (`.engine`) - NVIDIA optimization
- CoreML (`.mlmodel`) - iOS/macOS

## 🎓 Penggunaan Advanced

### Custom Training Configuration

Buat file `config.yaml`:

```yaml
model:
  architecture: yolov8x
  pretrained: true
  input_size: 1280

training:
  epochs: 300
  batch_size: 64
  workers: 16
  patience: 50
  device: '0'
  amp: true

hyperparameters:
  lr0: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  mosaic: 1.0
  mixup: 0.15
```

Training dengan config:
```bash
python scripts/train_yolov8.py --data dataset.yaml --config config.yaml
```

### Programmatic API

```python
from scripts.train_yolov8 import YOLOv8Trainer
from scripts.inference import CaptchaSolver

# Training
trainer = YOLOv8Trainer()
results = trainer.train('datasets/merged_captcha/dataset.yaml')

# Inference
solver = CaptchaSolver('runs/train/captcha_xxx/weights/best.pt')
result = solver.solve_captcha('test_image.jpg')

if result['solved']:
    print(f"✅ CAPTCHA solved!")
    print(f"Actions: {result['actions']}")
    print(f"Distance: {result['drag_info']['distance']:.1f}px")
else:
    print(f"❌ Failed: {result['message']}")
```

### Coordinate Calculator

```python
from utils.coordinate_calculator import CoordinateCalculator

calc = CoordinateCalculator()

# Detections dari YOLO
detections = [
    {'class': 0, 'conf': 0.95, 'bbox': [100, 150, 200, 250]},  # puzzle
    {'class': 1, 'conf': 0.92, 'bbox': [400, 180, 500, 280]},  # target
]

result = calc.calculate_drag_coordinates(
    detections,
    image_width=800,
    image_height=600,
    class_names=['puzzle_piece', 'drop_zone']
)

# Result contains:
# - puzzle center coordinates
# - target center coordinates
# - drag vector (dx, dy)
# - distance and angle
# - step-by-step actions for automation
```

## 💡 Tips Optimasi

### GPU A100 80GB

1. **Batch Size**: Mulai dengan 64, bisa dinaikkan sampai 128
2. **Image Size**: 1280 untuk detail maksimal, 640 untuk speed
3. **Workers**: 16-32 (sesuaikan dengan CPU cores)
4. **Cache**: Gunakan `disk` untuk dataset >10GB, `ram` untuk dataset kecil

### Multi-GPU Training

```python
# Otomatis detect semua GPU
model.train(data='dataset.yaml', device='0,1,2,3')  # 4 GPUs
```

### Monitoring Training

```bash
# TensorBoard
tensorboard --logdir runs/train

# Watch metrics real-time
watch -n 5 tail -20 runs/train/captcha_xxx/results.csv
```

## 📊 Expected Performance

### Training Metrics (setelah 300 epochs)

- **mAP50**: 0.85-0.95 (tergantung dataset quality)
- **mAP50-95**: 0.70-0.85
- **Precision**: 0.80-0.90
- **Recall**: 0.75-0.90

### Inference Speed (A100)

- **YOLOv8n**: ~200 FPS
- **YOLOv8s**: ~150 FPS
- **YOLOv8m**: ~100 FPS
- **YOLOv8l**: ~70 FPS
- **YOLOv8x**: ~50 FPS

### CAPTCHA Solving Success Rate

- **reCAPTCHA v2**: 85-95% (dengan model terlatih baik)
- **hCaptcha**: 80-90%
- **Drag Puzzle**: 75-90%

## 🔧 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python scripts/train_yolov8.py --data dataset.yaml --batch 32

# Reduce image size
python scripts/train_yolov8.py --data dataset.yaml --imgsz 640
```

### Slow Training

- Enable AMP (mixed precision): Enabled by default
- Increase workers: `--workers 32`
- Use SSD untuk dataset cache
- Enable multi-scale: Enabled by default

### Low mAP

- Increase training epochs: `--epochs 500`
- Data augmentation (mosaic, mixup): Enabled by default
- Collect more training data
- Balance class distribution
- Tune hyperparameters

## 📝 Dataset Format (YOLO)

Struktur dataset yang benar:

```
datasets/your_dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   └── img2.txt
│   ├── val/
│   └── test/
└── dataset.yaml
```

**dataset.yaml:**
```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 8  # number of classes
names: 
  - puzzle_piece
  - drop_zone
  - car
  - traffic_light
  - bus
  - bicycle
  - crosswalk
  - motorcycle
```

**Label format (img1.txt):**
```
0 0.516 0.380 0.157 0.206
1 0.708 0.421 0.142 0.189
```
Format: `class_id x_center y_center width height` (normalized 0-1)

## 🎯 Use Cases

### 1. CAPTCHA Research
- Analisis keamanan CAPTCHA systems
- Benchmark CAPTCHA difficulty
- Educational purposes

### 2. Automation Testing
- QA automation untuk website testing
- Integration testing dengan CAPTCHA

### 3. Accessibility Tools
- Assistive technology untuk users dengan disabilities

## ⚠️ Legal & Ethical Notice

Script ini dibuat untuk:
- ✅ Research dan educational purposes
- ✅ Authorized testing pada sistem Anda sendiri
- ✅ Accessibility tool development

**Jangan gunakan untuk:**
- ❌ Bypass CAPTCHA tanpa permission
- ❌ Automated abuse atau spam
- ❌ Melanggar Terms of Service website

Selalu patuhi terms of service dan hukum yang berlaku.

## 📖 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [Research: Breaking reCAPTCHAv2](https://arxiv.org/abs/2409.08831)
- [YOLO for CAPTCHA Recognition](https://arxiv.org/abs/2502.13740)

## 🤝 Contributing

Contributions welcome! Untuk improvement:
- Better dataset annotations
- Training optimization
- Additional CAPTCHA types support
- Bug fixes

## 📄 License

Script ini untuk educational dan research purposes. Dataset licenses bervariasi - check individual dataset sources.

## 💬 Support

Jika ada pertanyaan atau issues:
1. Check dokumentasi di atas
2. Validasi dataset dengan `dataset_validator.py`
3. Run demo dengan `python demo.py`
4. Check training logs di `runs/train/`

---

**Created for GPU A100 80GB - Optimized for maximum performance** 🚀
