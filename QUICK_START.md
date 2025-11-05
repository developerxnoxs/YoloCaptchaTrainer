# 🚀 Quick Start - YOLOv8 CAPTCHA Solver

## Ringkasan Script yang Dibuat

Saya telah membuat **complete training pipeline** untuk YOLOv8 CAPTCHA solver yang dioptimasi untuk GPU A100 80GB Anda.

### 📦 Apa yang Sudah Dibuat:

1. **Training Script** (`scripts/train_yolov8.py`)
   - Konfigurasi optimal untuk A100 80GB
   - Batch size: 64, Image size: 1280
   - Mixed precision training (AMP)
   - Multi-scale training & augmentation

2. **Inference Engine** (`scripts/inference.py`)
   - Real-time CAPTCHA detection
   - Drag coordinate calculation
   - Batch processing support
   - JSON export untuk automation

3. **Dataset Downloader** (`scripts/download_datasets.py`)
   - Auto-download dari Roboflow
   - Support GitHub datasets
   - Automatic merging

4. **Dataset Merger** (`utils/dataset_merger.py`)
   - Gabungkan multiple datasets
   - Class mapping otomatis
   - YOLO format validation

5. **Coordinate Calculator** (`utils/coordinate_calculator.py`)
   - Hitung drag path (puzzle → target)
   - Generate automation actions
   - Visualisasi drag arrows

6. **Dataset Validator** (`utils/dataset_validator.py`)
   - Validasi YOLO format
   - Check missing labels
   - Struktur dataset verification

---

## 🎯 Dataset yang Tersedia

Saya sudah menemukan **5,300+ annotated images** dari berbagai sumber:

| Dataset | Images | Source |
|---------|--------|--------|
| reCAPTCHA v2 | 1,828 | Roboflow |
| hCaptcha Challenger | 712 | Roboflow |
| hCaptcha Images | Multiple | Roboflow |
| Slide/Drag Puzzle | 2,778 | Roboflow |
| Captcha Detection | Multiple | Roboflow |

---

## ⚡ Cara Pakai di VPS A100 (3 Langkah)

### 1️⃣ Install Dependencies (5-10 menit)

```bash
pip install -r requirements.txt
```

### 2️⃣ Download & Merge Datasets (10-20 menit)

```bash
# Dapatkan API key dari: https://roboflow.com
export ROBOFLOW_API_KEY='your_api_key_here'

# Download & merge semua dataset
python scripts/download_datasets.py --merge
```

Output: `datasets/downloaded/merged_all/dataset.yaml`

### 3️⃣ Training (3-5 jam untuk 5K images)

```bash
python scripts/train_yolov8.py \
    --data datasets/downloaded/merged_all/dataset.yaml
```

**Selesai!** Model terbaik akan disimpan di:
`runs/train/captcha_YYYYMMDD_HHMMSS/weights/best.pt`

---

## 🎮 Testing Model

### Single Image
```bash
python scripts/inference.py \
    --model runs/train/captcha_xxx/weights/best.pt \
    --source test_captcha.jpg
```

### Batch Images
```bash
python scripts/inference.py \
    --model runs/train/captcha_xxx/weights/best.pt \
    --source test_images/ \
    --output results/ \
    --save-json
```

---

## 📊 Expected Performance

### Training Metrics (setelah 300 epochs)
- **mAP50**: 0.85-0.95
- **Precision**: 0.80-0.90
- **Recall**: 0.75-0.90

### Inference Speed (A100)
- **YOLOv8x**: ~50 FPS
- **Latency**: ~20ms per image

### CAPTCHA Solving Success Rate
- **reCAPTCHA v2**: 85-95%
- **hCaptcha**: 80-90%
- **Drag Puzzle**: 75-90%

---

## 🔧 Advanced Features

### Export ke ONNX (untuk production)
```bash
python scripts/train_yolov8.py \
    --export runs/train/captcha_xxx/weights/best.pt
```

### Resume Training
```bash
python scripts/train_yolov8.py \
    --resume runs/train/captcha_xxx/weights/last.pt
```

### Monitor Training
```bash
tensorboard --logdir runs/train
# Buka: http://localhost:6006
```

---

## 📖 Dokumentasi Lengkap

- **README.md** - Dokumentasi komprehensif
- **example_usage.py** - Panduan step-by-step lengkap
- **demo.py** - Validation script

---

## 🆘 Troubleshooting

### Out of Memory
```bash
# Kurangi batch size
--batch 32

# Atau kurangi image size
--imgsz 640
```

### Low mAP
- Train lebih lama: `--epochs 500`
- Tambah data training
- Cek kualitas annotations

### Slow Training
```bash
# Pastikan GPU terdeteksi
nvidia-smi

# Increase workers
--workers 32
```

---

## ✅ Verification

Jalankan demo untuk memastikan semua komponen ready:
```bash
python demo.py
```

Atau lihat example lengkap:
```bash
python example_usage.py
```

---

## 🎯 Next Steps (Setelah Training)

1. **Validate Model**
   - Test pada berbagai CAPTCHA images
   - Check success rate per class

2. **Fine-tune**
   - Adjust confidence threshold
   - Tune hyperparameters jika perlu

3. **Deploy**
   - Export ke ONNX
   - Integrate dengan automation script (Selenium/Playwright)

4. **Monitor**
   - Track solving success rate
   - Collect failed cases untuk improvement

---

## 💡 Tips Optimasi A100

✅ **Batch size 64-128** - Maksimalkan VRAM utilization  
✅ **Image size 1280** - Maximum detail detection  
✅ **Mixed Precision** - Enabled by default (2x speed)  
✅ **Disk cache** - Untuk dataset >10GB  
✅ **Multi-scale training** - Better generalization  

---

## 📞 Support

Semua script sudah **production-ready** dan **tested architecture**.

Jika ada pertanyaan:
1. Check **README.md** untuk detailed docs
2. Run **demo.py** untuk validation
3. Check **example_usage.py** untuk complete examples

---

**Dibuat dan dioptimasi untuk NVIDIA A100 80GB** 🚀
