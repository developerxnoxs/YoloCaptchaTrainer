# 🚀 YOLOv8 CAPTCHA Solver - Setup Otomatis untuk VPS GPU A100

> **Proyek ini sepenuhnya otomatis!** Cukup jalankan satu perintah dan biarkan script melakukan semuanya.

## ✨ Fitur Utama

- ✅ **Setup otomatis** - Install semua dependencies dengan 1 perintah
- ✅ **Download otomatis** - Download 5000+ annotated images dari Roboflow
- ✅ **Merge otomatis** - Gabungkan semua dataset secara otomatis
- ✅ **Training otomatis** - Train YOLOv8x untuk 300 epochs
- ✅ **Optimized untuk A100 80GB** - Batch size 64, image size 1280
- ✅ **Zero configuration** - Tidak perlu edit config file!

## 🎯 Quick Start (3 Langkah)

### 1️⃣ Setup (5 menit)

```bash
# Clone atau upload project ke VPS
cd yolov8-captcha-solver

# Berikan permission
chmod +x setup.sh run_auto_training.sh

# Jalankan setup
./setup.sh
```

### 2️⃣ Set API Key

```bash
# Dapatkan dari: https://app.roboflow.com/settings/api
export ROBOFLOW_API_KEY='your_api_key_here'
```

### 3️⃣ Run Training

```bash
# Jalankan semua (download + merge + train)
./run_auto_training.sh
```

**SELESAI!** Training akan berjalan otomatis 3-5 jam. Anda bisa tutup terminal (gunakan `screen` atau `tmux`).

## 📊 Apa Yang Akan Terjadi?

Script akan otomatis:

1. ✅ **Download datasets** (10-20 menit)
   - reCAPTCHA v2: 1,828 images
   - hCaptcha Challenger: 712 images
   - hCaptcha Images: Multiple
   - Slide/Drag Puzzle: 2,778 images
   - Captcha Detection: Multiple
   - **Total: 5000+ images**

2. ✅ **Merge datasets** (5 menit)
   - Gabungkan semua dataset
   - Normalize class labels
   - Split train/val/test (70/20/10)

3. ✅ **Train YOLOv8x** (3-5 jam)
   - Model: YOLOv8x (largest)
   - Batch size: 64
   - Image size: 1280
   - Epochs: 300
   - Mixed precision (AMP)
   - Multi-scale training

## 📁 Hasil Training

Semua hasil akan tersimpan di:

```
runs/train/captcha_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt          # Model terbaik (gunakan ini!)
│   └── last.pt          # Model terakhir
├── results.csv          # Metrics (mAP, precision, recall)
├── results.png          # Training curves
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
└── R_curve.png
```

## 🎮 Opsi Lanjutan

### Mode 1: Full Auto (Recommended)
```bash
./run_auto_training.sh
```

### Mode 2: Skip Download (Dataset Sudah Ada)
```bash
python auto_train.py --skip-download
```

### Mode 3: Download Dataset Tertentu
```bash
python auto_train.py --api-key YOUR_KEY --datasets recaptcha_roboflow slide_captcha
```

### Mode 4: Resume Training yang Terinterrupt
```bash
python auto_train.py --skip-download --skip-merge
```

## 🧪 Testing Model

```bash
# Test pada single image
python scripts/inference.py \
    --model runs/train/captcha_*/weights/best.pt \
    --image test_image.jpg \
    --output output/

# Test pada folder images
python scripts/inference.py \
    --model runs/train/captcha_*/weights/best.pt \
    --source test_images/ \
    --output output/
```

## 📈 Monitor Training

### Real-time Monitoring

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Monitor logs
tail -f runs/train/captcha_*/results.csv
```

### Gunakan Screen/Tmux (Recommended)

```bash
# Install screen
sudo apt install screen

# Start screen session
screen -S yolo_training

# Jalankan training
./run_auto_training.sh

# Detach: Ctrl+A lalu D
# Re-attach: screen -r yolo_training
```

## 🐛 Troubleshooting

### CUDA Out of Memory?
Edit `scripts/train_yolov8.py` line 40:
```python
'batch_size': 32,  # Kurangi dari 64 ke 32
```

### Dataset Download Gagal?
```bash
# Check API key
echo $ROBOFLOW_API_KEY

# Install ulang roboflow
source venv/bin/activate
pip install --upgrade roboflow
```

### Training Lambat?
```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU usage
nvidia-smi
```

## 📊 Expected Performance

Setelah training selesai:

- **mAP50**: 0.85 - 0.95
- **Precision**: 0.80 - 0.90
- **Recall**: 0.75 - 0.85
- **Model Size**: ~280MB
- **Inference Speed**: ~5-10ms per image (A100)

## 💾 Backup Model

```bash
# Backup best model
cp runs/train/captcha_*/weights/best.pt models/captcha_$(date +%Y%m%d).pt

# Download ke local (dari komputer lokal)
scp user@vps:/path/to/runs/train/captcha_*/weights/best.pt ./
```

## 📞 Dokumentasi Lengkap

Lihat [INSTALL.md](INSTALL.md) untuk:
- Instalasi manual step-by-step
- Advanced configuration
- Troubleshooting lengkap
- Export ke ONNX/TorchScript

## ⏱️ Timeline

Untuk VPS dengan A100 80GB:

| Step | Waktu | Status |
|------|-------|--------|
| Setup | 5-10 menit | ✅ Otomatis |
| Download datasets | 10-20 menit | ✅ Otomatis |
| Merge datasets | 5 menit | ✅ Otomatis |
| Training (300 epochs) | 3-5 jam | ✅ Otomatis |
| **TOTAL** | **~4-6 jam** | ✅ Otomatis |

## 🎯 Next Steps

Setelah training selesai:

1. ✅ Test model dengan inference script
2. ✅ Deploy ke production
3. ✅ Integrate dengan CAPTCHA solving system
4. ✅ Monitor performance & retrain dengan data baru

## 📋 Requirements

- **GPU**: NVIDIA A100 80GB (atau GPU lain dengan 24GB+ VRAM)
- **RAM**: 32GB+ (recommended 64GB)
- **Storage**: 100GB+ free space
- **OS**: Ubuntu 20.04/22.04
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8 atau 12.x

## 🔐 Security Note

Script ini **TIDAK pernah** menyimpan atau expose API key Anda:
- API key hanya digunakan melalui environment variable
- Tidak disimpan di file config
- Tidak di-print ke console/logs
- Aman untuk production use

## 📝 File Penting

```
setup.sh               → Install dependencies
run_auto_training.sh   → Main script (jalankan ini!)
auto_train.py          → Pipeline otomatis
INSTALL.md             → Dokumentasi lengkap
README_INDONESIAN.md   → File ini
```

## 🎉 Selamat Training!

Jika ada pertanyaan atau masalah:
1. Check [INSTALL.md](INSTALL.md) untuk troubleshooting
2. Verify GPU dengan `nvidia-smi`
3. Check logs di `runs/train/`

**Happy Training! 🚀**
