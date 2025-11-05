# 🚀 Panduan Instalasi - YOLOv8 CAPTCHA Solver untuk VPS GPU A100

Panduan lengkap untuk setup dan menjalankan training di VPS dengan GPU A100 (80GB).

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA A100 80GB (atau GPU lain dengan minimal 24GB VRAM)
- **RAM**: Minimal 32GB (recommended 64GB+)
- **Storage**: Minimal 100GB free space
- **CPU**: 8+ cores recommended

### Software
- **OS**: Ubuntu 20.04/22.04 atau Linux lainnya
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8 atau 12.x
- **NVIDIA Drivers**: Latest (525+)
- **Git**: Untuk clone repository

---

## 🎯 Quick Start (Fully Automated)

### Step 1: Clone Repository

```bash
# Clone project
git clone https://github.com/YOUR_USERNAME/yolov8-captcha-solver.git
cd yolov8-captcha-solver

# Atau upload semua file ke VPS Anda
```

### Step 2: Run Setup

```bash
# Berikan permission untuk execute
chmod +x setup.sh run_auto_training.sh

# Jalankan setup (install semua dependencies)
./setup.sh
```

Setup script akan otomatis:
- ✅ Check Python, CUDA, dan GPU
- ✅ Create virtual environment
- ✅ Install PyTorch dengan CUDA support
- ✅ Install semua dependencies (ultralytics, roboflow, dll)
- ✅ Create direktori yang diperlukan

### Step 3: Set API Key

Dapatkan Roboflow API Key:
1. Daftar di: https://roboflow.com
2. Pergi ke: https://app.roboflow.com/settings/api
3. Copy API key Anda

```bash
# Set environment variable
export ROBOFLOW_API_KEY='YOUR_API_KEY_HERE'

# ATAU simpan di ~/.bashrc agar persistent
echo "export ROBOFLOW_API_KEY='YOUR_API_KEY_HERE'" >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Run Auto Training

```bash
# Jalankan full pipeline (download + merge + train)
./run_auto_training.sh
```

**SELESAI!** Script akan otomatis:
1. ✅ Download 5+ datasets (5000+ images) dari Roboflow
2. ✅ Merge semua datasets
3. ✅ Train YOLOv8x model untuk 300 epochs
4. ✅ Save model terbaik ke `runs/train/`

---

## 🎮 Advanced Usage

### Mode 1: Full Automatic (Recommended)

```bash
# Download + Merge + Train (all in one)
python auto_train.py --api-key YOUR_KEY
```

### Mode 2: Skip Download (Gunakan Dataset yang Sudah Ada)

```bash
# Jika dataset sudah pernah didownload
python auto_train.py --skip-download
```

### Mode 3: Download Dataset Tertentu Saja

```bash
# Download hanya dataset tertentu
python auto_train.py --api-key YOUR_KEY --datasets recaptcha_roboflow slide_captcha
```

### Mode 4: Resume Training

```bash
# Jika training terinterrupt
python auto_train.py --skip-download --skip-merge
```

---

## 📁 Struktur Project

```
yolov8-captcha-solver/
├── setup.sh                    # Setup script (install dependencies)
├── run_auto_training.sh        # Main wrapper script
├── auto_train.py               # Pipeline otomatis
│
├── scripts/
│   ├── download_datasets.py    # Download datasets
│   ├── train_yolov8.py         # Training script
│   └── inference.py            # Inference/testing
│
├── utils/
│   ├── dataset_merger.py       # Merge multiple datasets
│   └── coordinate_calculator.py
│
├── datasets/
│   ├── downloaded/             # Dataset dari Roboflow (auto)
│   └── merged_all/             # Merged dataset (auto)
│
├── runs/
│   └── train/                  # Training results
│       └── captcha_YYYYMMDD_HHMMSS/
│           ├── weights/
│           │   ├── best.pt     # Model terbaik
│           │   └── last.pt     # Model terakhir
│           ├── results.csv     # Training metrics
│           └── *.png           # Training plots
│
└── requirements.txt            # Python dependencies
```

---

## 🔧 Manual Installation (Step-by-Step)

Jika Anda ingin install manual:

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Install Roboflow
pip install roboflow
```

### 2. Download Datasets

```bash
# List available datasets
python scripts/download_datasets.py --list

# Download all datasets
python scripts/download_datasets.py \
    --api-key YOUR_API_KEY \
    --output datasets/downloaded \
    --merge

# Download specific dataset
python scripts/download_datasets.py \
    --api-key YOUR_API_KEY \
    --dataset recaptcha_roboflow
```

### 3. Merge Datasets

```bash
# Merge akan otomatis dilakukan jika pakai flag --merge
# Atau manual dengan auto_train.py
```

### 4. Train Model

```bash
python scripts/train_yolov8.py \
    --data datasets/merged_all/dataset.yaml \
    --project runs/train \
    --name my_training
```

---

## 📊 Dataset Information

Script akan otomatis mendownload dataset berikut:

| Dataset | Images | Source | Classes |
|---------|--------|--------|---------|
| reCAPTCHA v2 | 1,828 | Roboflow | reCAPTCHA objects |
| hCaptcha Challenger | 712 | Roboflow | Elephants, animals |
| hCaptcha Images | Multiple | Roboflow | Various objects |
| Slide/Drag Puzzle | 2,778 | Roboflow | Puzzle pieces |
| Captcha Detection | Multiple | Roboflow | jigsaw, puzzle, target |

**Total**: 5000+ annotated images

---

## ⚙️ Training Configuration

Default configuration untuk A100 80GB:

```yaml
Model: YOLOv8x (largest)
Input Size: 1280x1280
Batch Size: 64
Epochs: 300
Workers: 16
Optimizer: AdamW
Mixed Precision: Enabled (AMP)
Multi-Scale: Enabled
Cache: Disk
```

**Estimated Training Time**: 3-5 hours untuk 5000 images

---

## 🧪 Testing Model

```bash
# Inference pada single image
python scripts/inference.py \
    --model runs/train/captcha_*/weights/best.pt \
    --image path/to/captcha.jpg \
    --output output/

# Batch inference
python scripts/inference.py \
    --model runs/train/captcha_*/weights/best.pt \
    --source path/to/images/ \
    --output output/
```

---

## 📈 Monitoring Training

### Secara Real-time

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs (dalam terminal terpisah)
tail -f runs/train/captcha_*/results.csv
```

### Setelah Training

```bash
# Lihat training plots
ls runs/train/captcha_*/

# Files yang akan dibuat:
# - confusion_matrix.png
# - F1_curve.png
# - P_curve.png
# - R_curve.png
# - PR_curve.png
# - results.png
```

---

## 🐛 Troubleshooting

### 1. CUDA Out of Memory

```bash
# Kurangi batch size di scripts/train_yolov8.py
# Line 40: 'batch_size': 32,  # Dari 64 ke 32
```

### 2. Roboflow Download Failed

```bash
# Check API key
echo $ROBOFLOW_API_KEY

# Install ulang roboflow
pip install --upgrade roboflow
```

### 3. Training Sangat Lambat

```bash
# Check apakah CUDA terdeteksi
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU usage
nvidia-smi
```

### 4. Dataset Tidak Ditemukan

```bash
# Check struktur folder
ls -la datasets/downloaded/
ls -la datasets/merged_all/

# Re-run merge
python auto_train.py --skip-download
```

---

## 💾 Backup & Export

### Backup Model

```bash
# Backup best model
cp runs/train/captcha_*/weights/best.pt models/captcha_best_$(date +%Y%m%d).pt

# Download dari VPS (dari local machine)
scp user@vps:/path/to/runs/train/captcha_*/weights/best.pt ./
```

### Export ke ONNX

```bash
python scripts/train_yolov8.py --export runs/train/captcha_*/weights/best.pt
```

---

## 📞 Support

Jika ada masalah:

1. Check logs di `runs/train/captcha_*/`
2. Verify GPU dengan `nvidia-smi`
3. Check Python dependencies dengan `pip list`
4. Re-run setup: `./setup.sh`

---

## ⏱️ Estimated Timeline

Untuk VPS dengan A100 80GB:

1. ✅ Setup: ~5-10 minutes
2. ✅ Download datasets: ~10-20 minutes (tergantung internet)
3. ✅ Merge datasets: ~5 minutes
4. ✅ Training (300 epochs): ~3-5 hours
5. ✅ **Total**: ~4-6 hours

---

## 🎯 Expected Results

Setelah training selesai, Anda akan mendapatkan:

- ✅ Model file: `best.pt` (~280MB)
- ✅ mAP50: ~0.85-0.95 (tergantung dataset)
- ✅ Precision: ~0.80-0.90
- ✅ Recall: ~0.75-0.85
- ✅ Training plots & metrics

---

## 🚀 Next Steps

Setelah training selesai:

1. Test model dengan inference script
2. Deploy ke production
3. Integrate dengan CAPTCHA solving pipeline
4. Monitor & retrain dengan data baru

**Happy Training! 🎉**
