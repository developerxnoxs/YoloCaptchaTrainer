# 🎯 MULAI DI SINI - YOLOv8 CAPTCHA Solver

> **Proyek ini siap dijalankan di VPS Anda!** Semua sudah otomatis.

## 📦 Isi Proyek

Proyek ini adalah sistem training YOLOv8 yang **sepenuhnya otomatis** untuk CAPTCHA solver. Anda hanya perlu:

1. Upload/clone ke VPS Anda
2. Jalankan 2 perintah
3. Tunggu 4-6 jam
4. Model siap digunakan!

## 🚀 Cara Menggunakan

### Step 1: Transfer ke VPS

```bash
# Option A: Clone dari repository
git clone https://github.com/YOUR_REPO/yolov8-captcha-solver.git
cd yolov8-captcha-solver

# Option B: Upload manual
# Upload semua file ke VPS Anda via SCP/SFTP
```

### Step 2: Jalankan Setup

```bash
# Berikan permission
chmod +x setup.sh run_auto_training.sh

# Jalankan setup (install dependencies)
./setup.sh
```

### Step 3: Set API Key & Run

```bash
# Set Roboflow API key
export ROBOFLOW_API_KEY='your_api_key_here'

# Jalankan training otomatis
./run_auto_training.sh
```

**SELESAI!** Script akan berjalan otomatis dan training akan selesai dalam 4-6 jam.

## 📁 File-File Penting

| File | Fungsi |
|------|--------|
| `setup.sh` | Install semua dependencies |
| `run_auto_training.sh` | **Jalankan ini untuk training!** |
| `auto_train.py` | Pipeline otomatis (dipanggil oleh run_auto_training.sh) |
| `INSTALL.md` | Dokumentasi lengkap (Bahasa Inggris) |
| `README_INDONESIAN.md` | Panduan lengkap (Bahasa Indonesia) |
| `START_HERE.md` | **File ini** |

## 📚 Dokumentasi

- **Quick Start** → Baca `README_INDONESIAN.md`
- **Panduan Lengkap** → Baca `INSTALL.md`
- **Manual Step-by-Step** → Lihat bagian "Manual Installation" di `INSTALL.md`

## 🎯 Apa yang Akan Terjadi?

Script akan otomatis:

```
1. Download datasets (10-20 menit)
   ↓ 5000+ annotated images dari Roboflow
   
2. Merge datasets (5 menit)
   ↓ Gabungkan semua dataset
   
3. Train YOLOv8x (3-5 jam)
   ↓ 300 epochs di A100 GPU
   
4. SELESAI!
   ↓ Model tersimpan di runs/train/
```

## 💡 Tips

### Gunakan Screen/Tmux

Training butuh 4-6 jam, gunakan screen agar bisa detach:

```bash
# Install screen
sudo apt install screen

# Start session
screen -S training

# Jalankan training
./run_auto_training.sh

# Detach: tekan Ctrl+A lalu D
# Re-attach nanti: screen -r training
```

### Monitor Progress

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Monitor logs
tail -f runs/train/captcha_*/results.csv
```

## 📊 Hasil yang Diharapkan

Setelah training selesai:

- ✅ Model: `runs/train/captcha_*/weights/best.pt`
- ✅ mAP50: 0.85 - 0.95
- ✅ Precision: 0.80 - 0.90
- ✅ Recall: 0.75 - 0.85
- ✅ Model size: ~280MB
- ✅ Inference speed: 5-10ms per image (A100)

## 🐛 Ada Masalah?

1. Check `README_INDONESIAN.md` → Bagian Troubleshooting
2. Check `INSTALL.md` → Bagian Troubleshooting
3. Verify GPU: `nvidia-smi`
4. Check logs: `ls runs/train/`

## 📞 Struktur Project

```
yolov8-captcha-solver/
│
├── setup.sh                 ← Install dependencies
├── run_auto_training.sh     ← JALANKAN INI!
├── auto_train.py            ← Pipeline otomatis
│
├── scripts/
│   ├── download_datasets.py
│   ├── train_yolov8.py
│   └── inference.py
│
├── utils/
│   ├── dataset_merger.py
│   └── coordinate_calculator.py
│
├── datasets/               ← Auto-created
│   ├── downloaded/         ← Dataset dari Roboflow
│   └── merged_all/         ← Merged dataset
│
├── runs/                   ← Auto-created
│   └── train/              ← Training results
│       └── captcha_*/
│           └── weights/
│               └── best.pt ← MODEL ANDA!
│
└── docs/
    ├── README_INDONESIAN.md
    ├── INSTALL.md
    └── START_HERE.md
```

## ✅ Checklist

Sebelum menjalankan, pastikan:

- [ ] VPS memiliki GPU A100 (atau GPU lain dengan 24GB+ VRAM)
- [ ] Ubuntu 20.04/22.04 terinstall
- [ ] Python 3.8+ tersedia
- [ ] CUDA & NVIDIA drivers terinstall
- [ ] Internet connection untuk download datasets
- [ ] Minimal 100GB storage kosong
- [ ] Punya Roboflow API key (gratis di https://roboflow.com)

## 🎉 Siap!

Sekarang Anda siap untuk training. Jalankan:

```bash
./run_auto_training.sh
```

Dan tunggu 4-6 jam. Model Anda akan siap! 🚀

---

**Pertanyaan?** Baca dokumentasi lengkap di:
- `README_INDONESIAN.md` (Bahasa Indonesia)
- `INSTALL.md` (English)
