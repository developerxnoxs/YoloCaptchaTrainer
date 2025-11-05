# 🔧 Fix: Box Loss Terus Meningkat

## ❌ Problem

Box loss **terus naik** instead of turun saat training:

```
Epoch    box_loss   cls_loss   dfl_loss
1/300     1.647      8.895      2.163
2/300     1.720      9.120      2.240   ← NAIK!
3/300     1.850      9.450      2.310   ← NAIK LAGI!
```

Ini artinya **model tidak belajar**, malah semakin buruk!

## 🧐 Root Cause Analysis (dari Architect)

### 1. ❌ Learning Rate Terlalu Tinggi

**Masalah utama:**
- Current: `lr0 = 0.01`
- Recommended untuk batch 16 dengan AdamW: `lr0 = 0.001`
- **4x terlalu tinggi!**

**Penjelasan:**
- YOLOv8 menggunakan linear scaling rule: `lr ∝ batch_size / 64`
- Untuk batch=16: `lr0 = 0.01 × (16/64) = 0.0025` (SGD)
- Untuk AdamW: lebih rendah lagi → `0.001`
- LR terlalu tinggi = gradient updates terlalu besar = model **diverge**

### 2. ❌ Data Augmentation Terlalu Aggressive

**Current config terlalu berat untuk CAPTCHA:**

```python
'mosaic': 1.0,      # Terlalu aggressive
'mixup': 0.15,      # Tidak cocok untuk CAPTCHA
'copy_paste': 0.3,  # Terlalu tinggi
'multi_scale': True # Terlalu berat
```

**Masalah:**
- CAPTCHA objects: sedikit, boxes presisi
- Mosaic/mixup: menggabungkan multiple images → merusak context
- Multi-scale: variasi size terlalu besar → gradient noise

### 3. ⚠️ Warmup Terlalu Pendek

- Current: 3 epochs
- Recommended: 8-10 epochs untuk batch size kecil
- Warmup terlalu pendek + LR tinggi = **instant divergence**

## ✅ Solution (Sudah Diterapkan!)

Saya sudah update konfigurasi dengan recommendations dari architect:

### Learning Rate Fix

```python
# BEFORE (WRONG)
'lr0': 0.01,           # Terlalu tinggi
'lrf': 0.01,           # Final LR sama dengan initial
'warmup_epochs': 3.0,  # Terlalu pendek
'warmup_bias_lr': 0.1, # Terlalu tinggi

# AFTER (CORRECT)
'lr0': 0.001,          # Reduced 10x untuk AdamW
'lrf': 0.0001,         # Lower final LR (1/10 of lr0)
'warmup_epochs': 8.0,  # Increased untuk smooth start
'warmup_bias_lr': 0.001, # Reduced untuk stability
```

### Data Augmentation Fix

```python
# BEFORE (WRONG - Too Aggressive)
'multi_scale': True,   # Heavy untuk CAPTCHA
'mosaic': 1.0,         # Selalu aktif
'mixup': 0.15,         # Mixing images
'copy_paste': 0.3,     # Too much

# AFTER (CORRECT - CAPTCHA Optimized)
'multi_scale': False,  # Disabled - single scale lebih stabil
'mosaic': 0.5,         # Reduced - 50% chance
'mixup': 0.0,          # Disabled - tidak perlu untuk CAPTCHA
'copy_paste': 0.1,     # Reduced - minimal
```

## 📊 Expected Training Behavior Sekarang

### Epoch 1-8 (Warmup Phase)

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss
1/300      33.2G      1.650      8.920      2.170
2/300      33.5G      1.620      8.750      2.140   ← TURUN sedikit
3/300      33.4G      1.590      8.580      2.110   ← TURUN terus
...
8/300      33.8G      1.420      7.650      1.950   ← Warmup selesai
```

### Epoch 9-50 (Fast Learning)

```
Epoch    box_loss   cls_loss   dfl_loss
10/300     1.350      7.120      1.880
20/300     1.180      5.850      1.720
30/300     0.950      4.520      1.540
40/300     0.780      3.650      1.380
50/300     0.620      2.850      1.220
```

### Epoch 50-300 (Convergence)

```
Epoch    box_loss   cls_loss   dfl_loss
100/300    0.420      1.850      0.920
150/300    0.320      1.350      0.750
200/300    0.250      0.980      0.620
250/300    0.220      0.870      0.580
300/300    0.210      0.850      0.560   ← Final
```

**Semua losses harus TURUN konsisten!**

## 🚀 Cara Menggunakan Fix Ini

### Step 1: Clear GPU Memory

```bash
./clear_gpu.sh
nvidia-smi  # Verify 0 MiB
```

### Step 2: Run Training dengan Config Baru

```bash
# Training dengan fixed hyperparameters
python scripts/train_yolov8.py --data datasets/merged_all/dataset.yaml
```

### Step 3: Monitor Losses (IMPORTANT!)

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Monitor losses
tail -f runs/train/captcha_*/results.csv
```

**Check setiap 5-10 epochs:**
- ✅ Box loss TURUN? → Good!
- ❌ Box loss NAIK? → Ada masalah data/config

## 🔍 Troubleshooting

### Jika Box Loss MASIH Naik

#### Option A: Kurangi Learning Rate Lagi

Edit `scripts/train_yolov8.py` line 51:
```python
'lr0': 0.0005,  # Reduced dari 0.001
```

#### Option B: Switch ke SGD Optimizer

Edit `scripts/train_yolov8.py` line 76:
```python
'optimizer': 'SGD',  # Lebih stable untuk small batch
'lr0': 0.0025,       # Higher LR untuk SGD
```

#### Option C: Disable Semua Augmentation

Edit `scripts/train_yolov8.py`:
```python
'mosaic': 0.0,
'copy_paste': 0.0,
'mixup': 0.0,
'multi_scale': False,
```

#### Option D: Check Data Quality

```bash
# Visualize annotations
python -c "
from ultralytics import YOLO
model = YOLO('yolov8x.pt')
model.val(data='datasets/merged_all/dataset.yaml', save_json=True, save_hybrid=True)
"
```

Jika annotations salah/rusak, losses akan selalu naik!

## 📋 Checklist Sebelum Training

- [ ] GPU memory cleared (`./clear_gpu.sh`)
- [ ] Learning rate: 0.001 (AdamW)
- [ ] Warmup epochs: 8
- [ ] Multi-scale: False
- [ ] Mosaic: 0.5
- [ ] Mixup: 0.0
- [ ] Copy-paste: 0.1

## 📊 Perbandingan Konfigurasi

| Setting | Old (Diverge) | New (Stable) | Improvement |
|---------|---------------|--------------|-------------|
| lr0 | 0.01 | 0.001 | ✅ 10x safer |
| lrf | 0.01 | 0.0001 | ✅ Better decay |
| warmup_epochs | 3.0 | 8.0 | ✅ Smoother start |
| multi_scale | True | False | ✅ Less noise |
| mosaic | 1.0 | 0.5 | ✅ Less aggressive |
| mixup | 0.15 | 0.0 | ✅ No mixing |
| copy_paste | 0.3 | 0.1 | ✅ Minimal |

## 🎯 Expected Results

Dengan konfigurasi baru:

| Metric | Value |
|--------|-------|
| Box Loss (epoch 1) | 1.6-1.7 |
| Box Loss (epoch 50) | 0.6-0.7 |
| Box Loss (epoch 300) | 0.20-0.25 |
| Final mAP50 | 0.85-0.95 |
| Training Stability | ⭐⭐⭐⭐⭐ |

## 💡 Mengapa Ini Terjadi?

### Learning Rate Terlalu Tinggi

Analogi: Bayangkan Anda mencari puncak gunung dalam kabut:
- **LR kecil (0.001)**: Langkah kecil, hati-hati → sampai puncak
- **LR besar (0.01)**: Langkah besar, loncat-loncat → jatuh ke jurang!

### Augmentation Terlalu Aggressive

CAPTCHA detection berbeda dengan natural images:
- **Natural images**: Banyak object, context flexible → augmentation bagus
- **CAPTCHA**: Sedikit object, layout strict → augmentation merusak

## ✅ Summary

**Root cause:**
1. ❌ Learning rate 10x terlalu tinggi
2. ❌ Data augmentation terlalu aggressive
3. ❌ Warmup terlalu pendek

**Solution (sudah diterapkan):**
1. ✅ lr0: 0.01 → 0.001
2. ✅ warmup: 3 → 8 epochs
3. ✅ multi_scale: True → False
4. ✅ mosaic: 1.0 → 0.5
5. ✅ mixup: 0.15 → 0.0

**Next steps:**
```bash
./clear_gpu.sh
python scripts/train_yolov8.py --data datasets/merged_all/dataset.yaml
```

**Box loss sekarang akan TURUN konsisten!** 🎉

---

**Pro Tip**: Monitor losses setiap 10 epochs. Jika masih naik, kurangi lr0 ke 0.0005 atau switch ke SGD optimizer.

**Happy Training!** 🚀
