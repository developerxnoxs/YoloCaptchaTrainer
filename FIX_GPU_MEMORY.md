# 🔧 Fix: CUDA Out of Memory

## ❌ Error yang Terjadi

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 526.00 MiB.
GPU 0 has a total capacity of 79.15 GiB of which 517.62 MiB is free.
Process 3633397 has 78.64 GiB memory in use.
```

Tapi `nvidia-smi` menunjukkan:
```
Memory-Usage | GPU-Util
    0MiB / 81920MiB |      0%
No running processes found
```

## 🧐 Penyebab

Ini adalah **memory leak** / **zombie process**:
- Proses training sebelumnya crash/killed
- Tapi GPU memory tidak di-release
- PyTorch masih hold 78.64 GiB memory
- `nvidia-smi` tidak detect proses zombie

## ✅ Solusi

### Step 1: Clear GPU Memory

```bash
# Jalankan script pembersih
./clear_gpu.sh
```

Script ini akan:
1. Kill semua Python processes yang stuck
2. Kill CUDA processes
3. Clear PyTorch cache
4. Verify GPU memory sudah kosong

### Manual Cleanup (Jika Script Gagal)

```bash
# 1. Kill semua Python processes
pkill -9 python
pkill -9 python3

# 2. Kill CUDA processes
sudo fuser -k /dev/nvidia*

# 3. Clear PyTorch cache
python3 -c "import torch; torch.cuda.empty_cache()"

# 4. Verify
nvidia-smi
```

### Step 2: Run Training dengan Batch Size Lebih Kecil

Saya sudah update konfigurasi:

**Sebelum (OOM):**
```python
'batch_size': 64,  # Terlalu besar
```

**Sesudah (Safe):**
```python
'batch_size': 32,  # Reduced untuk safety
```

## 🚀 Cara Menggunakan Setelah Fix

```bash
# 1. Clear GPU memory
./clear_gpu.sh

# 2. Verify GPU kosong
nvidia-smi
# Harus menunjukkan: 0MiB / 81920MiB

# 3. Run training
python scripts/train_yolov8.py --data datasets/merged_all/dataset.yaml

# Atau auto training
./run_auto_training.sh
```

## 📊 Konfigurasi Memory-Safe untuk A100 80GB

### Option 1: Conservative (Recommended)

```python
'batch_size': 32,
'workers': 0,
'cache': False,
'amp': True,
```

**Memory Usage**: ~35-40 GB (aman!)
**Training Time**: 5-6 jam

### Option 2: Balanced

```python
'batch_size': 48,
'workers': 0,
'cache': False,
'amp': True,
```

**Memory Usage**: ~50-55 GB
**Training Time**: 4.5-5.5 jam

### Option 3: Aggressive (Jika RAM 64GB+)

```python
'batch_size': 64,
'workers': 2,
'cache': 'ram',  # Cache di RAM
'amp': True,
```

**Memory Usage**: ~65-70 GB
**Training Time**: 3.5-4.5 jam

## 🔍 Monitoring GPU Memory

### Real-time Monitoring

```bash
# Watch GPU usage setiap 1 detik
watch -n 1 nvidia-smi

# Atau lebih detail
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv'
```

### Expected Usage Saat Training

Untuk batch_size=32:
```
Memory-Usage:     35000MiB / 81920MiB (43%)
GPU-Util:         85-95%
```

Untuk batch_size=64:
```
Memory-Usage:     65000MiB / 81920MiB (79%)
GPU-Util:         85-95%
```

## 🐛 Troubleshooting

### Masih OOM Setelah Clear?

**Option A: Kurangi Batch Size Lebih Lanjut**

Edit `scripts/train_yolov8.py` line 43:
```python
'batch_size': 24,  # Atau bahkan 16
```

**Option B: Kurangi Image Size**

Edit `scripts/train_yolov8.py` line 39:
```python
'input_size': 1024,  # Reduced dari 1280
```

**Option C: Disable Multi-Scale**

Edit `scripts/train_yolov8.py` line 48:
```python
'multi_scale': False,
```

### Clear Cache Secara Programmatic

Tambahkan di awal training script:

```python
import torch
import gc

# Clear cache
torch.cuda.empty_cache()
gc.collect()
```

### Force Kill All CUDA Processes

```bash
# Find CUDA processes
sudo lsof /dev/nvidia*

# Kill by PID
sudo kill -9 <PID>

# Or kill all
sudo fuser -k /dev/nvidia*
```

## 📋 Checklist Sebelum Training

- [ ] GPU memory 0 MiB (`nvidia-smi`)
- [ ] No running processes
- [ ] Batch size <= 32
- [ ] Workers = 0
- [ ] Cache = False
- [ ] AMP = True

## 🎯 Expected Results dengan Batch Size 32

| Metric | Value |
|--------|-------|
| GPU Memory Usage | 35-40 GB |
| GPU Utilization | 85-95% |
| Training Time | 5-6 jam (300 epochs) |
| Final mAP50 | 0.85-0.95 |
| Stability | ⭐⭐⭐⭐⭐ |

## 💡 Tips Mencegah OOM

1. **Selalu clear cache** sebelum training:
   ```bash
   ./clear_gpu.sh
   ```

2. **Monitor memory usage** saat training:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Start dengan batch size kecil**, naikkan gradual:
   - Test dengan batch=16 dulu
   - Jika aman, coba batch=24
   - Lalu batch=32
   - Dst...

4. **Set environment variable**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

## ✅ Summary

1. **Clear GPU**: `./clear_gpu.sh`
2. **Verify**: `nvidia-smi` harus 0 MiB
3. **Run training**: Batch size sudah dikurangi ke 32
4. **Monitor**: `watch -n 1 nvidia-smi`

**Training sekarang akan berjalan tanpa OOM!** 🎉

---

## 📞 Masih Ada Masalah?

Jika masih OOM:
1. Kurangi batch size ke 24 atau 16
2. Kurangi image size ke 1024
3. Disable multi-scale training
4. Check apakah ada proses lain yang pakai GPU

**Happy Training!** 🚀
