# Setup Google Drive Upload untuk Model YOLOv8

## Langkah-langkah Setup di VPS

### 1. Install Dependencies

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 2. Dapatkan Google Drive API Credentials

1. **Buka Google Cloud Console**
   - Kunjungi: https://console.cloud.google.com

2. **Buat Project Baru** (atau gunakan existing)
   - Klik "Select a project" → "New Project"
   - Beri nama project (misal: "CAPTCHA-Model-Upload")
   - Klik "Create"

3. **Enable Google Drive API**
   - Di dashboard, cari "Google Drive API"
   - Klik "Enable"

4. **Buat OAuth 2.0 Credentials**
   - Menu: APIs & Services → Credentials
   - Klik "Create Credentials" → "OAuth client ID"
   - Jika diminta, configure consent screen:
     - User Type: External
     - App name: "CAPTCHA Model Uploader"
     - User support email: email Anda
     - Developer contact: email Anda
     - Klik Save
   - Application type: "Desktop app"
   - Name: "CAPTCHA Uploader"
   - Klik "Create"

5. **Download Credentials**
   - Klik tombol download (ikon ⬇️)
   - Save sebagai `credentials.json`
   - Upload file ini ke VPS Anda (di folder root project)

### 3. Cara Menggunakan

#### A. Upload Otomatis (Experiment Terbaru)

```bash
# Upload experiment terbaru
python scripts/collect_and_upload_models.py
```

Pada first run:
- Browser akan terbuka untuk autentikasi
- Login dengan Google account Anda
- Klik "Allow"
- Token akan disimpan di `token.pickle` untuk next run

#### B. Upload Experiment Spesifik

```bash
# Upload experiment tertentu
python scripts/collect_and_upload_models.py --experiment captcha_20241105_123456
```

#### C. Upload ke Folder Google Drive Tertentu

```bash
# Dapatkan folder ID dari Google Drive URL:
# https://drive.google.com/drive/folders/FOLDER_ID_DISINI
python scripts/collect_and_upload_models.py --folder-id "FOLDER_ID_DISINI"
```

#### D. Hanya Collect & Zip (Tanpa Upload)

```bash
# Skip upload, hanya collect dan zip
python scripts/collect_and_upload_models.py --skip-upload
```

### 4. Parameter Lengkap

```bash
python scripts/collect_and_upload_models.py \
  --experiment captcha_20241105_123456 \  # Nama experiment spesifik
  --training-dir runs/train \             # Directory training
  --output-dir MODELS \                   # Output directory
  --credentials credentials.json \        # Credentials file
  --folder-id "FOLDER_ID" \              # Google Drive folder ID
  --skip-upload                          # Skip upload
```

### 5. File Yang Dikumpulkan

Script akan mengumpulkan:
- ✅ `weights/best.pt` - Model terbaik (PyTorch)
- ✅ `weights/last.pt` - Model terakhir (PyTorch)
- ✅ `weights/best.onnx` - Model ONNX (jika ada)
- ✅ `results.csv` - Metrics training
- ✅ `results.png` - Grafik training
- ✅ `confusion_matrix.png` - Confusion matrix
- ✅ `args.yaml` - Training arguments
- ✅ `data.yaml` - Dataset config
- ✅ `README.md` - Dokumentasi model

### 6. Troubleshooting

#### Error: credentials.json not found
```bash
# Pastikan file credentials.json ada di root project
ls -la credentials.json
```

#### Error: Failed to start local server
```bash
# Jika di VPS tanpa GUI, gunakan manual flow:
# Edit script, ganti `run_local_server` dengan `run_console`
```

Untuk VPS tanpa browser, gunakan method alternatif:

```python
# Di script collect_and_upload_models.py, ganti baris:
# self.creds = flow.run_local_server(port=0)

# Dengan:
self.creds = flow.run_console()
```

Kemudian ikuti instruksi di console untuk copy URL dan paste code.

#### Error: Access denied
```bash
# Pastikan scope sudah benar di OAuth consent screen
# Tambahkan test users jika app masih "Testing"
```

### 7. Automasi dengan Cron (Optional)

Untuk auto-upload setiap kali training selesai:

```bash
# Edit crontab
crontab -e

# Tambahkan (cek setiap jam, upload jika ada model baru):
0 * * * * cd /path/to/project && python scripts/collect_and_upload_models.py
```

### 8. Keamanan

⚠️ **PENTING:**
- Jangan commit `credentials.json` ke git
- Jangan commit `token.pickle` ke git
- Tambahkan ke `.gitignore`:

```
credentials.json
token.pickle
MODELS/
*.zip
```

### 9. Struktur Folder MODELS

```
MODELS/
├── model_captcha_20241105_123456_20241105_143022/
│   ├── weights/
│   │   ├── best.pt
│   │   ├── last.pt
│   │   └── best.onnx
│   ├── results.csv
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── args.yaml
│   ├── data.yaml
│   └── README.md
└── model_captcha_20241105_123456_20241105_143022.zip
```

### 10. Tips

1. **Backup Regular**: Jalankan setiap selesai training
2. **Naming Convention**: Folder otomatis diberi timestamp
3. **Cleanup**: Hapus folder lokal setelah upload berhasil
4. **Monitor**: Check Google Drive untuk memastikan upload berhasil

## Support

Jika ada masalah:
1. Check log error
2. Pastikan credentials.json valid
3. Pastikan Google Drive API enabled
4. Check quota Google Drive (15GB free)
