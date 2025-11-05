# Script Upload Model ke Google Drive

## Quick Start di VPS

### 1. Install Google Drive Dependencies

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 2. Setup Google Drive API

Ikuti panduan lengkap di file `SETUP_GOOGLE_DRIVE.md`

Ringkasan:
1. Buka https://console.cloud.google.com
2. Enable Google Drive API
3. Buat OAuth 2.0 credentials
4. Download sebagai `credentials.json`
5. Upload ke VPS (letakkan di root project)

### 3. Jalankan Script

```bash
# Upload experiment terbaru
python scripts/collect_and_upload_models.py

# Upload experiment spesifik
python scripts/collect_and_upload_models.py --experiment captcha_20241105_123456

# Upload ke folder Google Drive tertentu
python scripts/collect_and_upload_models.py --folder-id "FOLDER_ID_DARI_GDRIVE"

# Hanya collect dan zip (tanpa upload)
python scripts/collect_and_upload_models.py --skip-upload
```

## Apa yang Dilakukan Script?

1. **Collect**: Mengumpulkan semua file penting dari hasil training:
   - Model weights (.pt dan .onnx)
   - Results dan metrics
   - Config files
   - Grafik dan visualisasi

2. **Organize**: Membuat folder MODELS dengan struktur rapi dan timestamp

3. **Zip**: Compress semua file jadi 1 zip file

4. **Upload**: Upload zip ke Google Drive

## Output

```
MODELS/
└── model_captcha_20241105_123456_20241105_143022/
    ├── weights/
    │   ├── best.pt      (Model terbaik)
    │   ├── last.pt      (Model terakhir)
    │   └── best.onnx    (Model ONNX)
    ├── results.csv      (Metrics)
    ├── results.png      (Grafik)
    ├── confusion_matrix.png
    ├── args.yaml        (Training config)
    ├── data.yaml        (Dataset config)
    └── README.md        (Dokumentasi)

File ZIP: model_captcha_20241105_123456_20241105_143022.zip
```

## Lihat Panduan Lengkap

Buka file `SETUP_GOOGLE_DRIVE.md` untuk:
- Setup Google Cloud Console
- Troubleshooting
- Automasi dengan cron
- Tips keamanan
