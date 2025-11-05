"""
Script untuk mengumpulkan hasil training model YOLOv8, zip, dan upload ke Google Drive
Jalankan di VPS setelah training selesai
"""

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.file']

class ModelCollector:
    def __init__(self, training_dir='runs/train', output_dir='MODELS'):
        self.training_dir = Path(training_dir)
        self.output_dir = Path(output_dir)
        
    def collect_models(self, experiment_name=None):
        """
        Kumpulkan semua file penting dari hasil training
        
        Args:
            experiment_name: Nama experiment spesifik (misal: 'captcha_xxx')
                           Jika None, akan mengumpulkan experiment terbaru
        """
        print("=" * 70)
        print("📦 MENGUMPULKAN FILE MODEL")
        print("=" * 70)
        
        # Buat folder MODELS
        self.output_dir.mkdir(exist_ok=True)
        
        # Cari experiment
        if experiment_name:
            exp_path = self.training_dir / experiment_name
        else:
            # Ambil experiment terbaru
            experiments = sorted(self.training_dir.glob('captcha_*'), 
                               key=lambda x: x.stat().st_mtime, 
                               reverse=True)
            if not experiments:
                print("❌ Tidak ada hasil training ditemukan!")
                return None
            exp_path = experiments[0]
        
        if not exp_path.exists():
            print(f"❌ Experiment '{experiment_name}' tidak ditemukan!")
            return None
            
        print(f"✅ Mengumpulkan dari: {exp_path}")
        
        # Buat subfolder dengan timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_folder = self.output_dir / f"model_{exp_path.name}_{timestamp}"
        model_folder.mkdir(exist_ok=True)
        
        files_collected = []
        
        # 1. Copy weights (PT dan ONNX)
        weights_dir = exp_path / 'weights'
        if weights_dir.exists():
            dest_weights = model_folder / 'weights'
            dest_weights.mkdir(exist_ok=True)
            
            for weight_file in weights_dir.glob('*'):
                if weight_file.suffix in ['.pt', '.onnx']:
                    shutil.copy2(weight_file, dest_weights / weight_file.name)
                    files_collected.append(f"weights/{weight_file.name}")
                    print(f"  ✅ {weight_file.name}")
        
        # 2. Copy hasil evaluasi
        results_files = ['results.csv', 'results.png', 'confusion_matrix.png']
        for res_file in results_files:
            src = exp_path / res_file
            if src.exists():
                shutil.copy2(src, model_folder / res_file)
                files_collected.append(res_file)
                print(f"  ✅ {res_file}")
        
        # 3. Copy training args
        args_file = exp_path / 'args.yaml'
        if args_file.exists():
            shutil.copy2(args_file, model_folder / 'args.yaml')
            files_collected.append('args.yaml')
            print(f"  ✅ args.yaml")
        
        # 4. Copy dataset config
        data_yaml = exp_path / 'data.yaml'
        if data_yaml.exists():
            shutil.copy2(data_yaml, model_folder / 'data.yaml')
            files_collected.append('data.yaml')
            print(f"  ✅ data.yaml")
        
        # 5. Buat file README dengan info model
        readme_content = f"""# YOLOv8 CAPTCHA Solver Model
        
## Informasi Model
- Experiment: {exp_path.name}
- Dikumpulkan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Total file: {len(files_collected)}

## File Yang Dikumpulkan
{chr(10).join(f'- {f}' for f in files_collected)}

## Cara Menggunakan

### 1. Inference dengan PyTorch (.pt)
```python
from ultralytics import YOLO

model = YOLO('weights/best.pt')
results = model('captcha_image.jpg')
```

### 2. Inference dengan ONNX
```python
import onnxruntime as ort
import cv2
import numpy as np

session = ort.InferenceSession('weights/best.onnx')
img = cv2.imread('captcha_image.jpg')
img = cv2.resize(img, (640, 640))
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

outputs = session.run(None, {{session.get_inputs()[0].name: img}})
```

### 3. Load dengan API Inference
```bash
python scripts/inference.py --model weights/best.pt --source test_image.jpg
```

## Metrics
Lihat `results.csv` dan `results.png` untuk detail performa model.
"""
        
        with open(model_folder / 'README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"\n✅ Total {len(files_collected)} file dikumpulkan ke: {model_folder}")
        return model_folder
    
    def create_zip(self, model_folder):
        """Buat ZIP dari folder model"""
        print("\n" + "=" * 70)
        print("📦 MEMBUAT ZIP FILE")
        print("=" * 70)
        
        zip_path = Path(str(model_folder) + '.zip')
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in model_folder.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(model_folder.parent)
                    zipf.write(file, arcname)
                    print(f"  ✅ {arcname}")
        
        zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
        print(f"\n✅ ZIP dibuat: {zip_path} ({zip_size:.2f} MB)")
        return zip_path


class GoogleDriveUploader:
    def __init__(self, credentials_file='credentials.json'):
        self.credentials_file = credentials_file
        self.creds = None
        
    def authenticate(self):
        """Autentikasi dengan Google Drive API"""
        print("\n" + "=" * 70)
        print("🔐 AUTENTIKASI GOOGLE DRIVE")
        print("=" * 70)
        
        # Token file untuk menyimpan access token
        token_file = 'token.pickle'
        
        if os.path.exists(token_file):
            with open(token_file, 'rb') as token:
                self.creds = pickle.load(token)
        
        # Jika tidak ada credentials atau expired
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                print("♻️  Refresh token...")
                self.creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    print(f"❌ File '{self.credentials_file}' tidak ditemukan!")
                    print("\nCara mendapatkan credentials.json:")
                    print("1. Buka https://console.cloud.google.com")
                    print("2. Buat project baru atau pilih existing project")
                    print("3. Enable Google Drive API")
                    print("4. Buat OAuth 2.0 credentials")
                    print("5. Download credentials.json")
                    return False
                
                print("🌐 Membuka browser untuk autentikasi...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            # Simpan token untuk next run
            with open(token_file, 'wb') as token:
                pickle.dump(self.creds, token)
        
        print("✅ Autentikasi berhasil!")
        return True
    
    def upload_file(self, file_path, folder_id=None):
        """
        Upload file ke Google Drive
        
        Args:
            file_path: Path ke file yang akan diupload
            folder_id: ID folder Google Drive (optional)
        """
        if not self.authenticate():
            return None
        
        print("\n" + "=" * 70)
        print("☁️  UPLOAD KE GOOGLE DRIVE")
        print("=" * 70)
        
        try:
            service = build('drive', 'v3', credentials=self.creds)
            
            file_metadata = {
                'name': Path(file_path).name
            }
            
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            media = MediaFileUpload(file_path, resumable=True)
            
            print(f"📤 Uploading {Path(file_path).name}...")
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink, size'
            ).execute()
            
            print(f"\n✅ Upload berhasil!")
            print(f"📄 File: {file.get('name')}")
            print(f"🆔 ID: {file.get('id')}")
            print(f"📊 Size: {int(file.get('size', 0)) / (1024*1024):.2f} MB")
            print(f"🔗 Link: {file.get('webViewLink')}")
            
            return file
            
        except Exception as e:
            print(f"❌ Upload gagal: {str(e)}")
            return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kumpulkan, zip, dan upload model YOLOv8 ke Google Drive')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Nama experiment spesifik (default: terbaru)')
    parser.add_argument('--training-dir', type=str, default='runs/train',
                       help='Directory hasil training (default: runs/train)')
    parser.add_argument('--output-dir', type=str, default='MODELS',
                       help='Output directory (default: MODELS)')
    parser.add_argument('--credentials', type=str, default='credentials.json',
                       help='Google credentials file (default: credentials.json)')
    parser.add_argument('--folder-id', type=str, default=None,
                       help='Google Drive folder ID (optional)')
    parser.add_argument('--skip-upload', action='store_true',
                       help='Skip upload, hanya collect dan zip')
    
    args = parser.parse_args()
    
    # 1. Collect models
    collector = ModelCollector(args.training_dir, args.output_dir)
    model_folder = collector.collect_models(args.experiment)
    
    if not model_folder:
        return
    
    # 2. Create ZIP
    zip_path = collector.create_zip(model_folder)
    
    # 3. Upload ke Google Drive
    if not args.skip_upload:
        uploader = GoogleDriveUploader(args.credentials)
        result = uploader.upload_file(zip_path, args.folder_id)
        
        if result:
            print("\n" + "=" * 70)
            print("🎉 SELESAI!")
            print("=" * 70)
            print(f"✅ Model berhasil dikumpulkan, di-zip, dan diupload!")
            print(f"📦 Local ZIP: {zip_path}")
            print(f"☁️  Google Drive: {result.get('webViewLink')}")
    else:
        print("\n" + "=" * 70)
        print("✅ SELESAI! (Upload di-skip)")
        print("=" * 70)
        print(f"📦 ZIP file: {zip_path}")


if __name__ == '__main__':
    main()
