#!/usr/bin/env python3
"""
Auto Training Pipeline - YOLOv8 CAPTCHA Solver
Script otomatis untuk download dataset, merge, dan training
Optimized untuk GPU A100 80GB
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# Import local modules
from scripts.download_datasets import DatasetDownloader
from utils.dataset_merger import merge_recaptcha_hcaptcha_datasets
from scripts.train_yolov8 import YOLOv8Trainer


class AutoTrainingPipeline:
    """Pipeline otomatis untuk training YOLOv8"""
    
    def __init__(self, 
                 roboflow_api_key: str = None,
                 skip_download: bool = False,
                 skip_merge: bool = False,
                 datasets_to_download: list = None):
        """
        Initialize pipeline
        
        Args:
            roboflow_api_key: Roboflow API key
            skip_download: Skip download step jika dataset sudah ada
            skip_merge: Skip merge step jika merged dataset sudah ada
            datasets_to_download: List dataset yang akan didownload (default: semua)
        """
        self.api_key = roboflow_api_key or os.environ.get('ROBOFLOW_API_KEY')
        self.skip_download = skip_download
        self.skip_merge = skip_merge
        self.datasets_to_download = datasets_to_download
        
        # Directories
        self.download_dir = Path("datasets/downloaded")
        self.merged_dir = Path("datasets/merged_all")
        self.output_dir = Path("runs/train")
        
        # Stats
        self.start_time = None
        self.stats = {
            'datasets_downloaded': 0,
            'total_images': 0,
            'training_time': 0,
            'final_metrics': {}
        }
    
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    
    def check_requirements(self):
        """Check sistem requirements"""
        self.print_header("🔍 CHECKING REQUIREMENTS")
        
        # Check Python
        import sys
        print(f"✅ Python {sys.version}")
        
        # Check PyTorch & CUDA
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__}")
            
            if torch.cuda.is_available():
                print(f"✅ CUDA {torch.version.cuda}")
                print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
                print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print("⚠️  WARNING: CUDA tidak tersedia!")
                response = input("Lanjutkan tanpa GPU? (y/n): ")
                if response.lower() != 'y':
                    print("Cancelled.")
                    sys.exit(0)
        except ImportError:
            print("❌ PyTorch tidak terinstall!")
            print("Run: ./setup.sh")
            sys.exit(1)
        
        # Check Ultralytics
        try:
            from ultralytics import YOLO
            print("✅ Ultralytics YOLOv8")
        except ImportError:
            print("❌ Ultralytics tidak terinstall!")
            print("Run: pip install ultralytics")
            sys.exit(1)
        
        # Check Roboflow API Key
        if not self.skip_download:
            if self.api_key:
                print(f"✅ Roboflow API Key: {self.api_key[:8]}...")
            else:
                print("\n❌ ROBOFLOW_API_KEY tidak ditemukan!")
                print("\nDapatkan API key dari: https://app.roboflow.com/settings/api")
                print("Lalu set environment variable:")
                print("  export ROBOFLOW_API_KEY='your_api_key'")
                print("\nATAU jalankan dengan flag:")
                print("  python auto_train.py --api-key YOUR_API_KEY")
                
                response = input("\nMasukkan API key sekarang (atau tekan Enter untuk skip download): ")
                if response.strip():
                    self.api_key = response.strip()
                    print(f"✅ API Key set: {self.api_key[:8]}...")
                else:
                    print("⚠️  Download step akan di-skip")
                    self.skip_download = True
        
        print("\n✅ Semua requirements terpenuhi!\n")
    
    def download_datasets(self):
        """Download semua dataset dari Roboflow"""
        if self.skip_download:
            self.print_header("⏭️  SKIPPING DOWNLOAD (--skip-download)")
            return
        
        self.print_header("📥 DOWNLOADING DATASETS")
        
        downloader = DatasetDownloader(str(self.download_dir))
        
        # List available datasets
        print("Dataset yang akan didownload:")
        for key, info in downloader.datasets.items():
            if info['type'] == 'roboflow':
                if self.datasets_to_download is None or key in self.datasets_to_download:
                    print(f"  • {info['name']} ({info['images']} images)")
        
        print("")
        
        # Download datasets
        downloaded_paths = []
        
        for key, info in downloader.datasets.items():
            if info['type'] == 'roboflow':
                # Skip if specific datasets requested and this isn't one of them
                if self.datasets_to_download and key not in self.datasets_to_download:
                    continue
                
                try:
                    path = downloader.download_from_roboflow(key, self.api_key)
                    if path:
                        downloaded_paths.append(path)
                        self.stats['datasets_downloaded'] += 1
                except Exception as e:
                    print(f"⚠️  Failed to download {key}: {str(e)}")
                    continue
        
        print(f"\n✅ Downloaded {len(downloaded_paths)} datasets")
        return downloaded_paths
    
    def merge_datasets(self):
        """Merge semua dataset yang sudah didownload"""
        if self.skip_merge:
            self.print_header("⏭️  SKIPPING MERGE (--skip-merge)")
            
            # Check if merged dataset exists
            merged_yaml = self.merged_dir / "dataset.yaml"
            if merged_yaml.exists():
                print(f"Using existing merged dataset: {merged_yaml}")
                return str(merged_yaml)
            else:
                print("❌ Merged dataset tidak ditemukan!")
                print(f"Expected: {merged_yaml}")
                sys.exit(1)
        
        self.print_header("🔗 MERGING DATASETS")
        
        # Find all downloaded datasets
        dataset_paths = []
        
        if self.download_dir.exists():
            for item in self.download_dir.iterdir():
                if item.is_dir():
                    # Check if it has data.yaml or dataset.yaml
                    if (item / "data.yaml").exists() or (item / "dataset.yaml").exists():
                        dataset_paths.append(str(item))
                        print(f"  • Found: {item.name}")
        
        if not dataset_paths:
            print("❌ Tidak ada dataset yang ditemukan!")
            print(f"Directory: {self.download_dir}")
            sys.exit(1)
        
        print(f"\nMerging {len(dataset_paths)} datasets...")
        
        # Separate by type (for better class organization)
        recaptcha_paths = [p for p in dataset_paths if 'recaptcha' in str(p).lower()]
        other_paths = [p for p in dataset_paths if 'recaptcha' not in str(p).lower()]
        
        # Merge all datasets
        config_path = merge_recaptcha_hcaptcha_datasets(
            recaptcha_paths=recaptcha_paths,
            hcaptcha_paths=other_paths,
            output_dir=str(self.merged_dir)
        )
        
        print(f"\n✅ Datasets merged successfully!")
        print(f"📄 Config: {config_path}")
        
        return config_path
    
    def train_model(self, dataset_yaml: str):
        """Train YOLOv8 model"""
        self.print_header("🏋️  TRAINING YOLOV8 MODEL")
        
        print("Configuration:")
        print(f"  • Dataset: {dataset_yaml}")
        print(f"  • Model: YOLOv8x (largest)")
        print(f"  • Batch Size: 64")
        print(f"  • Image Size: 1280")
        print(f"  • Epochs: 300")
        print(f"  • Device: GPU (A100 80GB optimized)")
        print("")
        
        # Initialize trainer
        trainer = YOLOv8Trainer()
        
        # Training name with timestamp
        exp_name = f"captcha_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start training
        training_start = time.time()
        
        results = trainer.train(
            data_yaml=dataset_yaml,
            project=str(self.output_dir),
            name=exp_name
        )
        
        training_time = time.time() - training_start
        self.stats['training_time'] = training_time
        
        # Get best model path
        best_model = self.output_dir / exp_name / 'weights' / 'best.pt'
        
        print(f"\n✅ Training completed in {training_time / 3600:.2f} hours")
        print(f"📁 Best model: {best_model}")
        
        return str(best_model)
    
    def export_model(self, model_path: str):
        """Export model ke ONNX dan TorchScript"""
        self.print_header("📦 EXPORTING MODEL")
        
        trainer = YOLOv8Trainer()
        trainer.export_model(model_path, formats=['onnx', 'torchscript'])
        
        print("✅ Model exported!")
    
    def print_summary(self):
        """Print summary of the pipeline"""
        total_time = time.time() - self.start_time
        
        self.print_header("📊 TRAINING SUMMARY")
        
        print(f"Total Time: {total_time / 3600:.2f} hours")
        print(f"Datasets Downloaded: {self.stats['datasets_downloaded']}")
        print(f"Training Time: {self.stats['training_time'] / 3600:.2f} hours")
        print("")
        
        # Find best model
        if self.output_dir.exists():
            latest_run = max(self.output_dir.glob("captcha_*"), default=None, 
                           key=lambda p: p.stat().st_mtime)
            if latest_run:
                best_model = latest_run / 'weights' / 'best.pt'
                last_model = latest_run / 'weights' / 'last.pt'
                
                print("Model Locations:")
                print(f"  • Best: {best_model}")
                print(f"  • Last: {last_model}")
                print("")
                
                # Show results location
                print("Results:")
                print(f"  • Training plots: {latest_run}")
                print(f"  • Metrics: {latest_run / 'results.csv'}")
        
        print(f"\n{'='*70}")
        print("✅ PIPELINE COMPLETED!")
        print(f"{'='*70}\n")
    
    def run(self):
        """Jalankan full pipeline"""
        self.start_time = time.time()
        
        print(f"\n{'='*70}")
        print("  🚀 YOLOv8 CAPTCHA Solver - Auto Training Pipeline")
        print("  Optimized for NVIDIA A100 80GB")
        print(f"{'='*70}\n")
        
        try:
            # 1. Check requirements
            self.check_requirements()
            
            # 2. Download datasets
            self.download_datasets()
            
            # 3. Merge datasets
            dataset_yaml = self.merge_datasets()
            
            # 4. Train model
            best_model = self.train_model(dataset_yaml)
            
            # 5. Export model (optional)
            # self.export_model(best_model)
            
            # 6. Print summary
            self.print_summary()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Auto Training Pipeline untuk YOLOv8 CAPTCHA Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full auto (download + merge + train)
  python auto_train.py --api-key YOUR_KEY
  
  # Skip download (gunakan dataset yang sudah ada)
  python auto_train.py --skip-download
  
  # Download dataset tertentu saja
  python auto_train.py --api-key YOUR_KEY --datasets recaptcha_roboflow slide_captcha
  
  # Resume dari merged dataset
  python auto_train.py --skip-download --skip-merge
        """
    )
    
    parser.add_argument('--api-key', type=str, default=None,
                       help='Roboflow API key (or set ROBOFLOW_API_KEY env var)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download step (gunakan dataset yang sudah ada)')
    parser.add_argument('--skip-merge', action='store_true',
                       help='Skip merge step (gunakan merged dataset yang sudah ada)')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to download (default: all)')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = AutoTrainingPipeline(
        roboflow_api_key=args.api_key,
        skip_download=args.skip_download,
        skip_merge=args.skip_merge,
        datasets_to_download=args.datasets
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
