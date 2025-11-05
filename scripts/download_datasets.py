"""
Script untuk mendownload dataset reCAPTCHA v2, hCaptcha, dan Drag Puzzle dari berbagai sumber
Menggunakan Roboflow API dan direct download dari GitHub/Kaggle
"""

import os
import sys
from pathlib import Path
import subprocess
import shutil
import zipfile
import requests
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.dataset_merger import merge_recaptcha_hcaptcha_datasets


class DatasetDownloader:
    """Download dan setup dataset untuk training"""
    
    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset sources
        self.datasets = {
            'recaptcha_roboflow': {
                'name': 'reCAPTCHA v2 (Roboflow)',
                'type': 'roboflow',
                'workspace': 'my-workspace-4p8ud',
                'project': 'recaptcha-v2',
                'version': 1,
                'images': 1828,
                'format': 'yolov8'
            },
            'hcaptcha_challenger': {
                'name': 'hCaptcha Challenger (Roboflow)',
                'type': 'roboflow',
                'workspace': 'qin2dim',
                'project': 'hcaptcha-challenger',
                'version': 2,
                'images': 712,
                'format': 'yolov8'
            },
            'hcaptcha_images': {
                'name': 'hCaptcha Images (Roboflow)',
                'type': 'roboflow',
                'workspace': 'stopthecap',
                'project': 'hcaptcha-images',
                'version': 2,
                'images': 'multiple',
                'format': 'yolov8'
            },
            'slide_captcha': {
                'name': 'Slide/Drag Puzzle CAPTCHA (Roboflow)',
                'type': 'roboflow',
                'workspace': 'captcha-lwpyk',
                'project': 'slide_captcha',
                'version': 4,
                'images': 2778,
                'format': 'yolov8'
            },
            'captcha_detection': {
                'name': 'Captcha Detection Puzzle (Roboflow)',
                'type': 'roboflow',
                'workspace': 'captcha-detection',
                'project': 'captcha-detection-smkks',
                'version': 1,
                'images': 'multiple',
                'format': 'yolov8',
                'classes': ['jigsaw-piece', 'puzzle', 'target', 'box', 'arrow']
            },
            'hcaptcha_github': {
                'name': 'hCaptcha Dataset (GitHub)',
                'type': 'github',
                'repo': 'orlov-ai/hcaptcha-dataset',
                'url': 'https://github.com/orlov-ai/hcaptcha-dataset.git',
                'images': 'thousands',
                'classes': ['airplane', 'bicycle', 'boat', 'motorbus', 'motorcycle', 'seaplane', 'train', 'truck']
            }
        }
    
    def download_from_roboflow(self, dataset_key: str, api_key: str = None):
        """
        Download dataset dari Roboflow Universe
        
        Args:
            dataset_key: Key dari self.datasets
            api_key: Roboflow API key (optional, bisa diambil dari env)
        """
        if dataset_key not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        dataset_info = self.datasets[dataset_key]
        
        if dataset_info['type'] != 'roboflow':
            raise ValueError(f"Dataset {dataset_key} is not a Roboflow dataset")
        
        print(f"\n{'='*60}")
        print(f"📦 Downloading: {dataset_info['name']}")
        print(f"{'='*60}")
        print(f"Images: {dataset_info['images']}")
        print(f"Format: {dataset_info['format']}")
        
        try:
            from roboflow import Roboflow
            
            # Initialize Roboflow
            if api_key is None:
                api_key = os.environ.get('ROBOFLOW_API_KEY')
                if api_key is None:
                    print("\n⚠️  No API key provided. Please:")
                    print("1. Create account at: https://roboflow.com")
                    print("2. Get API key from: https://app.roboflow.com/settings/api")
                    print("3. Set environment variable: export ROBOFLOW_API_KEY='your_key'")
                    print("   OR pass as argument: --api-key your_key")
                    return None
            
            rf = Roboflow(api_key=api_key)
            
            # Download dataset
            project = rf.workspace(dataset_info['workspace']).project(dataset_info['project'])
            dataset = project.version(dataset_info['version']).download(dataset_info['format'])
            
            # Move to organized location
            dest_dir = self.output_dir / dataset_key
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            
            shutil.move(dataset.location, dest_dir)
            
            print(f"✅ Downloaded to: {dest_dir}")
            return str(dest_dir)
            
        except ImportError:
            print("❌ Roboflow package not installed")
            print("   Install with: pip install roboflow")
            return None
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            return None
    
    def download_from_github(self, dataset_key: str):
        """Download dataset dari GitHub"""
        if dataset_key not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        dataset_info = self.datasets[dataset_key]
        
        if dataset_info['type'] != 'github':
            raise ValueError(f"Dataset {dataset_key} is not a GitHub dataset")
        
        print(f"\n{'='*60}")
        print(f"📦 Downloading: {dataset_info['name']}")
        print(f"{'='*60}")
        
        dest_dir = self.output_dir / dataset_key
        
        try:
            # Clone repository
            print(f"🔄 Cloning from: {dataset_info['url']}")
            subprocess.run(['git', 'clone', dataset_info['url'], str(dest_dir)], 
                         check=True, capture_output=True)
            
            print(f"✅ Downloaded to: {dest_dir}")
            return str(dest_dir)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git clone failed: {e}")
            return None
        except FileNotFoundError:
            print("❌ Git not installed")
            return None
    
    def download_all(self, api_key: str = None, include_github: bool = False):
        """
        Download semua dataset
        
        Args:
            api_key: Roboflow API key
            include_github: Include GitHub datasets (larger download)
        """
        downloaded = []
        
        print("\n" + "="*60)
        print("📥 DOWNLOADING ALL DATASETS")
        print("="*60)
        
        # Download Roboflow datasets
        roboflow_datasets = [k for k, v in self.datasets.items() if v['type'] == 'roboflow']
        
        for dataset_key in roboflow_datasets:
            result = self.download_from_roboflow(dataset_key, api_key)
            if result:
                downloaded.append(result)
        
        # Download GitHub datasets if requested
        if include_github:
            github_datasets = [k for k, v in self.datasets.items() if v['type'] == 'github']
            
            for dataset_key in github_datasets:
                result = self.download_from_github(dataset_key)
                if result:
                    downloaded.append(result)
        
        print(f"\n{'='*60}")
        print(f"✅ Downloaded {len(downloaded)} datasets")
        print(f"{'='*60}")
        
        return downloaded
    
    def list_datasets(self):
        """List semua dataset yang tersedia"""
        print("\n" + "="*60)
        print("📋 AVAILABLE DATASETS")
        print("="*60)
        
        for key, info in self.datasets.items():
            print(f"\n🔹 {key}")
            print(f"   Name: {info['name']}")
            print(f"   Type: {info['type']}")
            print(f"   Images: {info['images']}")
            if 'classes' in info:
                print(f"   Classes: {', '.join(info['classes'])}")
        
        print("\n" + "="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download CAPTCHA datasets')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Roboflow API key (or set ROBOFLOW_API_KEY env var)')
    parser.add_argument('--output', type=str, default='datasets/downloaded',
                       help='Output directory')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to download (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    parser.add_argument('--include-github', action='store_true',
                       help='Include GitHub datasets (larger)')
    parser.add_argument('--merge', action='store_true',
                       help='Merge downloaded datasets after download')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.output)
    
    # List datasets if requested
    if args.list:
        downloader.list_datasets()
        return
    
    # Download specific or all datasets
    downloaded_paths = []
    
    if args.dataset:
        if args.dataset in downloader.datasets:
            dataset_info = downloader.datasets[args.dataset]
            if dataset_info['type'] == 'roboflow':
                path = downloader.download_from_roboflow(args.dataset, args.api_key)
            elif dataset_info['type'] == 'github':
                path = downloader.download_from_github(args.dataset)
            
            if path:
                downloaded_paths.append(path)
        else:
            print(f"❌ Unknown dataset: {args.dataset}")
            print("Use --list to see available datasets")
            return
    else:
        downloaded_paths = downloader.download_all(args.api_key, args.include_github)
    
    # Merge if requested
    if args.merge and downloaded_paths:
        print(f"\n{'='*60}")
        print("🔗 MERGING DATASETS")
        print(f"{'='*60}")
        
        # Separate by type
        recaptcha_paths = [p for p in downloaded_paths if 'recaptcha' in str(p).lower()]
        hcaptcha_paths = [p for p in downloaded_paths if 'hcaptcha' in str(p).lower() or 'captcha' in str(p).lower()]
        
        config_path = merge_recaptcha_hcaptcha_datasets(
            recaptcha_paths=recaptcha_paths,
            hcaptcha_paths=hcaptcha_paths,
            output_dir=f"{args.output}/merged_all"
        )
        
        print(f"\n✅ Merged dataset config: {config_path}")
        print(f"\n🚀 Ready to train with:")
        print(f"   python scripts/train_yolov8.py --data {config_path}")


if __name__ == "__main__":
    main()
