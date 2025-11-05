"""
Training Script untuk YOLOv8 - Optimized untuk GPU A100 80GB
Dataset: reCAPTCHA v2 + hCaptcha (Merged)
"""

import os
import sys
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import argparse
from datetime import datetime


class YOLOv8Trainer:
    """Trainer class untuk YOLOv8 dengan konfigurasi optimal A100"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  No GPU detected, using CPU")
    
    def _default_config(self):
        """Default configuration untuk A100 80GB"""
        return {
            'model': {
                'architecture': 'yolov8x',  # Largest model untuk A100
                'pretrained': True,
                'input_size': 1024,  # Reduced from 1280 to save memory
            },
            'training': {
                'epochs': 300,
                'batch_size': 16,  # Reduced from 32 for stability
                'workers': 0,  # 0 = single-threaded, NO shared memory usage!
                'patience': 50,
                'save_period': 10,
                'device': '0',  # GPU 0
                'multi_scale': True,
                'amp': True,  # Mixed precision training
                'cache': False,  # Disabled cache to avoid shm issues
                'persistent_workers': False,  # Disable persistent workers
            },
            'hyperparameters': {
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.15,
                'copy_paste': 0.3,
            },
            'optimization': {
                'optimizer': 'AdamW',  # Better convergence
                'close_mosaic': 10,  # Disable mosaic in last N epochs
            }
        }
    
    def _load_config(self, config_path: str):
        """Load configuration dari YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def train(self, data_yaml: str, project: str = 'runs/train', name: str = None):
        """
        Training YOLOv8 model
        
        Args:
            data_yaml: Path to dataset YAML config
            project: Project directory untuk save results
            name: Experiment name
        """
        if name is None:
            name = f"captcha_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting YOLOv8 Training")
        print(f"{'='*60}")
        print(f"📊 Dataset: {data_yaml}")
        print(f"🎯 Model: {self.config['model']['architecture']}")
        print(f"💪 Device: {self.device}")
        print(f"🔢 Batch Size: {self.config['training']['batch_size']}")
        print(f"🔄 Epochs: {self.config['training']['epochs']}")
        print(f"📏 Input Size: {self.config['model']['input_size']}")
        print(f"{'='*60}\n")
        
        # Initialize model
        model_name = self.config['model']['architecture']
        if self.config['model']['pretrained']:
            model = YOLO(f"{model_name}.pt")
            print(f"✅ Loaded pretrained {model_name} model")
        else:
            model = YOLO(f"{model_name}.yaml")
            print(f"✅ Initialized {model_name} model from scratch")
        
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': self.config['training']['epochs'],
            'imgsz': self.config['model']['input_size'],
            'batch': self.config['training']['batch_size'],
            'workers': self.config['training']['workers'],
            'device': self.config['training']['device'],
            'project': project,
            'name': name,
            'patience': self.config['training']['patience'],
            'save_period': self.config['training']['save_period'],
            'cache': self.config['training']['cache'],
            'amp': self.config['training']['amp'],
            'multi_scale': self.config['training']['multi_scale'],
            'verbose': True,
            'plots': True,
            'save': True,
            'save_json': True,
        }
        
        # Add hyperparameters
        train_args.update(self.config['hyperparameters'])
        train_args.update(self.config['optimization'])
        
        # Start training
        print("🏋️  Training started...\n")
        results = model.train(**train_args)
        
        print(f"\n{'='*60}")
        print(f"✅ Training completed!")
        print(f"📁 Results saved to: {Path(project) / name}")
        print(f"{'='*60}\n")
        
        # Validate best model
        best_model_path = Path(project) / name / 'weights' / 'best.pt'
        if best_model_path.exists():
            print("🔍 Validating best model...")
            best_model = YOLO(str(best_model_path))
            metrics = best_model.val(data=data_yaml)
            
            print(f"\n📊 Best Model Metrics:")
            print(f"   mAP50: {metrics.box.map50:.4f}")
            print(f"   mAP50-95: {metrics.box.map:.4f}")
            print(f"   Precision: {metrics.box.mp:.4f}")
            print(f"   Recall: {metrics.box.mr:.4f}")
        
        return results
    
    def resume_training(self, checkpoint_path: str):
        """Resume training dari checkpoint"""
        print(f"🔄 Resuming training from {checkpoint_path}")
        model = YOLO(checkpoint_path)
        results = model.train(resume=True)
        return results
    
    def export_model(self, model_path: str, formats: list = ['onnx', 'torchscript']):
        """
        Export model ke berbagai format
        
        Args:
            model_path: Path to trained model (.pt)
            formats: List of export formats
        """
        print(f"\n{'='*60}")
        print(f"📦 Exporting model: {model_path}")
        print(f"{'='*60}")
        
        model = YOLO(model_path)
        
        for fmt in formats:
            print(f"\n📤 Exporting to {fmt.upper()}...")
            try:
                export_path = model.export(format=fmt, dynamic=True)
                print(f"   ✅ Exported to: {export_path}")
            except Exception as e:
                print(f"   ❌ Export failed: {str(e)}")
        
        print(f"\n{'='*60}")
        print(f"✅ Export completed!")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for CAPTCHA Detection')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to training config YAML (optional)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    parser.add_argument('--export', type=str, default=None,
                       help='Export model to ONNX/TorchScript')
    
    args = parser.parse_args()
    
    trainer = YOLOv8Trainer(args.config)
    
    if args.export:
        # Export mode
        trainer.export_model(args.export)
    elif args.resume:
        # Resume training
        trainer.resume_training(args.resume)
    else:
        # Normal training
        trainer.train(args.data, args.project, args.name)


if __name__ == "__main__":
    main()
