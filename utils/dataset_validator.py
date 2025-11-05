"""
Dataset Validator untuk validasi struktur dan format dataset YOLO
"""

import os
from pathlib import Path
import yaml
from typing import Dict, List
import cv2


class DatasetValidator:
    """Validator untuk YOLO dataset"""
    
    def __init__(self, dataset_yaml: str):
        self.dataset_yaml = Path(dataset_yaml)
        self.config = self._load_config()
        self.errors = []
        self.warnings = []
    
    def _load_config(self) -> Dict:
        """Load dataset YAML configuration"""
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.dataset_yaml}")
        
        with open(self.dataset_yaml, 'r') as f:
            return yaml.safe_load(f)
    
    def validate(self) -> bool:
        """Run complete validation"""
        print(f"🔍 Validating dataset: {self.dataset_yaml}")
        print("=" * 60)
        
        # Validate YAML structure
        self._validate_yaml_structure()
        
        # Validate paths
        self._validate_paths()
        
        # Validate images and labels
        self._validate_data()
        
        # Print results
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_yaml_structure(self):
        """Validate YAML file structure"""
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        
        for key in required_keys:
            if key not in self.config:
                self.errors.append(f"Missing required key in YAML: {key}")
        
        # Validate class count
        if 'nc' in self.config and 'names' in self.config:
            if self.config['nc'] != len(self.config['names']):
                self.errors.append(
                    f"Class count mismatch: nc={self.config['nc']}, "
                    f"but {len(self.config['names'])} names provided"
                )
    
    def _validate_paths(self):
        """Validate dataset paths"""
        base_path = Path(self.config.get('path', '.'))
        
        for split in ['train', 'val', 'test']:
            if split in self.config:
                split_path = base_path / self.config[split]
                if not split_path.exists():
                    self.errors.append(f"{split} path does not exist: {split_path}")
    
    def _validate_data(self):
        """Validate images and annotations"""
        base_path = Path(self.config.get('path', '.'))
        
        for split in ['train', 'val', 'test']:
            if split not in self.config:
                continue
            
            print(f"\n📁 Validating {split} split...")
            
            img_dir = base_path / self.config[split]
            label_dir = img_dir.parent.parent / 'labels' / split
            
            if not label_dir.exists():
                label_dir = img_dir.parent / 'labels' / split
            
            # Count files
            img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            label_files = list(label_dir.glob('*.txt')) if label_dir.exists() else []
            
            print(f"   Images: {len(img_files)}")
            print(f"   Labels: {len(label_files)}")
            
            # Check for missing labels
            missing_labels = 0
            for img_path in img_files[:100]:  # Sample first 100
                label_path = label_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    missing_labels += 1
            
            if missing_labels > 0:
                self.warnings.append(
                    f"{split}: {missing_labels}/100 sampled images have missing labels"
                )
            
            # Validate label format
            invalid_labels = 0
            for label_path in list(label_files)[:100]:  # Sample first 100
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                invalid_labels += 1
                                break
                            
                            # Validate class ID
                            class_id = int(parts[0])
                            if class_id >= self.config['nc']:
                                invalid_labels += 1
                                break
                            
                            # Validate coordinates
                            coords = [float(x) for x in parts[1:5]]
                            if not all(0 <= c <= 1 for c in coords):
                                invalid_labels += 1
                                break
                except Exception as e:
                    invalid_labels += 1
            
            if invalid_labels > 0:
                self.errors.append(
                    f"{split}: {invalid_labels}/100 sampled labels have invalid format"
                )
    
    def _print_results(self):
        """Print validation results"""
        print("\n" + "=" * 60)
        print("📊 Validation Results")
        print("=" * 60)
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   - {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ Dataset is valid!")
        elif not self.errors:
            print("\n✅ Dataset is valid (with warnings)")
        else:
            print("\n❌ Dataset validation failed")
        
        print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_validator.py <dataset.yaml>")
        sys.exit(1)
    
    validator = DatasetValidator(sys.argv[1])
    is_valid = validator.validate()
    sys.exit(0 if is_valid else 1)
