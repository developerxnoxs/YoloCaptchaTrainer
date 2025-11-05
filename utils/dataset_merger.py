"""
Dataset Merger untuk menggabungkan dataset reCAPTCHA v2 dan hCaptcha
Format: YOLO (txt annotations)
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import List, Dict
import random
from tqdm import tqdm


class DatasetMerger:
    """Menggabungkan multiple datasets dalam format YOLO"""
    
    def __init__(self, output_dir: str = "datasets/merged"):
        self.output_dir = Path(output_dir)
        self.class_mapping = {}
        self.global_classes = []
        
    def merge_datasets(self, 
                      dataset_paths: List[str],
                      split_ratios: Dict[str, float] = {'train': 0.7, 'val': 0.2, 'test': 0.1},
                      shuffle: bool = True):
        """
        Menggabungkan multiple datasets
        
        Args:
            dataset_paths: List path ke dataset (format YOLO)
            split_ratios: Rasio pembagian train/val/test
            shuffle: Shuffle data sebelum split
        """
        print(f"🔄 Merging {len(dataset_paths)} datasets...")
        
        # Collect all images and annotations
        all_data = []
        
        for dataset_path in dataset_paths:
            dataset_path = Path(dataset_path)
            dataset_name = dataset_path.name
            
            # Read dataset.yaml
            yaml_path = dataset_path / "data.yaml"
            if not yaml_path.exists():
                yaml_path = dataset_path / "dataset.yaml"
            
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    dataset_config = yaml.safe_load(f)
                    dataset_classes = dataset_config.get('names', [])
            else:
                print(f"⚠️  Warning: No YAML config found for {dataset_name}, using default classes")
                dataset_classes = []
            
            # Update global class mapping
            self._update_class_mapping(dataset_name, dataset_classes)
            
            # Collect images and annotations
            for split in ['train', 'val', 'test', 'images', 'labels']:
                split_dir = dataset_path / split
                if not split_dir.exists():
                    continue
                
                # Find images
                image_dir = split_dir if split == 'images' else split_dir / 'images'
                label_dir = split_dir if split == 'labels' else split_dir / 'labels'
                
                if not image_dir.exists():
                    image_dir = split_dir
                if not label_dir.exists():
                    label_dir = split_dir.parent / 'labels' / split
                
                for img_path in image_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        label_path = label_dir / f"{img_path.stem}.txt"
                        if label_path.exists():
                            all_data.append({
                                'image': img_path,
                                'label': label_path,
                                'dataset': dataset_name
                            })
        
        print(f"📊 Total images collected: {len(all_data)}")
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(all_data)
        
        # Split data
        total = len(all_data)
        train_size = int(total * split_ratios['train'])
        val_size = int(total * split_ratios['val'])
        
        splits = {
            'train': all_data[:train_size],
            'val': all_data[train_size:train_size + val_size],
            'test': all_data[train_size + val_size:]
        }
        
        # Copy files to output directory
        for split_name, split_data in splits.items():
            print(f"\n📁 Processing {split_name} split ({len(split_data)} images)...")
            
            img_dir = self.output_dir / 'images' / split_name
            lbl_dir = self.output_dir / 'labels' / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            
            for item in tqdm(split_data, desc=f"Copying {split_name}"):
                # Copy image
                img_dest = img_dir / f"{item['dataset']}_{item['image'].name}"
                shutil.copy2(item['image'], img_dest)
                
                # Convert and copy label
                lbl_dest = lbl_dir / f"{item['dataset']}_{item['image'].stem}.txt"
                self._convert_labels(item['label'], lbl_dest, item['dataset'])
        
        # Create dataset.yaml
        self._create_yaml_config()
        
        print(f"\n✅ Dataset merged successfully!")
        print(f"📍 Output directory: {self.output_dir}")
        print(f"📊 Classes: {len(self.global_classes)}")
        print(f"   {self.global_classes}")
        
        return str(self.output_dir / "dataset.yaml")
    
    def _update_class_mapping(self, dataset_name: str, classes: List[str]):
        """Update global class mapping"""
        dataset_mapping = {}
        
        for local_idx, class_name in enumerate(classes):
            if class_name not in self.global_classes:
                self.global_classes.append(class_name)
            
            global_idx = self.global_classes.index(class_name)
            dataset_mapping[local_idx] = global_idx
        
        self.class_mapping[dataset_name] = dataset_mapping
    
    def _convert_labels(self, src_label: Path, dst_label: Path, dataset_name: str):
        """Convert labels to global class indices"""
        with open(src_label, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                local_class_id = int(parts[0])
                # Map to global class ID
                if dataset_name in self.class_mapping:
                    global_class_id = self.class_mapping[dataset_name].get(local_class_id, local_class_id)
                else:
                    global_class_id = local_class_id
                
                # Reconstruct line with global class ID
                converted_lines.append(f"{global_class_id} {' '.join(parts[1:])}\n")
        
        with open(dst_label, 'w') as f:
            f.writelines(converted_lines)
    
    def _create_yaml_config(self):
        """Create YOLO dataset configuration file"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.global_classes),
            'names': self.global_classes
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n📄 Dataset config saved to: {yaml_path}")


def merge_recaptcha_hcaptcha_datasets(
    recaptcha_paths: List[str],
    hcaptcha_paths: List[str],
    output_dir: str = "datasets/merged_captcha"
):
    """
    Convenience function untuk merge dataset reCAPTCHA dan hCaptcha
    
    Args:
        recaptcha_paths: List of reCAPTCHA v2 dataset paths
        hcaptcha_paths: List of hCaptcha dataset paths
        output_dir: Output directory untuk merged dataset
    """
    all_paths = recaptcha_paths + hcaptcha_paths
    
    merger = DatasetMerger(output_dir)
    config_path = merger.merge_datasets(
        dataset_paths=all_paths,
        split_ratios={'train': 0.7, 'val': 0.2, 'test': 0.1},
        shuffle=True
    )
    
    return config_path


if __name__ == "__main__":
    # Example usage
    print("Dataset Merger for YOLOv8 - reCAPTCHA v2 & hCaptcha")
    print("=" * 60)
    
    # Example paths (sesuaikan dengan lokasi dataset Anda)
    recaptcha_datasets = [
        # "path/to/recaptcha_dataset_1",
        # "path/to/recaptcha_dataset_2",
    ]
    
    hcaptcha_datasets = [
        # "path/to/hcaptcha_dataset_1",
        # "path/to/hcaptcha_dataset_2",
    ]
    
    if recaptcha_datasets or hcaptcha_datasets:
        config_path = merge_recaptcha_hcaptcha_datasets(
            recaptcha_datasets,
            hcaptcha_datasets,
            output_dir="datasets/merged_captcha"
        )
        print(f"\n✅ Merged dataset config: {config_path}")
    else:
        print("\n⚠️  No dataset paths specified.")
        print("Edit the script and add your dataset paths.")
