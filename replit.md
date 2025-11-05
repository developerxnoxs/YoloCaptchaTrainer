# YOLOv8 CAPTCHA Solver

## Overview

A Python-based machine learning system that trains YOLOv8 models to detect and solve various CAPTCHA challenges including reCAPTCHA v2, hCaptcha, and drag-based puzzle CAPTCHAs. The system is optimized for high-performance GPU training (specifically A100 80GB) and provides a complete pipeline from dataset preparation through model training to real-time inference.

The application enables automated CAPTCHA solving by:
1. Detecting objects within CAPTCHA images using computer vision
2. Calculating drag coordinates for puzzle-based challenges
3. Merging multiple CAPTCHA datasets for comprehensive training
4. Exporting trained models for deployment (ONNX, TorchScript)

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Machine Learning Pipeline

**Model Framework**: YOLOv8 (Ultralytics implementation)
- Uses the largest YOLOv8x architecture for maximum accuracy
- Implements mixed precision training (AMP) for GPU efficiency
- Supports multi-scale training for robust object detection
- Configurable for various GPU configurations, optimized for A100 80GB

**Training Strategy**:
- Large batch sizes (64) leveraging GPU memory capacity
- Extended training epochs (300) with early stopping (patience: 50)
- Multi-class detection supporting various CAPTCHA object types
- Disk caching for efficient data loading
- Automatic model checkpointing every 10 epochs

### Data Processing Architecture

**Dataset Management**:
- **DatasetMerger**: Combines multiple CAPTCHA datasets while maintaining YOLO format consistency
- Handles class mapping across different dataset sources
- Implements train/validation/test splitting (70/20/10 default)
- Supports shuffling and stratified sampling

**Dataset Sources**:
- Roboflow API integration for automated dataset downloads
- Support for reCAPTCHA v2, hCaptcha, and drag puzzle datasets
- Automated download and merging pipeline
- Dataset validation system to ensure format correctness

**Data Validation**:
- YAML structure validation for YOLO format compliance
- Image-label pair verification
- Path existence checking
- Format consistency validation

### Inference Architecture

**CaptchaSolver Engine**:
- Real-time object detection on CAPTCHA images
- Configurable confidence and IoU thresholds
- Support for both .pt (PyTorch) and .onnx model formats
- Visualization capabilities with bounding boxes

**Coordinate Calculation System**:
- Identifies puzzle pieces and target drop zones
- Calculates optimal drag paths for puzzle-based CAPTCHAs
- Provides centroid-to-centroid coordinate mapping
- Returns actionable coordinate data for automation scripts

### Code Organization

**Modular Structure**:
- `scripts/`: Core execution scripts (training, inference, dataset download)
- `utils/`: Reusable utility modules (dataset operations, coordinate math)
- `models/`: Storage for trained model artifacts
- `datasets/`: Dataset storage with organized splits
- `output/`: Inference results and visualizations
- `logs/`: Training metrics and logs

**Key Design Patterns**:
- Class-based architecture for trainers and solvers
- Configuration-driven design (YAML configs, default parameters)
- Path-based file management using pathlib
- Separation of concerns (data/model/inference layers)

## External Dependencies

### Deep Learning Framework
- **PyTorch** (>=2.0.0): Core deep learning framework
- **torchvision** (>=0.15.0): Computer vision utilities
- **Ultralytics** (>=8.0.0): YOLOv8 implementation and training pipeline

### Computer Vision & Image Processing
- **OpenCV** (>=4.8.0): Image processing and manipulation
- **Pillow** (>=10.0.0): Image loading and basic operations
- **albumentations** (>=1.3.0): Advanced image augmentation

### Data Science & Utilities
- **NumPy** (>=1.24.0): Numerical computing
- **pandas** (>=2.0.0): Data manipulation
- **scikit-learn** (>=1.3.0): ML utilities and metrics
- **PyYAML** (>=6.0): Configuration file handling

### Visualization & Monitoring
- **matplotlib** (>=3.7.0): Plotting and visualization
- **seaborn** (>=0.12.0): Statistical visualizations
- **tqdm** (>=4.65.0): Progress bars

### Model Export & Deployment
- **ONNX** (>=1.14.0): Model export format
- **onnxruntime-gpu** (>=1.15.0): GPU-accelerated ONNX inference

### Dataset Management
- **Roboflow API**: Automated dataset downloading from Roboflow platform
  - Requires API key from https://roboflow.com
  - Free tier: 10,000 API calls/month
  - Used for accessing pre-labeled CAPTCHA datasets

### GPU Requirements
- CUDA-compatible GPU (optimized for NVIDIA A100 80GB)
- CUDA toolkit and cuDNN (installed via PyTorch)
- Supports multi-GPU setups through PyTorch DDP