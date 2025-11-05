"""
Inference Script untuk YOLOv8 - CAPTCHA Detection & Solving
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json
from typing import List, Dict, Union, Optional
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.coordinate_calculator import CoordinateCalculator


class CaptchaSolver:
    """Inference engine untuk mendeteksi dan solve CAPTCHA"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize CAPTCHA solver
        
        Args:
            model_path: Path to trained YOLOv8 model (.pt or .onnx)
            conf_threshold: Confidence threshold untuk deteksi
            iou_threshold: IoU threshold untuk NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        print(f"📦 Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Classes: {len(self.class_names)}")
        print(f"   {list(self.class_names.values())}")
        
        # Initialize coordinate calculator
        self.coord_calculator = CoordinateCalculator()
    
    def predict(self, image: Union[str, np.ndarray], 
                visualize: bool = False,
                save_path: Optional[str] = None) -> Dict:
        """
        Run inference pada image
        
        Args:
            image: Image path atau numpy array
            visualize: Visualize hasil deteksi
            save_path: Path untuk save hasil visualisasi
            
        Returns:
            Dict dengan detection results dan drag coordinates
        """
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Parse results
        detections = []
        for box in results.boxes:
            det = {
                'class': int(box.cls[0]),
                'class_name': self.class_names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].cpu().numpy().tolist()
            }
            detections.append(det)
        
        # Get image dimensions
        img = results.orig_img
        height, width = img.shape[:2]
        
        # Calculate drag coordinates
        drag_result = None
        if detections:
            drag_result = self.coord_calculator.calculate_drag_coordinates(
                detections,
                width,
                height,
                list(self.class_names.values())
            )
        
        # Prepare output
        output = {
            'image_size': {'width': width, 'height': height},
            'detections': detections,
            'num_detections': len(detections),
            'drag_solution': drag_result
        }
        
        # Visualize if requested
        if visualize and drag_result and drag_result['status'] == 'success':
            vis_img = self.coord_calculator.visualize_drag_path(
                img, drag_result, save_path
            )
            
            if save_path is None:
                # Display
                cv2.imshow('CAPTCHA Solution', vis_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return output
    
    def predict_batch(self, image_paths: List[str], 
                     output_dir: Optional[str] = None) -> List[Dict]:
        """
        Batch inference pada multiple images
        
        Args:
            image_paths: List of image paths
            output_dir: Directory untuk save results
            
        Returns:
            List of prediction results
        """
        results = []
        
        for img_path in image_paths:
            print(f"\n🔍 Processing: {img_path}")
            
            save_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = Path(img_path).stem + '_solved.jpg'
                save_path = os.path.join(output_dir, filename)
            
            result = self.predict(img_path, visualize=True, save_path=save_path)
            result['image_path'] = img_path
            results.append(result)
            
            # Print summary
            print(f"   Detections: {result['num_detections']}")
            if result['drag_solution']:
                print(f"   Drag Status: {result['drag_solution']['status']}")
                if result['drag_solution']['status'] == 'success':
                    drag = result['drag_solution']['drag']
                    print(f"   Distance: {drag['distance']:.1f}px")
                    print(f"   Angle: {drag['angle']:.1f}°")
        
        return results
    
    def solve_captcha(self, image: Union[str, np.ndarray]) -> Dict:
        """
        High-level function untuk solve CAPTCHA
        
        Returns:
            Dict dengan solution actions
        """
        result = self.predict(image, visualize=False)
        
        if not result['drag_solution']:
            return {
                'solved': False,
                'message': 'No objects detected'
            }
        
        drag_solution = result['drag_solution']
        
        if drag_solution['status'] != 'success':
            return {
                'solved': False,
                'message': drag_solution.get('message', 'Cannot solve')
            }
        
        return {
            'solved': True,
            'actions': drag_solution['actions'],
            'drag_info': {
                'from': drag_solution['drag']['from'],
                'to': drag_solution['drag']['to'],
                'distance': drag_solution['drag']['distance'],
                'angle': drag_solution['drag']['angle']
            }
        }
    
    def export_results(self, results: List[Dict], output_path: str):
        """Export results ke JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 CAPTCHA Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt or .onnx)')
    parser.add_argument('--source', type=str, required=True,
                       help='Image path or directory')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default='output/predictions',
                       help='Output directory for results')
    parser.add_argument('--save-json', action='store_true',
                       help='Save results to JSON')
    
    args = parser.parse_args()
    
    # Initialize solver
    solver = CaptchaSolver(args.model, args.conf, args.iou)
    
    # Check if source is directory or file
    source_path = Path(args.source)
    
    if source_path.is_dir():
        # Batch processing
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(list(source_path.glob(ext)))
        
        print(f"\n📁 Found {len(image_paths)} images")
        results = solver.predict_batch([str(p) for p in image_paths], args.output)
        
    else:
        # Single image
        results = [solver.predict(str(source_path), visualize=True, 
                                  save_path=f"{args.output}/{source_path.stem}_solved.jpg")]
    
    # Save JSON if requested
    if args.save_json:
        json_path = f"{args.output}/results.json"
        solver.export_results(results, json_path)
    
    print(f"\n✅ Inference completed!")
    print(f"📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
