"""
Coordinate Calculator untuk menghitung posisi drag pada hCaptcha
Menghitung koordinat dari puzzle piece ke target drop zone
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2


class CoordinateCalculator:
    """Kalkulator koordinat untuk drag-based CAPTCHA challenges"""
    
    def __init__(self):
        self.puzzle_class_names = ['puzzle_piece', 'piece', 'drag_object']
        self.target_class_names = ['drop_zone', 'target', 'target_area']
    
    def calculate_drag_coordinates(self,
                                   detections: List[Dict],
                                   image_width: int,
                                   image_height: int,
                                   class_names: List[str]) -> Dict:
        """
        Menghitung koordinat drag dari puzzle piece ke target zone
        
        Args:
            detections: List of detection results [{'class': int, 'conf': float, 'bbox': [x1, y1, x2, y2]}]
            image_width: Lebar image
            image_height: Tinggi image
            class_names: List nama classes
            
        Returns:
            Dict dengan informasi drag coordinates
        """
        # Separate puzzle pieces dan targets
        puzzle_pieces = []
        target_zones = []
        
        for det in detections:
            class_id = det['class']
            class_name = class_names[class_id] if class_id < len(class_names) else ''
            
            if any(pname in class_name.lower() for pname in self.puzzle_class_names):
                puzzle_pieces.append(det)
            elif any(tname in class_name.lower() for tname in self.target_class_names):
                target_zones.append(det)
        
        if not puzzle_pieces:
            return {
                'status': 'no_puzzle_detected',
                'message': 'No puzzle piece detected'
            }
        
        if not target_zones:
            return {
                'status': 'no_target_detected',
                'message': 'No target zone detected'
            }
        
        # Gunakan puzzle piece dengan confidence tertinggi
        puzzle = max(puzzle_pieces, key=lambda x: x['conf'])
        target = max(target_zones, key=lambda x: x['conf'])
        
        # Calculate centers
        puzzle_center = self._get_bbox_center(puzzle['bbox'])
        target_center = self._get_bbox_center(target['bbox'])
        
        # Calculate drag vector
        drag_vector = (
            target_center[0] - puzzle_center[0],
            target_center[1] - puzzle_center[1]
        )
        
        # Calculate distance
        distance = np.sqrt(drag_vector[0]**2 + drag_vector[1]**2)
        
        # Calculate angle (in degrees)
        angle = np.degrees(np.arctan2(drag_vector[1], drag_vector[0]))
        
        result = {
            'status': 'success',
            'puzzle': {
                'bbox': puzzle['bbox'],
                'center': puzzle_center,
                'confidence': puzzle['conf'],
                'class': puzzle['class']
            },
            'target': {
                'bbox': target['bbox'],
                'center': target_center,
                'confidence': target['conf'],
                'class': target['class']
            },
            'drag': {
                'from': puzzle_center,
                'to': target_center,
                'vector': drag_vector,
                'distance': distance,
                'angle': angle
            },
            'actions': self._generate_drag_actions(puzzle_center, target_center)
        }
        
        return result
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point dari bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)
    
    def _generate_drag_actions(self, 
                               start: Tuple[float, float],
                               end: Tuple[float, float],
                               steps: int = 10) -> List[Dict]:
        """
        Generate intermediate drag actions untuk smooth drag
        
        Args:
            start: Starting coordinates
            end: Ending coordinates
            steps: Number of intermediate steps
            
        Returns:
            List of action dicts with coordinates dan timing
        """
        actions = []
        
        # Mouse down at start
        actions.append({
            'action': 'mousedown',
            'x': int(start[0]),
            'y': int(start[1]),
            'delay_ms': 100
        })
        
        # Intermediate drag steps
        for i in range(1, steps):
            t = i / steps
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            
            actions.append({
                'action': 'mousemove',
                'x': int(x),
                'y': int(y),
                'delay_ms': 50
            })
        
        # Mouse up at end
        actions.append({
            'action': 'mousemove',
            'x': int(end[0]),
            'y': int(end[1]),
            'delay_ms': 50
        })
        
        actions.append({
            'action': 'mouseup',
            'x': int(end[0]),
            'y': int(end[1]),
            'delay_ms': 100
        })
        
        return actions
    
    def visualize_drag_path(self,
                           image: np.ndarray,
                           drag_result: Dict,
                           output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualisasi drag path pada image
        
        Args:
            image: Input image (BGR)
            drag_result: Result dari calculate_drag_coordinates
            output_path: Optional path untuk save image
            
        Returns:
            Annotated image
        """
        if drag_result['status'] != 'success':
            return image
        
        vis_img = image.copy()
        
        # Draw puzzle bounding box (green)
        puzzle_bbox = drag_result['puzzle']['bbox']
        cv2.rectangle(vis_img,
                     (int(puzzle_bbox[0]), int(puzzle_bbox[1])),
                     (int(puzzle_bbox[2]), int(puzzle_bbox[3])),
                     (0, 255, 0), 3)
        
        # Draw target bounding box (blue)
        target_bbox = drag_result['target']['bbox']
        cv2.rectangle(vis_img,
                     (int(target_bbox[0]), int(target_bbox[1])),
                     (int(target_bbox[2]), int(target_bbox[3])),
                     (255, 0, 0), 3)
        
        # Draw centers
        puzzle_center = drag_result['puzzle']['center']
        target_center = drag_result['target']['center']
        
        cv2.circle(vis_img, 
                  (int(puzzle_center[0]), int(puzzle_center[1])),
                  8, (0, 255, 0), -1)
        cv2.circle(vis_img,
                  (int(target_center[0]), int(target_center[1])),
                  8, (255, 0, 0), -1)
        
        # Draw drag arrow (red)
        cv2.arrowedLine(vis_img,
                       (int(puzzle_center[0]), int(puzzle_center[1])),
                       (int(target_center[0]), int(target_center[1])),
                       (0, 0, 255), 3, tipLength=0.3)
        
        # Add text info
        info_text = f"Distance: {drag_result['drag']['distance']:.1f}px"
        cv2.putText(vis_img, info_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)
        
        angle_text = f"Angle: {drag_result['drag']['angle']:.1f}°"
        cv2.putText(vis_img, angle_text,
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)
        
        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, vis_img)
        
        return vis_img
    
    def calculate_multiple_drags(self,
                                detections: List[Dict],
                                image_width: int,
                                image_height: int,
                                class_names: List[str]) -> List[Dict]:
        """
        Menghitung multiple drag operations jika ada banyak puzzle pieces
        
        Returns:
            List of drag results
        """
        # Separate puzzle pieces dan targets
        puzzle_pieces = []
        target_zones = []
        
        for det in detections:
            class_id = det['class']
            class_name = class_names[class_id] if class_id < len(class_names) else ''
            
            if any(pname in class_name.lower() for pname in self.puzzle_class_names):
                puzzle_pieces.append(det)
            elif any(tname in class_name.lower() for tname in self.target_class_names):
                target_zones.append(det)
        
        results = []
        
        # Match each puzzle to nearest target
        for puzzle in puzzle_pieces:
            puzzle_center = self._get_bbox_center(puzzle['bbox'])
            
            # Find nearest target
            nearest_target = None
            min_distance = float('inf')
            
            for target in target_zones:
                target_center = self._get_bbox_center(target['bbox'])
                distance = np.sqrt(
                    (target_center[0] - puzzle_center[0])**2 +
                    (target_center[1] - puzzle_center[1])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_target = target
            
            if nearest_target:
                # Create mock detections for single calculation
                single_result = self.calculate_drag_coordinates(
                    [puzzle, nearest_target],
                    image_width,
                    image_height,
                    class_names
                )
                
                if single_result['status'] == 'success':
                    results.append(single_result)
        
        return results


if __name__ == "__main__":
    print("Coordinate Calculator for Drag-based CAPTCHA")
    print("=" * 60)
    
    # Example usage
    calculator = CoordinateCalculator()
    
    # Mock detections
    mock_detections = [
        {'class': 0, 'conf': 0.95, 'bbox': [100, 150, 200, 250]},  # puzzle
        {'class': 1, 'conf': 0.92, 'bbox': [400, 180, 500, 280]},  # target
    ]
    
    class_names = ['puzzle_piece', 'drop_zone']
    
    result = calculator.calculate_drag_coordinates(
        mock_detections,
        image_width=800,
        image_height=600,
        class_names=class_names
    )
    
    print("\n📊 Drag Calculation Result:")
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"From: {result['drag']['from']}")
        print(f"To: {result['drag']['to']}")
        print(f"Distance: {result['drag']['distance']:.1f}px")
        print(f"Angle: {result['drag']['angle']:.1f}°")
        print(f"\nActions: {len(result['actions'])} steps")
