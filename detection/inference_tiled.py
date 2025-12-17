"""
Contribution #2: Tiling Inference for Tiny Logos
=================================================
Improves detection of extremely small logos by:
- Splitting images into overlapping tiles
- Running detector on each tile independently
- Fusing predictions with global NMS

No retraining required - uses existing model weights.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch


class TiledInference:
    """
    Tiling-based inference for improved small object detection.
    
    Args:
        model_path: Path to trained YOLO model weights
        tile_size: Size of each tile (default: 1024)
        overlap: Overlap ratio between tiles (default: 0.2 = 20%)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for final NMS fusion
    """
    
    def __init__(
        self,
        model_path: str,
        tile_size: int = 1024,
        overlap: float = 0.2,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.stride = int(tile_size * (1 - overlap))
        
    def create_tiles(self, image: np.ndarray) -> List[Dict]:
        """
        Split image into overlapping tiles.
        
        Returns:
            List of dicts with 'image' (tile), 'x_offset', 'y_offset'
        """
        h, w = image.shape[:2]
        tiles = []
        
        y_positions = list(range(0, h - self.tile_size + 1, self.stride))
        if y_positions[-1] + self.tile_size < h:
            y_positions.append(h - self.tile_size)
            
        x_positions = list(range(0, w - self.tile_size + 1, self.stride))
        if x_positions[-1] + self.tile_size < w:
            x_positions.append(w - self.tile_size)
        
        for y in y_positions:
            for x in x_positions:
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append({
                    'image': tile,
                    'x_offset': x,
                    'y_offset': y
                })
        
        return tiles
    
    def remap_detections(
        self,
        tile_detections: Results,
        x_offset: int,
        y_offset: int
    ) -> np.ndarray:
        """
        Remap tile detections to original image coordinates.
        
        Returns:
            Array of shape (N, 6) with [x1, y1, x2, y2, conf, class]
        """
        if tile_detections.boxes is None or len(tile_detections.boxes) == 0:
            return np.array([]).reshape(0, 6)
        
        boxes = tile_detections.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confs = tile_detections.boxes.conf.cpu().numpy()
        classes = tile_detections.boxes.cls.cpu().numpy()
        
        # Remap coordinates to original image
        boxes[:, [0, 2]] += x_offset  # x coordinates
        boxes[:, [1, 3]] += y_offset  # y coordinates
        
        # Stack into [x1, y1, x2, y2, conf, class]
        detections = np.column_stack([boxes, confs, classes])
        
        return detections
    
    def global_nms(self, all_detections: np.ndarray) -> np.ndarray:
        """
        Apply global Non-Maximum Suppression across all tiles.
        
        Args:
            all_detections: Array of shape (N, 6) with [x1, y1, x2, y2, conf, class]
            
        Returns:
            Filtered detections after NMS
        """
        if len(all_detections) == 0:
            return all_detections
        
        # Group by class
        unique_classes = np.unique(all_detections[:, 5])
        keep_detections = []
        
        for cls in unique_classes:
            cls_mask = all_detections[:, 5] == cls
            cls_detections = all_detections[cls_mask]
            
            # Convert to tensor for torchvision NMS
            boxes = torch.tensor(cls_detections[:, :4], dtype=torch.float32)
            scores = torch.tensor(cls_detections[:, 4], dtype=torch.float32)
            
            # Apply NMS
            keep_indices = torch.ops.torchvision.nms(
                boxes, scores, self.iou_threshold
            )
            
            keep_detections.append(cls_detections[keep_indices.numpy()])
        
        if len(keep_detections) == 0:
            return np.array([]).reshape(0, 6)
            
        return np.vstack(keep_detections)
    
    def predict_tiled(self, image_path: str, verbose: bool = False) -> Dict:
        """
        Run tiled inference on a single image.
        
        Returns:
            Dict with 'detections', 'n_tiles', 'n_detections_before_nms', 'n_detections_after_nms'
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = image.shape[:2]
        
        # Handle small images (no tiling needed)
        if h <= self.tile_size and w <= self.tile_size:
            if verbose:
                print(f"Image {Path(image_path).name}: {w}x{h} - using single-pass inference")
            results = self.model.predict(
                image_path,
                conf=self.conf_threshold,
                verbose=False
            )[0]
            
            if results.boxes is None or len(results.boxes) == 0:
                detections = np.array([]).reshape(0, 6)
            else:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                detections = np.column_stack([boxes, confs, classes])
            
            return {
                'detections': detections,
                'n_tiles': 1,
                'n_detections_before_nms': len(detections),
                'n_detections_after_nms': len(detections)
            }
        
        # Create tiles
        tiles = self.create_tiles(image)
        if verbose:
            print(f"Image {Path(image_path).name}: {w}x{h} - split into {len(tiles)} tiles")
        
        # Run detection on each tile
        all_detections = []
        for tile_info in tiles:
            tile_results = self.model.predict(
                tile_info['image'],
                conf=self.conf_threshold,
                verbose=False
            )[0]
            
            # Remap to original coordinates
            remapped = self.remap_detections(
                tile_results,
                tile_info['x_offset'],
                tile_info['y_offset']
            )
            
            if len(remapped) > 0:
                all_detections.append(remapped)
        
        # Combine all detections
        if len(all_detections) == 0:
            all_detections = np.array([]).reshape(0, 6)
        else:
            all_detections = np.vstack(all_detections)
        
        n_before_nms = len(all_detections)
        
        # Apply global NMS
        final_detections = self.global_nms(all_detections)
        n_after_nms = len(final_detections)
        
        if verbose:
            print(f"  Detections: {n_before_nms} → {n_after_nms} (after NMS)")
        
        return {
            'detections': final_detections,
            'n_tiles': len(tiles),
            'n_detections_before_nms': n_before_nms,
            'n_detections_after_nms': n_after_nms
        }
    
    def visualize_detection(
        self,
        image_path: str,
        detections: np.ndarray,
        output_path: str = None,
        class_names: List[str] = None
    ):
        """
        Visualize detections on the image.
        
        Args:
            image_path: Path to original image
            detections: Array of shape (N, 6) with [x1, y1, x2, y2, conf, class]
            output_path: Where to save visualization (optional)
            class_names: List of class names (optional)
        """
        image = cv2.imread(str(image_path))
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            if class_names and cls < len(class_names):
                label = f"{class_names[cls]} {conf:.2f}"
            else:
                label = f"Class {cls} {conf:.2f}"
            
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Saved visualization to {output_path}")
        else:
            cv2.imshow("Tiled Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tiled inference for YOLO')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--image', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--tile-size', type=int, default=1024, help='Tile size')
    parser.add_argument('--overlap', type=float, default=0.2, help='Overlap ratio (0-1)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--output', type=str, help='Output visualization path (for single image) or directory (for multiple images)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Run tiled inference
    tiler = TiledInference(
        model_path=args.model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Check if input is a file or directory
    input_path = Path(args.image)
    
    if input_path.is_file():
        # Single image
        result = tiler.predict_tiled(str(input_path), verbose=True)
        
        print(f"\nResults:")
        print(f"  Tiles created: {result['n_tiles']}")
        print(f"  Detections before NMS: {result['n_detections_before_nms']}")
        print(f"  Final detections: {result['n_detections_after_nms']}")
        
        # Visualize if output path provided
        if args.output:
            tiler.visualize_detection(str(input_path), result['detections'], args.output)
    
    elif input_path.is_dir():
        # Multiple images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in directory: {input_path}")
            return
        
        print(f"Processing {len(image_files)} images from {input_path}")
        
        # Create output directory if specified
        output_dir = None
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        total_detections = 0
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {image_file.name}")
            result = tiler.predict_tiled(str(image_file), verbose=args.verbose)
            
            print(f"  Tiles: {result['n_tiles']}, Detections: {result['n_detections_after_nms']}")
            total_detections += result['n_detections_after_nms']
            
            # Visualize if output directory provided
            if output_dir:
                output_file = output_dir / f"{image_file.stem}_tiled{image_file.suffix}"
                tiler.visualize_detection(str(image_file), result['detections'], str(output_file))
        
        print(f"\n{'='*60}")
        print(f"Processed {len(image_files)} images")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections/len(image_files):.2f}")
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return


if __name__ == '__main__':
    main()
