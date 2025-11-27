"""
Complete Logo Detection + Recognition Pipeline

Combines YOLOv8 detection with ResNet50 classification.

Usage:
    from pipeline.logo_pipeline import LogoDetectionRecognitionPipeline
    
    pipeline = LogoDetectionRecognitionPipeline(
        yolo_model_path='detection/runs/detect/.../weights/best.pt',
        resnet_model_path='recognition/weights/model_for_inference.pth'
    )
    
    results, image = pipeline.detect_and_recognize('test_image.jpg')
    pipeline.visualize_results(image, results)
"""

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from recognition.model import LogoRecognitionModel


class LogoDetectionRecognitionPipeline:
    """
    Two-stage pipeline:
    1. YOLOv8: Detect logo bounding boxes
    2. ResNet50: Classify each detected logo
    """
    
    def __init__(
        self,
        yolo_model_path='detection/runs/detect/logodet3k_yolov8s_baseline50/weights/best.pt',
        resnet_model_path='recognition/weights/model_for_inference.pth',
        device='cuda',
        brand_mapping_path='LogoDet-3K/annotations.json'
    ):
        """
        Initialize the pipeline with trained models.
        
        Args:
            yolo_model_path: Path to YOLOv8 detection weights
            resnet_model_path: Path to ResNet50 classification checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print("="*70)
        print("🚀 INITIALIZING LOGO DETECTION + RECOGNITION PIPELINE")
        print("="*70)
        
        # Load YOLOv8 Detection Model
        print("\n📦 Loading YOLOv8 detection model...")
        self.yolo_model = YOLO(yolo_model_path)
        print(f"✓ YOLOv8 loaded from: {yolo_model_path}")
        
        # Load ResNet50 Classification Model
        print("\n📦 Loading ResNet50 classification model...")
        checkpoint = torch.load(resnet_model_path, map_location=self.device, weights_only=False)
        
        # Get config and mappings
        self.num_classes = checkpoint['config']['num_classes']
        self.idx_to_brand = checkpoint['idx_to_brand']
        self.brand_to_idx = checkpoint['brand_to_idx']
        
        # Initialize model
        self.resnet_model = LogoRecognitionModel(
            num_classes=self.num_classes,
            pretrained=False
        )
        self.resnet_model.load_state_dict(checkpoint['model_state_dict'])
        self.resnet_model = self.resnet_model.to(self.device)
        self.resnet_model.eval()
        
        print(f"✓ ResNet50 loaded from: {resnet_model_path}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Test Accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
        
        # Image preprocessing for ResNet
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("\n✅ Pipeline ready!")
        print("="*70 + "\n")
    
    def detect_and_recognize(self, image_path, conf_threshold=0.25, top_k=5):
        """
        Run complete pipeline on an image.
        
        Args:
            image_path: Path to input image
            conf_threshold: YOLOv8 confidence threshold (0-1)
            top_k: Number of top brand predictions to return
            
        Returns:
            results: List of detections with classifications
            image_rgb: Original image in RGB format
        """
        # Read image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1: Run YOLO Detection
        print(f"🔍 Running YOLO detection on: {Path(image_path).name}")
        yolo_results = self.yolo_model(image_path, conf=conf_threshold, verbose=False)[0]
        
        boxes = yolo_results.boxes
        print(f"✓ Found {len(boxes)} logo detections")
        
        # Step 2: Classify each detected logo
        results = []
        
        with torch.no_grad():
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                
                # Crop logo region
                logo_crop = image_rgb[y1:y2, x1:x2]
                
                if logo_crop.size == 0:
                    continue
                
                # Preprocess for ResNet
                logo_pil = Image.fromarray(logo_crop)
                logo_tensor = self.transform(logo_pil).unsqueeze(0).to(self.device)
                
                # Get classification
                logits, embeddings = self.resnet_model(logo_tensor)
                probs = F.softmax(logits, dim=1)[0]
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes))
                
                predictions = []
                for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                    brand_name = self.get_brand_name(idx)
                    predictions.append({
                        'brand': brand_name,
                        'confidence': float(prob),
                        'class_id': int(idx)
                    })
                
                results.append({
                    'detection_id': i + 1,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'detection_conf': float(conf),
                    'predictions': predictions
                })
        
        return results, image_rgb
    
    def get_brand_name(self, class_idx):
        """
        Args:
            class_idx: Class index (int or tensor)
        
        Returns:
            str: Brand name
        """
        idx = int(class_idx)
        
        # Try both int and str keys
        if idx in self.idx_to_brand:
            return self.idx_to_brand[idx]
        elif str(idx) in self.idx_to_brand:
            return self.idx_to_brand[str(idx)]
        else:
            return f"Unknown_{idx}"
    
    def visualize_results(self, image_rgb, results, save_path=None, show=True):
        """
        Visualize detection and classification results.
        
        Args:
            image_rgb: Original image in RGB format
            results: Detection results from detect_and_recognize()
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(image_rgb)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
        
        for result, color in zip(results, colors):
            x1, y1, x2, y2 = result['bbox']
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Create label
            top_pred = result['predictions'][0]
            label = f"{top_pred['brand']}\n{top_pred['confidence']:.2%}"
            
            # Add text background
            ax.text(
                x1, y1 - 10,
                label,
                color='white',
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
            )
            
            # Print details
            print(f"\n🏷️  Detection #{result['detection_id']}")
            print(f"   BBox: ({x1}, {y1}, {x2}, {y2})")
            print(f"   Detection Confidence: {result['detection_conf']:.2%}")
            print(f"   Top-3 Predictions:")
            for j, pred in enumerate(result['predictions'][:3], 1):
                print(f"      {j}. {pred['brand']}: {pred['confidence']:.2%}")
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n💾 Saved visualization to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def process_folder(self, input_folder, output_folder='outputs/pipeline_results', conf_threshold=0.25):
        """
        Process all images in a folder.
        
        Args:
            input_folder: Path to folder containing images
            output_folder: Path to save results
            conf_threshold: YOLO confidence threshold
            
        Returns:
            all_results: List of results for all images
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        
        print(f"\n📁 Processing {len(image_files)} images from: {input_folder}")
        
        all_results = []
        
        for img_file in image_files:
            print(f"\n{'='*70}")
            print(f"Processing: {img_file.name}")
            print(f"{'='*70}")
            
            results, image_rgb = self.detect_and_recognize(str(img_file), conf_threshold)
            
            # Save visualization
            save_path = output_path / f"result_{img_file.stem}.jpg"
            self.visualize_results(image_rgb, results, save_path=str(save_path), show=False)
            
            all_results.append({
                'image': img_file.name,
                'detections': results
            })
        
        # Save JSON summary
        json_path = output_path / 'results_summary.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Processed {len(image_files)} images")
        print(f"📁 Results saved to: {output_folder}")
        
        return all_results


# Demo usage
if __name__ == "__main__":
    # Initialize pipeline (will fail until weights are provided)
    try:
        pipeline = LogoDetectionRecognitionPipeline(
            yolo_model_path='detection/runs/detect/logodet3k_yolov8s_baseline50/weights/best.pt',
            resnet_model_path='recognition/weights/model_for_inference.pth',
            device='cuda'
        )
        
        print("\n✅ Pipeline initialized successfully!")
        print("\nExample usage:")
        print("  results, image = pipeline.detect_and_recognize('test.jpg')")
        print("  pipeline.visualize_results(image, results)")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Model weights not found: {e}")
        print("Please provide trained model weights before running the pipeline.")
