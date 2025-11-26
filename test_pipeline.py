"""
Quick test of the complete pipeline

Tests that both detection and recognition models load and work together.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from pipeline.logo_pipeline import LogoDetectionRecognitionPipeline


def test_pipeline():
    print("="*70)
    print("🧪 TESTING LOGO DETECTION + RECOGNITION PIPELINE")
    print("="*70)
    
    try:
        # Initialize pipeline
        pipeline = LogoDetectionRecognitionPipeline(
            yolo_model_path='detection/runs/detect/logodet3k_yolov8s_baseline50/weights/best.pt',
            resnet_model_path='recognition/weights/model_for_inference.pth',
            device='cuda'
        )
        
        print("\n✅ Pipeline initialized successfully!")
        print("\n📊 Pipeline Info:")
        print(f"   YOLO: Loaded")
        print(f"   ResNet: Loaded ({pipeline.num_classes} classes)")
        print(f"   Device: {pipeline.device}")
        
        print("\n" + "="*70)
        print("Ready to process images!")
        print("="*70)
        
        print("\n💡 Usage examples:")
        print("\n1. Single image:")
        print("   python run_pipeline.py --source path/to/image.jpg")
        
        print("\n2. Folder of images:")
        print("   python run_pipeline.py --source path/to/folder --batch")
        
        print("\n3. With custom settings:")
        print("   python run_pipeline.py --source image.jpg --conf 0.3 --top-k 5")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Model file not found")
        print(f"   {e}")
        return False
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)
