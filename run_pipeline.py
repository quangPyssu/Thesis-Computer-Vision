"""
Run Complete Logo Detection + Recognition Pipeline

Process images through YOLOv8 detection and ResNet50 classification.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from pipeline.logo_pipeline import LogoDetectionRecognitionPipeline


def main():
    parser = argparse.ArgumentParser(description='Logo Detection + Recognition Pipeline')
    
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image or folder')
    parser.add_argument('--yolo-model', type=str, 
                        default='detection/runs/detect/logodet3k_yolov8s_baseline50/weights/best.pt',
                        help='Path to YOLOv8 weights')
    parser.add_argument('--resnet-model', type=str,
                        default='recognition/weights/model_for_inference.pth',
                        help='Path to ResNet50 checkpoint')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top predictions to return')
    parser.add_argument('--output', type=str, default='outputs/pipeline_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--batch', action='store_true',
                        help='Process entire folder')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display results')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("="*70)
    print("🚀 LOGO DETECTION + RECOGNITION PIPELINE")
    print("="*70)
    
    try:
        pipeline = LogoDetectionRecognitionPipeline(
            yolo_model_path=args.yolo_model,
            resnet_model_path=args.resnet_model,
            device=args.device
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Make sure you have:")
        print("  1. Trained YOLOv8 detection model")
        print("  2. Trained ResNet50 recognition model")
        print("  3. Correct paths in configs/pipeline_config.yaml")
        sys.exit(1)
    
    # Process images
    source_path = Path(args.source)
    
    if args.batch or source_path.is_dir():
        # Batch processing
        print(f"\n📁 Batch processing folder: {args.source}")
        results = pipeline.process_folder(
            input_folder=args.source,
            output_folder=args.output,
            conf_threshold=args.conf
        )
    else:
        # Single image
        print(f"\n🖼️  Processing single image: {args.source}")
        results, image_rgb = pipeline.detect_and_recognize(
            image_path=args.source,
            conf_threshold=args.conf,
            top_k=args.top_k
        )
        
        # Save and display
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"result_{source_path.stem}.jpg"
        
        pipeline.visualize_results(
            image_rgb,
            results,
            save_path=str(save_path),
            show=not args.no_show
        )
    
    print("\n✅ Pipeline completed successfully!")
    print(f"📁 Results saved to: {args.output}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
