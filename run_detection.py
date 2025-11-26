"""
Run YOLOv8 Logo Detection Training

Train or resume YOLOv8 model for logo detection.
"""

from ultralytics import YOLO
import yaml
from pathlib import Path


def train_yolo(config_path='configs/detection_config.yaml', resume=False, resume_path=None):
    """
    Train YOLOv8 detection model
    
    Args:
        config_path: Path to detection configuration file
        resume: Whether to resume from checkpoint
        resume_path: Path to checkpoint to resume from
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    train_cfg = config['training']
    
    print("="*70)
    print("🚀 TRAINING YOLOV8 LOGO DETECTION MODEL")
    print("="*70)
    
    if resume and resume_path:
        # Resume from checkpoint
        print(f"\n📦 Resuming from checkpoint: {resume_path}")
        model = YOLO(resume_path)
        
        results = model.train(
            resume=True,
            epochs=train_cfg['epochs']
        )
    else:
        # Train from pretrained weights
        print(f"\n📦 Loading pretrained model: {model_cfg['pretrained_weights']}")
        model = YOLO(model_cfg['pretrained_weights'])
        
        results = model.train(
            data=train_cfg['data_yaml'],
            epochs=train_cfg['epochs'],
            imgsz=train_cfg['image_size'],
            batch=train_cfg['batch_size'],
            device=train_cfg['device'],
            workers=train_cfg['workers'],
            patience=train_cfg['patience'],
            save=True,
            name='logodet3k_yolo_training'
        )
    
    print("\n✅ Training completed!")
    print(f"📁 Results saved to: {results.save_dir}")
    print(f"🏆 Best model: {results.save_dir}/weights/best.pt")
    
    return results


def detect_images(model_path, image_source, conf_threshold=0.25, save=True):
    """
    Run detection on images
    
    Args:
        model_path: Path to trained YOLO model
        image_source: Path to image or folder
        conf_threshold: Confidence threshold
        save: Whether to save results
    """
    print("="*70)
    print("🔍 RUNNING YOLO DETECTION")
    print("="*70)
    
    model = YOLO(model_path)
    
    results = model(
        image_source,
        conf=conf_threshold,
        save=save,
        project='outputs/detection_results'
    )
    
    print(f"\n✅ Detection completed!")
    print(f"📁 Processed {len(results)} images")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Logo Detection')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'detect', 'resume'],
                        help='Operation mode')
    parser.add_argument('--config', type=str, default='configs/detection_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='detection/runs/detect/*/weights/best.pt',
                        help='Path to model weights (for detection/resume)')
    parser.add_argument('--source', type=str, default='data/logodet3k_yolo/images/test',
                        help='Image source for detection')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detection')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_yolo(config_path=args.config)
    elif args.mode == 'resume':
        train_yolo(config_path=args.config, resume=True, resume_path=args.model)
    elif args.mode == 'detect':
        detect_images(args.model, args.source, args.conf)
