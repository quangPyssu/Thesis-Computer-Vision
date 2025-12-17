"""
Contribution #1: Small-Logo-Aware Training
============================================
Enhanced YOLOv8 training with:
- Higher resolution (960x960) for better small logo detection
- Copy-paste augmentation for synthetic small logo instances
- Stronger data augmentation (scale jitter, translation, rotation, color)
- Mosaic augmentation enabled

All changes relative to baseline YOLOv8n (640x640, default augmentation).
"""

from ultralytics import YOLO

if __name__ == '__main__':
    # Load pretrained model - you can choose:
    # Option 1: Start from COCO pretrained (same as baseline)
    # model = YOLO('yolov8n.pt')
    
    # Option 2: Fine-tune from your baseline (uncomment below)
    # model = YOLO('runs/detect/logodet3k_yolov8s_baseline50/weights/best.pt')
    
    # Option 3: RESUME from interrupted training
    model = YOLO('runs/detect/logodet3k_yolov8n_smalllogo_contrib13/weights/last.pt')

    results = model.train(
        data='../data/logodet3k_yolo/data.yaml',
        resume=True,  # Resume training from checkpoint
        
        # Training duration - match baseline
        epochs=40,
        patience=0,
        
        # Resolution improvement for small logos
        imgsz=960,  # Increased from 640 (baseline) - adjust based on GPU memory
        
        # Batch size - reduce if OOM (out of memory)
        batch=16,  # Reduced from 32 due to larger image size
        
        # Hardware
        device=0,
        workers=2,
        
        # Augmentation for small logos
        # Mosaic: mix 4 images to create diverse scenes
        mosaic=1.0,  # Enabled (default)
        
        # Copy-paste: paste logo instances to increase small object count
        copy_paste=0.3,  # 30% probability of copy-paste augmentation
        
        # Scale jitter: helps with varying logo sizes
        scale=0.7,  # Stronger than baseline (0.5), range [1-0.7, 1+0.7]
        
        # Translation: more spatial variation
        translate=0.2,  # Stronger than baseline (0.1)
        
        # Rotation: small angles to preserve logo readability
        degrees=10.0,  # Small rotation (baseline: 0.0)
        
        # Color augmentation: HSV adjustments
        hsv_h=0.02,  # Hue (slightly stronger than baseline 0.015)
        hsv_s=0.7,   # Saturation (same as baseline)
        hsv_v=0.4,   # Value/brightness (same as baseline)
        
        # Perspective and shear: minimal to preserve logo shape
        perspective=0.0,
        shear=0.0,
        
        # Flipping
        flipud=0.0,  # No vertical flip (logos have orientation)
        fliplr=0.5,  # 50% horizontal flip (many logos are symmetric)
        
        # Mixup: disabled (mosaic + copy-paste is sufficient)
        mixup=0.0,
        
        # Save settings
        name='logodet3k_yolov8n_smalllogo_contrib1',
        save=True,
    )

    print("\n" + "="*70)
    print("Contribution #1: Small-Logo-Aware Training COMPLETED!")
    print("="*70)
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print("\nKey changes from baseline:")
    print("  - Resolution: 640 → 960")
    print("  - Copy-paste augmentation: enabled (0.3)")
    print("  - Stronger scale jitter: 0.5 → 0.7")
    print("  - Translation: 0.1 → 0.2")
    print("  - Rotation: 0° → 10°")
    print("="*70)
