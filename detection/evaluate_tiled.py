"""
Evaluate Tiled Inference vs Single-Pass Inference
==================================================
Compares detection performance with and without tiling on test set.
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from inference_tiled import TiledInference


def evaluate_single_pass(
    model_path: str,
    data_yaml: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict:
    """
    Run standard single-pass inference evaluation.
    
    Returns metrics dict with mAP@0.5, mAP@0.5:0.95, inference time, etc.
    """
    print("\n" + "="*70)
    print("SINGLE-PASS INFERENCE (Baseline)")
    print("="*70)
    
    model = YOLO(model_path)
    
    # Run validation (uses full image inference)
    start_time = time.time()
    results = model.val(
        data=data_yaml,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=True
    )
    eval_time = time.time() - start_time
    
    metrics = {
        'method': 'single_pass',
        'map50': float(results.box.map50),
        'map50_95': float(results.box.map),
        'precision': float(results.box.p[0]) if hasattr(results.box, 'p') else 0.0,
        'recall': float(results.box.r[0]) if hasattr(results.box, 'r') else 0.0,
        'inference_time': eval_time,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold
    }
    
    print(f"\nResults:")
    print(f"  mAP@0.5: {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Total time: {eval_time:.2f}s")
    
    return metrics


def evaluate_tiled_inference(
    model_path: str,
    data_yaml: str,
    tile_size: int = 1024,
    overlap: float = 0.2,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict:
    """
    Run tiled inference evaluation.
    
    Note: This provides inference results but mAP calculation requires
    matching with ground truth annotations. For now, we report detection
    statistics and can manually compare with validation results.
    """
    print("\n" + "="*70)
    print(f"TILED INFERENCE (tile_size={tile_size}, overlap={overlap})")
    print("="*70)
    
    # Initialize tiled inference
    tiler = TiledInference(
        model_path=model_path,
        tile_size=tile_size,
        overlap=overlap,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    # Load test images from data.yaml
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    test_path = Path(data_config['path']) / data_config['test']
    test_images = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
    
    print(f"Found {len(test_images)} test images")
    
    # Run tiled inference on all test images
    all_results = []
    total_tiles = 0
    total_detections_before_nms = 0
    total_detections_after_nms = 0
    
    start_time = time.time()
    
    for img_path in tqdm(test_images, desc="Processing images"):
        result = tiler.predict_tiled(str(img_path), verbose=False)
        
        all_results.append({
            'image': img_path.name,
            'n_tiles': result['n_tiles'],
            'n_detections_before_nms': result['n_detections_before_nms'],
            'n_detections_after_nms': result['n_detections_after_nms'],
            'detections': result['detections'].tolist()
        })
        
        total_tiles += result['n_tiles']
        total_detections_before_nms += result['n_detections_before_nms']
        total_detections_after_nms += result['n_detections_after_nms']
    
    eval_time = time.time() - start_time
    
    avg_tiles = total_tiles / len(test_images)
    avg_detections = total_detections_after_nms / len(test_images)
    
    metrics = {
        'method': 'tiled_inference',
        'tile_size': tile_size,
        'overlap': overlap,
        'n_images': len(test_images),
        'total_tiles': total_tiles,
        'avg_tiles_per_image': avg_tiles,
        'total_detections_before_nms': total_detections_before_nms,
        'total_detections_after_nms': total_detections_after_nms,
        'avg_detections_per_image': avg_detections,
        'inference_time': eval_time,
        'avg_time_per_image': eval_time / len(test_images),
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold
    }
    
    print(f"\nResults:")
    print(f"  Total tiles created: {total_tiles}")
    print(f"  Avg tiles per image: {avg_tiles:.2f}")
    print(f"  Total detections (after NMS): {total_detections_after_nms}")
    print(f"  Avg detections per image: {avg_detections:.2f}")
    print(f"  Total time: {eval_time:.2f}s")
    print(f"  Avg time per image: {eval_time/len(test_images):.3f}s")
    
    # Note about mAP calculation
    print(f"\n⚠ Note: To calculate mAP for tiled inference, you need to:")
    print(f"  1. Save detections in YOLO format")
    print(f"  2. Run YOLO val() with these predictions")
    print(f"  OR use pycocotools to compare with ground truth")
    
    return metrics, all_results


def compare_methods(
    model_path: str,
    data_yaml: str,
    tile_size: int = 1024,
    overlap: float = 0.2,
    output_file: str = None
):
    """
    Compare single-pass and tiled inference.
    """
    print("\n" + "="*70)
    print("COMPARING SINGLE-PASS vs TILED INFERENCE")
    print("="*70)
    
    # Evaluate single-pass
    single_metrics = evaluate_single_pass(model_path, data_yaml)
    
    # Evaluate tiled
    tiled_metrics, tiled_results = evaluate_tiled_inference(
        model_path, data_yaml, tile_size, overlap
    )
    
    # Comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\nSingle-Pass:")
    print(f"  mAP@0.5: {single_metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95: {single_metrics['map50_95']:.4f}")
    print(f"  Time: {single_metrics['inference_time']:.2f}s")
    
    print(f"\nTiled Inference:")
    print(f"  Avg tiles per image: {tiled_metrics['avg_tiles_per_image']:.2f}")
    print(f"  Time: {tiled_metrics['inference_time']:.2f}s")
    print(f"  Slowdown: {tiled_metrics['inference_time']/single_metrics['inference_time']:.2f}x")
    
    print(f"\n💡 For thesis comparison:")
    print(f"  Run YOLO validation on tiled predictions to get mAP metrics")
    print(f"  Expected: tiled inference should improve recall on small logos")
    
    # Save results
    if output_file:
        results = {
            'single_pass': single_metrics,
            'tiled_inference': tiled_metrics,
            'tiled_detections': tiled_results
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare single-pass and tiled inference'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model weights (e.g., runs/detect/best.pt)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='../data/logodet3k_yolo/data.yaml',
        help='Path to data.yaml'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=1024,
        help='Tile size for tiled inference'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.2,
        help='Overlap ratio (0-1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='detection/runs/tiled_comparison.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--single-only',
        action='store_true',
        help='Run only single-pass evaluation'
    )
    parser.add_argument(
        '--tiled-only',
        action='store_true',
        help='Run only tiled evaluation'
    )
    
    args = parser.parse_args()
    
    if args.single_only:
        evaluate_single_pass(args.model, args.data)
    elif args.tiled_only:
        evaluate_tiled_inference(
            args.model, args.data, args.tile_size, args.overlap
        )
    else:
        compare_methods(
            args.model, args.data, args.tile_size, args.overlap, args.output
        )


if __name__ == '__main__':
    main()
