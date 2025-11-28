"""
Complete Automated Evaluation Workflow

Tự động:
1. Đọc annotations.json để lấy danh sách test images
2. Copy test images vào folder tạm
3. Chạy detection + recognition pipeline
4. Phân tích comprehensive performance
5. Lưu tất cả kết quả

Usage:
    python run_full_evaluation.py --data-root LogoDet-3K --annotation LogoDet-3K/annotations.json
"""

import argparse
import subprocess
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def create_test_folder_from_annotations(annotation_file, data_root, output_folder):
    """
    Bước 1: Tạo folder chứa test images từ annotations
    
    Args:
        annotation_file: Path to annotations.json
        data_root: Root folder chứa images (train/test/val folders)
        output_folder: Folder để copy test images vào
    
    Returns:
        test_folder_path, num_test_images
    """
    print("\n" + "="*70)
    print("📂 STEP 1: CREATING TEST FOLDER FROM ANNOTATIONS")
    print("="*70)
    
    # Load annotations
    print(f"\n📥 Loading annotations from: {annotation_file}")
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Get test images
    test_images = [img for img in data['images'] if img.get('split') == 'test']
    
    print(f"✅ Found {len(test_images)} test images")
    
    # Create output folder
    test_folder = Path(output_folder) / 'test_images'
    test_folder.mkdir(parents=True, exist_ok=True)
    
    # Copy test images
    data_root_path = Path(data_root)
    copied_count = 0
    not_found = []
    
    print(f"\n📋 Copying test images to: {test_folder}")
    
    for img in tqdm(test_images, desc="Copying images"):
        img_filename = img['file_name']
        
        # Try to find image in different locations
        possible_paths = [
            data_root_path / 'test' / img_filename,
            data_root_path / img_filename,
            data_root_path / 'images' / img_filename,
        ]
        
        source_path = None
        for path in possible_paths:
            if path.exists():
                source_path = path
                break
        
        if source_path:
            dest_path = test_folder / img_filename
            shutil.copy2(source_path, dest_path)
            copied_count += 1
        else:
            not_found.append(img_filename)
    
    print(f"\n✅ Copied {copied_count}/{len(test_images)} images")
    
    if not_found:
        print(f"⚠️  Warning: {len(not_found)} images not found")
        if len(not_found) <= 10:
            print(f"   Missing: {', '.join(not_found[:10])}")
        else:
            print(f"   Missing: {', '.join(not_found[:10])}... and {len(not_found)-10} more")
    
    return test_folder, copied_count

def run_pipeline(test_folder, args):
    """
    Bước 2: Chạy detection + recognition pipeline
    """
    print("\n" + "="*70)
    print("🚀 STEP 2: RUNNING DETECTION + RECOGNITION PIPELINE")
    print("="*70)
    
    pipeline_output = Path(args.output) / 'pipeline_results'
    
    pipeline_cmd = [
        'python', 'run_pipeline.py',
        '--source', str(test_folder),
        '--yolo-model', args.yolo_model,
        '--resnet-model', args.resnet_model,
        '--conf', str(args.conf),
        '--output', str(pipeline_output),
        '--device', args.device,
        '--batch',
        '--no-show'
    ]
    
    print(f"\n🔧 Running command:")
    print(' '.join(pipeline_cmd))
    
    try:
        subprocess.run(pipeline_cmd, check=True)
        print("\n✅ Pipeline completed successfully!")
        return pipeline_output
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        return None

def run_analysis(pipeline_output, annotation_file, output_folder):
    """
    Bước 3: Phân tích comprehensive performance
    """
    print("\n" + "="*70)
    print("📊 STEP 3: COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Import analyzer
    from logo_recognition_analyzer import LogoRecognitionAnalyzer
    
    # Initialize analyzer
    analyzer = LogoRecognitionAnalyzer(annotation_file)
    
    # Load predictions
    predictions_file = pipeline_output / 'results_summary.json'
    
    if not predictions_file.exists():
        print(f"❌ Predictions file not found: {predictions_file}")
        return None
    
    analyzer.load_predictions(str(predictions_file))
    
    # Run evaluation
    results = analyzer.generate_full_report()
    
    return results, analyzer

def save_results(results, analyzer, output_folder):
    """
    Bước 4: Lưu tất cả kết quả ra file
    """
    print("\n" + "="*70)
    print("💾 STEP 4: SAVING RESULTS")
    print("="*70)
    
    output_path = Path(output_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save performance metrics JSON
    metrics_output = {
        'timestamp': timestamp,
        'evaluation_metrics': {
            'top1_accuracy': float(results['top1_acc']),
            'top5_accuracy': float(results['top5_acc']),
            'top10_accuracy': float(results['top10_acc']),
            'f1_macro': float(results['f1_macro']),
            'f1_weighted': float(results['f1_weighted']),
            'precision_macro': float(results['precision_macro']),
            'recall_macro': float(results['recall_macro'])
        },
        'dataset_info': {
            'total_predictions': int(len(results['labels'])),
            'num_classes': int(analyzer.num_classes),
            'num_test_images': int(len(analyzer.ground_truth))
        }
    }
    
    metrics_file = output_path / f'performance_metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"✅ Saved metrics to: {metrics_file}")
    
    # 2. Save detailed predictions vs ground truth
    detailed_results = []
    for i in range(len(results['labels'])):
        detailed_results.append({
            'sample_id': int(i),
            'ground_truth_id': int(results['labels'][i]),
            'ground_truth_name': analyzer.categories.get(results['labels'][i], 'Unknown'),
            'predicted_id': int(results['predictions'][i]),
            'predicted_name': analyzer.categories.get(results['predictions'][i], 'Unknown'),
            'correct': bool(results['labels'][i] == results['predictions'][i]),
            'confidence': float(results['probabilities'][i][results['predictions'][i]])
        })
    
    detailed_file = output_path / f'detailed_results_{timestamp}.json'
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"✅ Saved detailed results to: {detailed_file}")
    
    # 3. Save summary report as text
    summary_file = output_path / f'summary_report_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LOGO RECOGNITION EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("="*70 + "\n")
        f.write("DATASET INFORMATION\n")
        f.write("="*70 + "\n")
        f.write(f"Total Test Images:    {len(analyzer.ground_truth):,}\n")
        f.write(f"Total Predictions:    {len(results['labels']):,}\n")
        f.write(f"Number of Classes:    {analyzer.num_classes:,}\n\n")
        
        f.write("="*70 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("="*70 + "\n\n")
        
        f.write("📊 Accuracy Metrics:\n")
        f.write(f"  Top-1 Accuracy:  {results['top1_acc']:.2f}%\n")
        f.write(f"  Top-5 Accuracy:  {results['top5_acc']:.2f}%\n")
        f.write(f"  Top-10 Accuracy: {results['top10_acc']:.2f}%\n\n")
        
        f.write("📊 F1 Scores:\n")
        f.write(f"  F1 (Macro):    {results['f1_macro']:.4f}\n")
        f.write(f"  F1 (Weighted): {results['f1_weighted']:.4f}\n\n")
        
        f.write("📊 Precision & Recall:\n")
        f.write(f"  Precision (Macro): {results['precision_macro']:.4f}\n")
        f.write(f"  Recall (Macro):    {results['recall_macro']:.4f}\n\n")
        
        # Per-brand statistics
        f.write("="*70 + "\n")
        f.write("PER-BRAND STATISTICS (Top 20 by sample count)\n")
        f.write("="*70 + "\n\n")
        
        from collections import defaultdict
        brand_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for label, pred in zip(results['labels'], results['predictions']):
            brand_name = analyzer.categories.get(label, f'Unknown_{label}')
            brand_stats[brand_name]['total'] += 1
            if label == pred:
                brand_stats[brand_name]['correct'] += 1
        
        # Sort by total count
        sorted_brands = sorted(brand_stats.items(), 
                             key=lambda x: x[1]['total'], 
                             reverse=True)[:20]
        
        for i, (brand, stats) in enumerate(sorted_brands, 1):
            acc = (stats['correct'] / stats['total']) * 100
            f.write(f"{i:2d}. {brand:40s} {acc:6.2f}% ({stats['correct']:4d}/{stats['total']:4d})\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✅ Saved summary report to: {summary_file}")
    
    # 4. Save per-brand accuracy CSV
    brand_stats_list = []
    for brand, stats in brand_stats.items():
        brand_stats_list.append({
            'brand': brand,
            'correct': stats['correct'],
            'total': stats['total'],
            'accuracy': (stats['correct'] / stats['total']) * 100
        })
    
    import pandas as pd
    df_brands = pd.DataFrame(brand_stats_list)
    df_brands = df_brands.sort_values('total', ascending=False)
    
    csv_file = output_path / f'per_brand_accuracy_{timestamp}.csv'
    df_brands.to_csv(csv_file, index=False)
    print(f"✅ Saved per-brand accuracy to: {csv_file}")
    
    return {
        'metrics_file': metrics_file,
        'detailed_file': detailed_file,
        'summary_file': summary_file,
        'csv_file': csv_file
    }

def main():
    parser = argparse.ArgumentParser(
        description='Automated Logo Recognition Evaluation Pipeline'
    )
    
    # Required arguments
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root folder containing train/test/val subfolders')
    parser.add_argument('--annotation', type=str, required=True,
                        help='Path to annotations.json')
    
    # Model arguments
    parser.add_argument('--yolo-model', type=str,
                        default='detection/runs/detect/logodet3k_yolov8s_baseline50/weights/best.pt',
                        help='Path to YOLOv8 weights')
    parser.add_argument('--resnet-model', type=str,
                        default='recognition/weights/model_for_inference.pth',
                        help='Path to ResNet50 checkpoint')
    
    # Pipeline arguments
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    
    # Output arguments
    parser.add_argument('--output', type=str, 
                        default='outputs/evaluation',
                        help='Output directory for all results')
    
    # Optional flags
    parser.add_argument('--skip-copy', action='store_true',
                        help='Skip copying images (use existing test_images folder)')
    parser.add_argument('--skip-pipeline', action='store_true',
                        help='Skip pipeline (use existing predictions)')
    parser.add_argument('--keep-test-folder', action='store_true',
                        help='Keep test images folder after evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🎯 AUTOMATED LOGO RECOGNITION EVALUATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data root:        {args.data_root}")
    print(f"  Annotation file:  {args.annotation}")
    print(f"  Output folder:    {args.output}")
    print(f"  Device:           {args.device}")
    print(f"  Detection conf:   {args.conf}")
    
    try:
        # Step 1: Create test folder
        if not args.skip_copy:
            test_folder, num_images = create_test_folder_from_annotations(
                args.annotation,
                args.data_root,
                args.output
            )
            
            if num_images == 0:
                print("\n❌ No test images found! Please check your paths.")
                sys.exit(1)
        else:
            test_folder = Path(args.output) / 'test_images'
            print(f"\n⏭️  Skipping copy, using existing folder: {test_folder}")
        
        # Step 2: Run pipeline
        if not args.skip_pipeline:
            pipeline_output = run_pipeline(test_folder, args)
            if pipeline_output is None:
                print("\n❌ Pipeline failed!")
                sys.exit(1)
        else:
            pipeline_output = Path(args.output) / 'pipeline_results'
            print(f"\n⏭️  Skipping pipeline, using existing results: {pipeline_output}")
        
        # Step 3: Run analysis
        result = run_analysis(pipeline_output, args.annotation, args.output)
        if result is None:
            print("\n❌ Analysis failed!")
            sys.exit(1)
        
        results, analyzer = result
        
        # Step 4: Save all results
        saved_files = save_results(results, analyzer, args.output)
        
        # Cleanup test folder if not keeping
        if not args.keep_test_folder and not args.skip_copy:
            print(f"\n🗑️  Cleaning up test folder: {test_folder}")
            shutil.rmtree(test_folder)
        
        # Final summary
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📊 Key Results:")
        print(f"  Top-1 Accuracy:  {results['top1_acc']:.2f}%")
        print(f"  Top-5 Accuracy:  {results['top5_acc']:.2f}%")
        print(f"  F1 Score:        {results['f1_macro']:.4f}")
        
        print(f"\n📁 Saved Files:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type:15s}: {file_path}")
        
        print(f"\n📁 All results in: {args.output}")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()