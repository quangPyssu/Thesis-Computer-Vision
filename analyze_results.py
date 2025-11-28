"""
Analyze Existing Pipeline Results

Chỉ chạy analysis từ results_summary.json có sẵn

Usage:
    python analyze_results.py \
        --predictions outputs/pipeline_results/results_summary.json \
        --annotation LogoDet-3K/annotations.json \
        --output outputs/analysis_only
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def save_results(results, analyzer, output_folder):
    """
    Lưu tất cả kết quả ra file
    """
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
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
            'confidence': float(results['probabilities'][i][results['predictions'][i]]) if results['predictions'][i] < len(results['probabilities'][i]) else 0.0
        })
    
    detailed_file = output_path / f'detailed_results_{timestamp}.json'
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"✅ Saved detailed results to: {detailed_file}")
    
    # 3. Save summary report as text
    summary_file = output_path / f'summary_report_{timestamp}.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
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
        
        brand_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for label, pred in zip(results['labels'], results['predictions']):
            brand_name = analyzer.categories.get(label, f'Unknown_{label}')
            brand_stats[brand_name]['total'] += 1
            if label == pred:
                brand_stats[brand_name]['correct'] += 1
        
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
        'metrics_file': str(metrics_file),
        'detailed_file': str(detailed_file),
        'summary_file': str(summary_file),
        'csv_file': str(csv_file)
    }

def main():
    parser = argparse.ArgumentParser(
        description='Analyze existing pipeline results'
    )
    
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to results_summary.json from pipeline')
    parser.add_argument('--annotation', type=str, required=True,
                        help='Path to annotations.json')
    parser.add_argument('--output', type=str, default='outputs/analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots (faster)')
    
    args = parser.parse_args()
    
    # Check files exist
    predictions_path = Path(args.predictions)
    annotation_path = Path(args.annotation)
    
    if not predictions_path.exists():
        print(f"❌ Predictions file not found: {predictions_path}")
        sys.exit(1)
    
    if not annotation_path.exists():
        print(f"❌ Annotation file not found: {annotation_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("📊 ANALYZING PIPELINE RESULTS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Predictions:  {args.predictions}")
    print(f"  Annotations:  {args.annotation}")
    print(f"  Output:       {args.output}")
    
    try:
        # Import analyzer
        from logo_recognition_analyzer import LogoRecognitionAnalyzer
        
        # Initialize analyzer
        print("\n" + "="*70)
        print("🔧 INITIALIZING ANALYZER")
        print("="*70)
        
        analyzer = LogoRecognitionAnalyzer(str(annotation_path))
        
        # Load predictions
        print("\n" + "="*70)
        print("📥 LOADING PREDICTIONS")
        print("="*70)
        
        analyzer.load_predictions(str(predictions_path))
        
        # Prepare data
        print("\n" + "="*70)
        print("🔄 PREPARING EVALUATION DATA")
        print("="*70)
        
        analyzer.prepare_evaluation_data()
        
        # Evaluate
        print("\n" + "="*70)
        print("📊 EVALUATING PERFORMANCE")
        print("="*70)
        
        results = analyzer.evaluate_comprehensive()
        
        # Generate plots (if not disabled)
        if not args.no_plots:
            print("\n" + "="*70)
            print("📈 GENERATING PLOTS")
            print("="*70)
            
            try:
                analyzer.plot_overall_metrics()
                analyzer.plot_accuracy_pie_chart()
                print("✅ Plots generated successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not generate plots: {e}")
        
        # Save results
        saved_files = save_results(results, analyzer, args.output)
        
        # Print summary
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print(f"\n📊 Key Results:")
        print(f"  Total Predictions: {len(results['labels']):,}")
        print(f"  Top-1 Accuracy:    {results['top1_acc']:.2f}%")
        print(f"  Top-5 Accuracy:    {results['top5_acc']:.2f}%")
        print(f"  Top-10 Accuracy:   {results['top10_acc']:.2f}%")
        print(f"  F1 Score (Macro):  {results['f1_macro']:.4f}")
        print(f"  Precision (Macro): {results['precision_macro']:.4f}")
        print(f"  Recall (Macro):    {results['recall_macro']:.4f}")
        
        print(f"\n📁 Saved Files:")
        for file_type, file_path in saved_files.items():
            print(f"  • {file_path}")
        
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