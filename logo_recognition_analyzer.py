"""
Logo Recognition Performance Analyzer
Analyze results from run_pipeline.py
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    top_k_accuracy_score, 
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

class LogoRecognitionAnalyzer:
    """Phân tích comprehensive performance của recognition model"""
    
    def __init__(self, annotation_file='LogoDet-3K/annotations.json'):
        """
        Load ground truth annotations
        
        Args:
            annotation_file: Đường dẫn đến file annotations.json (COCO format)
        """
        print("="*70)
        print("📂 Loading ground truth annotations...")
        print("="*70)
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Build mappings
        self.images = {img['id']: img for img in data['images']}
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}
        self.num_classes = len(self.categories)
        
        # Build ground truth for test set
        self.ground_truth = {}
        for ann in data['annotations']:
            img_info = self.images[ann['image_id']]
            if img_info.get('split') == 'test':
                img_name = img_info['file_name']
                if img_name not in self.ground_truth:
                    self.ground_truth[img_name] = []
                self.ground_truth[img_name].append({
                    'brand_id': ann['category_id'],
                    'brand_name': self.categories[ann['category_id']],
                    'bbox': ann['bbox'],
                    'ann_id': ann['id']
                })
        
        print(f"✅ Loaded ground truth for {len(self.ground_truth)} test images")
        print(f"   Total brands: {self.num_classes}")
        
        self.predictions = None
        self.all_labels = None
        self.all_preds = None
        self.all_probs = None
    
    def load_predictions(self, predictions_file):
        """
        Load predictions từ pipeline output (results_summary.json)
        
        Args:
            predictions_file: Đường dẫn đến results_summary.json
        """
        print(f"\n📥 Loading predictions from: {predictions_file}")
        
        with open(predictions_file, 'r') as f:
            self.predictions = json.load(f)
        
        print(f"✅ Loaded predictions for {len(self.predictions)} images")
    
    def prepare_evaluation_data(self):
        """
        Chuẩn bị data cho evaluation
        """
        print("\n🔄 Preparing evaluation data...")
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        matched_count = 0
        unmatched_count = 0
        
        for pred_item in tqdm(self.predictions, desc="Processing predictions"):
            image_name = pred_item['image']
            
            # Get ground truth
            gt_labels = self.ground_truth.get(image_name, [])
            if not gt_labels:
                unmatched_count += 1
                continue
            
            # Get predictions
            detections = pred_item.get('detections', [])
            
            # Match each detection with ground truth
            for i, detection in enumerate(detections):
                if i >= len(gt_labels):
                    break
                
                gt = gt_labels[i]
                gt_brand_id = gt['brand_id']
                
                # Get prediction
                if detection['predictions']:
                    pred_brand_name = detection['predictions'][0]['brand']
                    pred_brand_id = next((k for k, v in self.categories.items() 
                                         if v == pred_brand_name), -1)
                    
                    # Get probability distribution
                    probs = np.zeros(self.num_classes)
                    for pred in detection['predictions']:
                        brand_id = next((k for k, v in self.categories.items() 
                                       if v == pred['brand']), -1)
                        if brand_id != -1 and brand_id < self.num_classes:
                            probs[brand_id] = pred['confidence']
                    
                    all_labels.append(gt_brand_id)
                    all_preds.append(pred_brand_id)
                    all_probs.append(probs)
                    matched_count += 1
        
        self.all_labels = np.array(all_labels)
        self.all_preds = np.array(all_preds)
        self.all_probs = np.array(all_probs)
        
        print(f"\n✅ Prepared {len(self.all_labels)} matched predictions")
        print(f"   Matched: {matched_count}, Unmatched: {unmatched_count}")
        
        return self.all_labels, self.all_preds, self.all_probs
    
    def evaluate_comprehensive(self):
        """
        Comprehensive evaluation (giống style code của bạn)
        """
        if self.all_labels is None:
            self.prepare_evaluation_data()
        
        print("\n" + "="*70)
        print("🔍 COMPREHENSIVE EVALUATION")
        print("="*70)
        
        # Define all classes
        all_classes = np.arange(self.num_classes)
        
        # Calculate metrics
        top1_acc = accuracy_score(self.all_labels, self.all_preds) * 100
        
        try:
            top5_acc = top_k_accuracy_score(
                self.all_labels, self.all_probs, k=5, labels=all_classes
            ) * 100
        except:
            top5_acc = 0.0
        
        try:
            top10_acc = top_k_accuracy_score(
                self.all_labels, self.all_probs, k=10, labels=all_classes
            ) * 100
        except:
            top10_acc = 0.0
        
        f1_macro = f1_score(self.all_labels, self.all_preds, 
                           average='macro', zero_division=0)
        f1_weighted = f1_score(self.all_labels, self.all_preds,
                              average='weighted', zero_division=0)
        precision_macro = precision_score(self.all_labels, self.all_preds,
                                         average='macro', zero_division=0)
        recall_macro = recall_score(self.all_labels, self.all_preds,
                                    average='macro', zero_division=0)
        
        # Print results (giống format của bạn)
        print("\n📊 Accuracy Metrics:")
        print(f"  Top-1 Accuracy:  {top1_acc:.2f}%")
        print(f"  Top-5 Accuracy:  {top5_acc:.2f}%")
        print(f"  Top-10 Accuracy: {top10_acc:.2f}%")
        
        print(f"\n📊 F1 Scores:")
        print(f"  F1 (Macro):    {f1_macro:.4f}")
        print(f"  F1 (Weighted): {f1_weighted:.4f}")
        
        print(f"\n📊 Precision & Recall:")
        print(f"  Precision (Macro): {precision_macro:.4f}")
        print(f"  Recall (Macro):    {recall_macro:.4f}")
        
        print("\n" + "="*70)
        
        self.results = {
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'top10_acc': top10_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'predictions': self.all_preds,
            'labels': self.all_labels,
            'probabilities': self.all_probs
        }
        
        return self.results
    
    def plot_overall_metrics(self, figsize=(14, 6)):
        """Vẽ tổng hợp metrics"""
        if not hasattr(self, 'results'):
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Top-K Accuracy
        accuracies = [self.results['top1_acc'], 
                     self.results['top5_acc'], 
                     self.results['top10_acc']]
        k_values = ['Top-1', 'Top-5', 'Top-10']
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        
        bars = ax1.bar(k_values, accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Top-K Accuracy', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.2f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
        
        # F1, Precision, Recall
        metrics = [self.results['f1_macro'],
                  self.results['precision_macro'],
                  self.results['recall_macro']]
        metric_names = ['F1', 'Precision', 'Recall']
        colors2 = ['#3498db', '#9b59b6', '#1abc9c']
        
        bars2 = ax2.bar(metric_names, metrics, color=colors2, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Macro Average Metrics', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, score in zip(bars2, metrics):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.4f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        
        plt.suptitle(f'Overall Performance (N={len(self.all_labels):,})',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_accuracy_pie_chart(self, figsize=(10, 6)):
        """Pie chart accuracy"""
        if not hasattr(self, 'results'):
            return
        
        correct = (self.all_labels == self.all_preds).sum()
        incorrect = len(self.all_labels) - correct
        top1_acc = self.results['top1_acc']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sizes = [correct, incorrect]
        labels = [f'Correct\n{top1_acc:.2f}%', f'Incorrect\n{100-top1_acc:.2f}%']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.1, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                          colors=colors, autopct='%d',
                                          startangle=90,
                                          textprops={'fontsize': 13, 'fontweight': 'bold'})
        
        for i, autotext in enumerate(autotexts):
            autotext.set_text(f'{sizes[i]:,}')
            autotext.set_color('white')
            autotext.set_fontsize(14)
        
        ax.set_title(f'Top-1 Recognition Accuracy\nTotal: {len(self.all_labels):,}',
                    fontsize=15, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    
    def generate_full_report(self):
        """Generate full report"""
        print("\n" + "="*70)
        print("🚀 GENERATING COMPREHENSIVE REPORT")
        print("="*70)
        
        self.prepare_evaluation_data()
        results = self.evaluate_comprehensive()
        
        print("\n📊 Generating charts...")
        self.plot_overall_metrics()
        self.plot_accuracy_pie_chart()
        
        print("\n✅ Report completed!")
        return results