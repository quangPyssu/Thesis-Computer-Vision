# Detection Results

This folder contains the training results and visualizations for the YOLOv8n logo detection model trained on the LogoDet-3K dataset.

## Files

### Configuration & Results
- **results_summary.json** - Comprehensive training metrics, configuration, and performance summary

### Training Visualizations
- **training_curves.png** - Training and validation metrics over 40 epochs including:
  - Box, classification, and DFL losses
  - Precision and recall
  - mAP@0.5 and mAP@0.5:0.95

### Performance Analysis
- **confusion_matrix.png** - Raw confusion matrix showing detection performance across classes
- **confusion_matrix_normalized.png** - Normalized confusion matrix for easier comparison
- **pr_curve.png** - Precision-Recall curve showing model performance trade-offs
- **f1_curve.png** - F1 score across different confidence thresholds

### Dataset Information
- **labels_distribution.jpg** - Distribution of bounding box labels in the training dataset

## Key Metrics (Final Epoch 40)

- **mAP@0.5**: 81.34%
- **mAP@0.5:0.95**: 60.85%
- **Precision**: 82.99%
- **Recall**: 73.04%
- **Training Time**: 9.56 hours

## Model Details

- **Architecture**: YOLOv8n (nano)
- **Dataset**: LogoDet-3K (3000+ logo classes)
- **Input Size**: 640×640
- **Batch Size**: 32
- **Total Epochs**: 40

## Source

All charts and metrics are generated from the training run located at:
`detection/runs/detect/logodet3k_yolov8s_baseline50/`
