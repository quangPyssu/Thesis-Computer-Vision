# Logo Detection and Recognition System

Complete pipeline combining **YOLOv8** object detection with **ResNet50** logo classification for brand recognition in images.

## 📁 Project Structure

```
Thesis/
├── detection/              # YOLOv8 logo detection
│   ├── train_yolo.py      # Training script
│   ├── convert_logodet3k_to_yolo.py  # Data conversion
│   ├── yolo*.pt           # Pretrained YOLO weights
│   └── runs/              # Training outputs
│       └── detect/
│           └── */weights/
│               ├── best.pt
│               └── last.pt
│
├── recognition/           # ResNet50 logo classification
│   ├── model.py          # Model architecture
│   ├── train_resnet.py   # Training placeholder
│   └── weights/          # Trained model checkpoints
│       └── model_for_inference.pth  (to be provided)
│
├── pipeline/             # Combined detection + recognition
│   ├── logo_pipeline.py  # Main pipeline class
│   └── __init__.py
│
├── data/                 # Datasets
│   └── logodet3k_yolo/  # YOLO format data
│       ├── data.yaml
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│
├── configs/              # Configuration files
│   ├── detection_config.yaml
│   ├── recognition_config.yaml
│   └── pipeline_config.yaml
│
├── outputs/              # Pipeline results
│   └── pipeline_results/
│
├── run_detection.py      # Detection training/inference
├── run_recognition.py    # Recognition training
├── run_pipeline.py       # Complete pipeline
├── activate_yolo.ps1     # Environment activation
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Activate conda environment
.\activate_yolo.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Detection Model (YOLOv8)

```powershell
# Train from scratch
python run_detection.py --mode train

# Resume training
python run_detection.py --mode resume --model detection/runs/detect/*/weights/last.pt

# Run detection only
python run_detection.py --mode detect --model detection/runs/detect/*/weights/best.pt --source test_image.jpg
```

### 3. Train Recognition Model (ResNet50)

Use the Jupyter notebook for training:
```
📓 Open and run: logo_recognition_notebook.ipynb
```

After training, export the model:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {'num_classes': num_classes},
    'idx_to_brand': idx_to_brand,
    'brand_to_idx': brand_to_idx,
    'test_accuracy': test_accuracy
}, 'recognition/weights/model_for_inference.pth')
```

### 4. Run Complete Pipeline

```powershell
# Process single image
python run_pipeline.py --source test_image.jpg

# Process entire folder
python run_pipeline.py --source path/to/images --batch

# Custom configuration
python run_pipeline.py ^
    --source test.jpg ^
    --yolo-model detection/runs/detect/best_model/weights/best.pt ^
    --resnet-model recognition/weights/model_for_inference.pth ^
    --conf 0.3 ^
    --top-k 5 ^
    --output results/
```

## 🎯 Pipeline Overview

### Stage 1: Detection (YOLOv8)
- Detect logo bounding boxes in images
- Fast and accurate object detection
- Returns: bbox coordinates + confidence scores

### Stage 2: Recognition (ResNet50)
- Classify each detected logo
- 3000 brand classes (LogoDet-3K)
- Returns: top-K brand predictions + confidence scores

### Combined Output
Each detection contains:
```python
{
    'detection_id': 1,
    'bbox': [x1, y1, x2, y2],
    'detection_conf': 0.95,
    'predictions': [
        {'brand': 'Nike', 'confidence': 0.98, 'class_id': 42},
        {'brand': 'Adidas', 'confidence': 0.01, 'class_id': 17},
        ...
    ]
}
```

## 📊 Dataset: LogoDet-3K

- **Classes**: 3,000 brand logos
- **Images**: ~150K total
- **Format**: COCO annotations → YOLO format
- **Splits**: Train / Val / Test

## ⚙️ Configuration

Edit YAML files in `configs/`:

- `detection_config.yaml` - YOLOv8 training parameters
- `recognition_config.yaml` - ResNet50 training parameters  
- `pipeline_config.yaml` - Pipeline settings

## 🔧 Key Features

### Detection Module
- Multi-scale detection
- Data augmentation
- Transfer learning from COCO
- GPU optimization

### Recognition Module
- ResNet50 backbone
- ArcFace loss for metric learning
- Mixed precision training
- Class balancing

### Pipeline
- End-to-end inference
- Batch processing
- Visualization tools
- JSON export

## 📈 Training Tips

### Detection
```python
# Increase GPU utilization
batch_size = 32  # or 64, 128
workers = 2      # or 4 (may have issues on Windows)

# Resume training
model = YOLO('detection/runs/detect/*/weights/last.pt')
model.train(resume=True, epochs=50)
```

### Recognition
- Use mixed precision (`amp=True`)
- Balance classes (max 500 samples/class)
- Early stopping with patience
- TensorBoard monitoring

## 🖼️ Visualization

Results include:
- Annotated images with bboxes
- Brand labels + confidence scores
- Color-coded detections
- JSON summary files

## 📝 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
PyYAML>=6.0
albumentations>=1.3.0
```

## 🔄 Workflow

1. **Prepare Data** → Convert LogoDet-3K to YOLO format
2. **Train Detection** → YOLOv8 on logo bounding boxes
3. **Train Recognition** → ResNet50 on cropped logos
4. **Run Pipeline** → Combined detection + classification
5. **Evaluate** → Metrics, visualizations, export

## 🐛 Troubleshooting

### Low GPU Utilization
- Increase `batch_size`
- Set `workers=0` on Windows
- Check CUDA availability

### Training Interrupted
```python
# Resume from checkpoint
model = YOLO('path/to/last.pt')
model.train(resume=True)
```

### Out of Memory
- Reduce `batch_size`
- Use smaller model (`yolov8n` instead of `yolov8s`)
- Enable mixed precision

## 📚 References

- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **LogoDet-3K**: Logo detection benchmark dataset
- **ArcFace**: Angular margin loss for face recognition

## 📧 Contact

For questions or issues, please contact the project maintainer.

---

**Status**: 🟡 In Development
- ✅ Detection module complete
- 🟡 Recognition model training in progress
- ⏳ Pipeline awaiting recognition weights
