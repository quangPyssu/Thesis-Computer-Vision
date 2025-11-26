# Quick Start Guide

## For Detection Training (YOLOv8)

```powershell
# Navigate to detection folder
cd detection

# Train YOLOv8
python train_yolo.py
```

## For Full Pipeline (After recognition weights are ready)

```powershell
# From root directory
python run_pipeline.py --source test_image.jpg
```

## Training Progress

- ✅ Detection: YOLOv8 model training
- 🟡 Recognition: ResNet50 weights pending
- ⏳ Pipeline: Awaiting recognition model

See README.md for complete documentation.
