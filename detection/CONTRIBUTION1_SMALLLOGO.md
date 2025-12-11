# Contribution #1: Small-Logo-Aware Training

## Overview

This contribution enhances the baseline YOLOv8 detector to better detect small logos through:
1. **Higher resolution training** (960×960 vs 640×640 baseline)
2. **Copy-paste augmentation** to synthesize more small logo instances
3. **Stronger data augmentation** (scale jitter, translation, rotation, color)

**Key principle:** Improve small logo detection through training-time improvements while keeping the same YOLOv8-s architecture.

---

## What Changed from Baseline

| Parameter | Baseline | Contribution #1 | Justification |
|-----------|----------|-----------------|---------------|
| **Resolution** | 640×640 | 960×960 | Small logos need more pixels |
| **Copy-Paste** | ✗ (0.0) | ✓ (0.3) | Synthesize crowded scenes with more small logos |
| **Scale Jitter** | 0.5 | 0.7 | Handle varying logo sizes better |
| **Translation** | 0.1 | 0.2 | More spatial variation |
| **Rotation** | 0° | ±10° | Logos appear at slight angles |
| **Batch Size** | 32 | 16 | GPU memory constraint with larger images |

All other settings (optimizer, learning rate, epochs, HSV augmentation) remain the same.

---

## How to Run

### Option 1: From COCO Pretrained (Same as Baseline)

```bash
cd detection
python train_yolo_smalllogo.py
```

This starts from `yolov8s.pt` (COCO pretrained), same initialization as baseline.

### Option 2: Fine-tune from Baseline

If you want to start from your baseline checkpoint instead:

1. Open `detection/train_yolo_smalllogo.py`
2. Comment out line 17: `# model = YOLO('yolov8s.pt')`
3. Uncomment line 20: `model = YOLO('runs/detect/logodet3k_yolov8s_baseline50/weights/best.pt')`
4. Run: `python train_yolo_smalllogo.py`

---

## Training Configuration

**Location:** `configs/detection_smalllogo_config.yaml`

Key augmentation parameters:
- `mosaic: 1.0` - Mix 4 images per training sample
- `copy_paste: 0.3` - 30% chance to paste logo instances from other images
- `scale: 0.7` - Scale images from 0.3× to 1.7× (vs 0.5× to 1.5× baseline)
- `translate: 0.2` - Shift images up to ±20% (vs ±10% baseline)
- `degrees: 10.0` - Rotate ±10° (vs 0° baseline)

---

## Expected Outputs

After training completes, you'll find:

```
detection/runs/detect/logodet3k_yolov8n_smalllogo_contrib1/
├── weights/
│   ├── best.pt          # Best checkpoint (use for evaluation)
│   └── last.pt          # Last epoch checkpoint
├── results.csv          # Training metrics per epoch
├── results.png          # Training curves
└── confusion_matrix.png # Confusion matrix
```

---

## For Your Thesis

### Methodology Section

> Building on the baseline YOLOv8 detector, we introduce a **small-logo-aware training setup** as our first contribution.
> 
> First, we increase the input resolution from 640×640 to 960×960 to better preserve small logos that may occupy only a few dozen pixels in the original image.
> 
> Second, we adopt **stronger data augmentation** tailored for small objects:
> - **Copy-paste augmentation** (probability 0.3) synthesizes crowded scenes by pasting logo instances from other images, increasing the effective count of small training examples.
> - **Stronger scale jittering** (range 0.3× to 1.7×) helps the model handle logos at varying distances.
> - **Increased translation** (±20% vs ±10%) and **small rotation** (±10°) add spatial variation while preserving logo readability.
> - **Mosaic augmentation** (enabled by default) mixes four images to create diverse contextual backgrounds.
> 
> All other training settings—optimizer (AdamW), learning rate schedule, and number of epochs—are kept identical to the baseline to isolate the effect of our resolution and augmentation changes.

### Experiments Table

| Model | Epochs | Input Size | Copy-Paste | Aug Changes | mAP@0.5 | mAP@0.5:0.95 | Notes |
|-------|--------|------------|------------|-------------|---------|--------------|-------|
| **Baseline YOLOv8n** | 40 | 640×640 | ✗ | default | [from results.csv] | [from results.csv] | Reference |
| **YOLOv8n + small-logo aug (ours)** | 40 | 960×960 | ✓ | stronger scale/translate/rotate | [run & fill in] | [run & fill in] | Contribution #1 |

Fill in the metrics after training by checking `results.csv` (best epoch row).

---

## GPU Memory Notes

- **960×960 training** requires ~8-12 GB VRAM with batch size 16
- If you get OOM (out of memory) errors:
  - Reduce batch size: `batch=8` or `batch=4`
  - Or reduce resolution: `imgsz=832` or `imgsz=768`
  - Document the final resolution you used in the thesis

---

## Next Steps

After this training completes:
1. **Compare metrics** with baseline (from `results.csv`)
2. **Contribution #2:** Implement tiling inference to further improve small logo recall
3. **(Optional) Contribution #3:** Add P2 head for architectural improvement
