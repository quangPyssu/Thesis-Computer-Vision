# Contribution #2: Tiling Inference for Tiny Logos

## Overview

This contribution improves detection of extremely small logos by changing only the **inference strategy**—no retraining required. At test time, large images are split into overlapping tiles, each tile is processed independently, and predictions are fused with global NMS.

**Key principle:** Increase effective resolution for small objects at inference time while keeping the same trained model.

---

## What Changed from Baseline/Contribution #1

| Aspect | Single-Pass (Baseline) | Tiled Inference (Contribution #2) |
|--------|------------------------|-----------------------------------|
| **Training** | Same | Same (no retraining) |
| **Inference strategy** | Full image in one pass | Split into overlapping tiles |
| **Effective resolution** | Fixed (640 or 960) | Higher per tile (1024+) |
| **Processing** | 1 inference per image | N inferences (N = number of tiles) |
| **Prediction fusion** | None | Global NMS across all tiles |
| **Speed** | Fast | Slower (proportional to tile count) |
| **Small logo recall** | Baseline | Improved (higher resolution per region) |

---

## How It Works

### 1. Tile Creation
- Input image split into overlapping tiles (e.g., 1024×1024)
- Overlap (e.g., 20%) ensures logos near tile boundaries are captured
- Small images (< tile size) processed directly without tiling

### 2. Independent Detection
- Run detector on each tile independently
- Each tile processed at full resolution (1024×1024)
- Logos that were tiny in full image now occupy more pixels

### 3. Coordinate Remapping
- Map tile-local coordinates back to original image space
- Track which tile each detection came from

### 4. Global NMS Fusion
- Collect all detections from all tiles
- Apply Non-Maximum Suppression globally
- Merge overlapping predictions (same logo detected in multiple tiles)
- Filter by confidence threshold

---

## Implementation

### Files Created

1. **`detection/inference_tiled.py`** - Core tiling inference implementation
   - `TiledInference` class with configurable tile size and overlap
   - Automatic tile creation with intelligent positioning
   - Global NMS fusion across tiles
   - Visualization support

2. **`detection/evaluate_tiled.py`** - Evaluation and comparison script
   - Compare single-pass vs tiled inference
   - Measure detection statistics and timing
   - Export results for analysis

---

## Usage

### Single Image Inference

```bash
cd detection

# Run tiled inference on one image
python inference_tiled.py \
  --model runs/detect/logodet3k_yolov8n_baseline/weights/best.pt \
  --image ../data/logodet3k_yolo/images/test/test_000001.jpg \
  --tile-size 1024 \
  --overlap 0.2 \
  --output tiled_result.jpg \
  --verbose
```

**Parameters:**
- `--tile-size`: Size of each tile (default: 1024)
- `--overlap`: Overlap ratio 0-1 (default: 0.2 = 20%)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: NMS IoU threshold (default: 0.45)

### Evaluate on Test Set

```bash
# Compare single-pass vs tiled on full test set
python evaluate_tiled.py \
  --model runs/detect/logodet3k_yolov8n_smalllogo_contrib1/weights/best.pt \
  --data ../data/logodet3k_yolo/data.yaml \
  --tile-size 1024 \
  --overlap 0.2 \
  --output runs/tiled_comparison.json
```

This will:
1. Run standard single-pass inference (baseline)
2. Run tiled inference on same test set
3. Report detection statistics and timing
4. Save results to JSON

---

## Expected Results

### Detection Statistics

Tiled inference typically shows:
- **Higher recall** on small logos (< 32×32 pixels)
- **More detections** before NMS (due to multiple tiles)
- **Similar final detection count** after global NMS
- **Slower inference** (proportional to number of tiles)

### Speed Trade-off

For a 2000×2000 image with 1024×1024 tiles and 20% overlap:
- Number of tiles: ~4-6 tiles
- Inference time: ~3-5× slower than single-pass
- Recall improvement: +5-15% on small objects (typical)

---

## For Your Thesis

### Methodology Section

> To further improve recall on extremely small logos, we introduce a **tiling-based inference strategy** as our second contribution.
> 
> At test time, each input image is decomposed into overlapping tiles of size 1024×1024 with 20% overlap. The detector is run independently on each tile, effectively increasing the receptive resolution for small objects—a logo that occupies only 16×16 pixels in a 2048×2048 image occupies 32×32 pixels when its containing tile is resized to 1024×1024.
> 
> After processing all tiles, detections are reprojected to the original image coordinates. We apply **global Non-Maximum Suppression (NMS)** across all tiles to merge overlapping predictions of the same logo instance, using an IoU threshold of 0.45.
> 
> Crucially, this approach requires **no retraining**—it can be applied to any trained detector (baseline or Contribution #1) purely as an inference-time modification. The trade-off is increased computational cost proportional to the number of tiles.

### Experiments Table

| Model | Training | Input / Inference | Tiling | mAP@0.5 | mAP@0.5:0.95 | Inference Time | Comment |
|-------|----------|-------------------|--------|---------|--------------|----------------|---------|
| Baseline YOLOv8n | 640, 40ep | 640, full image | ✗ | [baseline] | [baseline] | 1.0× | Reference |
| YOLOv8n + small-logo aug | 960, 40ep | 960, full image | ✗ | [contrib1] | [contrib1] | 1.2× | Contribution #1 |
| YOLOv8n + small-logo aug | 960, 40ep | 1024-tile, 20% overlap | ✓ | [contrib2] | [contrib2] | 3-5× | **Contribution #2** |

**How to fill in metrics:**
1. Run `evaluate_tiled.py` to get single-pass metrics (mAP from YOLO validation)
2. For tiled metrics, you need to either:
   - Use detection statistics as proxy (detection count, confidence distribution)
   - OR implement proper mAP calculation with ground truth matching
   - OR manually run YOLO validation with tiled predictions

### Results Discussion Points

- **Small logo performance**: "Tiled inference improved recall on logos smaller than 32×32 pixels by X%, as these objects now occupy a larger fraction of the tile resolution."

- **Speed trade-off**: "The computational cost increased by ~Nx due to processing N tiles per image, making tiled inference more suitable for offline evaluation than real-time applications."

- **When to use**: "Tiling is most beneficial for high-resolution images (> 1500px) with many small logos. For standard resolution images, single-pass inference remains more efficient."

---

## Python API Usage

For integration into your pipeline:

```python
from detection.inference_tiled import TiledInference

# Initialize
tiler = TiledInference(
    model_path='runs/detect/best.pt',
    tile_size=1024,
    overlap=0.2
)

# Run on single image
result = tiler.predict_tiled('path/to/image.jpg', verbose=True)

# Access detections: [x1, y1, x2, y2, conf, class]
detections = result['detections']
print(f"Found {len(detections)} logos")

# Visualize
tiler.visualize_detection(
    'path/to/image.jpg',
    detections,
    output_path='result.jpg'
)
```

---

## Tuning Parameters

### Tile Size
- **Smaller tiles (512)**: More tiles, slower, better for very small logos
- **Larger tiles (1536)**: Fewer tiles, faster, may miss tiny logos
- **Default (1024)**: Good balance for most cases

### Overlap
- **No overlap (0.0)**: Fastest, but logos on boundaries may be split
- **High overlap (0.3-0.5)**: Better boundary handling, but slower
- **Default (0.2)**: Good compromise

### When to Adjust
- **High-res images (> 2000px)**: Use larger tiles (1536) or more overlap
- **Many tiny logos**: Use smaller tiles (768-896)
- **Speed critical**: Reduce overlap to 0.1 or use adaptive tiling

---

## Limitations & Future Work

1. **Computational cost**: Linear increase with number of tiles
   - Future: Adaptive tiling (only tile regions with potential small objects)

2. **mAP calculation**: Requires matching predictions with ground truth
   - Future: Implement proper evaluation pipeline with pycocotools

3. **Memory usage**: All tiles processed sequentially
   - Future: Batch processing of tiles (if GPU memory allows)

4. **Boundary effects**: Logos on tile boundaries detected multiple times
   - Mitigated by: overlap + NMS, but not perfect
   - Future: Weighted fusion based on distance from tile center

---

## Next Steps

1. **Run evaluation**: Use `evaluate_tiled.py` on your test set
2. **Compare metrics**: Fill in the thesis table with your results
3. **Analyze improvements**: Look at per-class performance, especially on small logos
4. **(Optional) Contribution #3**: Implement P2 head for architectural improvement

---

## Troubleshooting

**Issue: Out of memory during tiling**
- Reduce tile size: `--tile-size 768`
- Process tiles one at a time (already default)

**Issue: Too many duplicate detections**
- Increase NMS threshold: `--iou 0.5` or `0.6`
- Reduce overlap: `--overlap 0.1`

**Issue: Missing small logos**
- Increase overlap: `--overlap 0.3`
- Reduce tile size: `--tile-size 896`
- Lower confidence threshold: `--conf 0.15`
