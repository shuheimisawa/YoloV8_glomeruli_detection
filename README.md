# Two-Stage Glomeruli Detection System

Clean implementation of YOLOv8-based glomeruli detection for Whole Slide Images (WSI).

## Project Structure

```
.
├── Data/                           # Dataset directory
│   ├── raw_slides/                 # WSI files (.svs)
│   └── annotations/                # GeoJSON annotations
├── yolo_dataset/                   # Prepared YOLO dataset
├── runs/detect/                    # Training results
│   └── glomeruli_batch8_test/
│       └── weights/best.pt        # Trained model weights
├── clean_inference_test.py        # Test script for small WSI sections
├── clean_wsi_inference.py         # Full WSI inference system
├── train_yolo.py                  # Model training script
└── complete_yolo_prep.py          # Dataset preparation script
```

## Usage

### 1. Data Preparation
```bash
python complete_yolo_prep.py
```
Creates YOLO-format dataset from WSI slides and GeoJSON annotations.

### 2. Training
```bash
python train_yolo.py --epochs 100 --batch 8
```
Trains YOLOv8m model for glomeruli detection (normal vs sclerotic).

### 3. Inference

#### Test on small section:
```bash
python clean_inference_test.py
```

#### Process full WSI:
```bash
python clean_wsi_inference.py --wsi Data/raw_slides/TPATH002.svs --confidence 0.3
```

## Model Details

- **Architecture**: YOLOv8m (medium)
- **Input size**: 1024×1024 patches
- **Classes**: 2 (normal, sclerotic)
- **Training**: 100 epochs with early stopping

## Inference Pipeline

1. **Patch extraction**: Sliding window (1024×1024) with 128px overlap
2. **Tissue detection**: Otsu thresholding to filter background
3. **Batch processing**: 4-8 patches at a time for memory efficiency
4. **NMS**: Remove duplicate detections from overlapping patches
5. **Output**: CSV, JSON, and visualization maps

## Key Features

- Proper memory management with batch processing and cache clearing
- Accurate coordinate mapping from patches to WSI space
- Non-Maximum Suppression for handling overlapping detections
- Multiple output formats for analysis
- GPU acceleration with automatic fallback to CPU

## Performance

- Processes 32768×40960 WSI in ~47 seconds
- Memory-efficient streaming of patches
- Automatic garbage collection and GPU cache management