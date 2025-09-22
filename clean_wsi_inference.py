#!/usr/bin/env python3
"""
Full WSI inference system with proper memory management and coordinate tracking.
Processes entire WSI slides with the trained YOLOv8 glomeruli detection model.
"""

import os
import gc
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from ultralytics import YOLO
import openslide
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional, Generator
import logging
from dataclasses import dataclass, asdict
import time
import json
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GlomerulusDetection:
    """Store complete glomerulus detection information."""
    detection_id: int
    wsi_x1: int
    wsi_y1: int
    wsi_x2: int
    wsi_y2: int
    center_x: int
    center_y: int
    width: int
    height: int
    area: int
    confidence: float
    class_id: int
    class_name: str
    patch_x: int
    patch_y: int


class WSIGlomeruliDetector:
    """Full WSI glomeruli detection system with optimized processing."""
    
    def __init__(self,
                 model_path: str,
                 patch_size: int = 1024,
                 overlap: int = 128,
                 batch_size: int = 8,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 tissue_threshold: float = 0.1,
                 device: str = None):
        """
        Initialize WSI detector.
        
        Args:
            model_path: Path to trained YOLOv8 model
            patch_size: Size of patches (must match training)
            overlap: Overlap between patches
            batch_size: Number of patches to process at once
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            tissue_threshold: Minimum tissue percentage in patch
            device: Device for inference (auto-detect if None)
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.tissue_threshold = tissue_threshold
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Class information
        self.class_names = ['normal', 'sclerotic']
        self.class_colors = {'normal': '#00FF00', 'sclerotic': '#FF0000'}
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")
        
        # Statistics
        self.stats = {
            'total_patches': 0,
            'tissue_patches': 0,
            'skipped_patches': 0,
            'total_detections': 0,
            'detections_by_class': {name: 0 for name in self.class_names}
        }
    
    def check_tissue_content(self, patch: np.ndarray) -> bool:
        """
        Check if patch contains sufficient tissue.
        
        Args:
            patch: Image patch
            
        Returns:
            True if patch contains tissue
        """
        # Convert to grayscale
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        else:
            gray = patch
        
        # Apply Otsu's threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate tissue percentage
        tissue_pixels = np.sum(binary < 128)  # Assuming tissue is darker
        total_pixels = binary.size
        tissue_ratio = tissue_pixels / total_pixels
        
        return tissue_ratio > self.tissue_threshold
    
    def generate_patches_from_wsi(self, 
                                 slide: openslide.OpenSlide,
                                 level: int = 0) -> Generator:
        """
        Generate patches from WSI with memory-efficient streaming.
        
        Args:
            slide: OpenSlide object
            level: Pyramid level to use
            
        Yields:
            Batches of (patches, coordinates)
        """
        width, height = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        
        # Calculate total patches for progress bar
        n_patches_x = (width - self.patch_size) // self.stride + 1
        n_patches_y = (height - self.patch_size) // self.stride + 1
        total_patches = n_patches_x * n_patches_y
        
        logger.info(f"WSI dimensions: {width}x{height} at level {level}")
        logger.info(f"Expected patches: ~{total_patches}")
        
        batch_patches = []
        batch_coords = []
        
        with tqdm(total=total_patches, desc="Processing patches") as pbar:
            # Iterate through patches
            for y in range(0, height - self.patch_size + 1, self.stride):
                for x in range(0, width - self.patch_size + 1, self.stride):
                    # Read patch from WSI
                    location = (int(x * downsample), int(y * downsample))
                    patch = slide.read_region(location, level, (self.patch_size, self.patch_size))
                    patch = patch.convert('RGB')
                    patch_array = np.array(patch)
                    
                    self.stats['total_patches'] += 1
                    
                    # Check tissue content
                    if not self.check_tissue_content(patch_array):
                        self.stats['skipped_patches'] += 1
                        pbar.update(1)
                        continue
                    
                    self.stats['tissue_patches'] += 1
                    batch_patches.append(patch_array)
                    batch_coords.append((x, y))
                    
                    # Yield batch when full
                    if len(batch_patches) >= self.batch_size:
                        yield batch_patches, batch_coords
                        batch_patches = []
                        batch_coords = []
                    
                    pbar.update(1)
            
            # Handle edge patches
            # Right edge
            if width % self.stride != 0:
                x = width - self.patch_size
                for y in range(0, height - self.patch_size + 1, self.stride):
                    location = (int(x * downsample), int(y * downsample))
                    patch = slide.read_region(location, level, (self.patch_size, self.patch_size))
                    patch = patch.convert('RGB')
                    patch_array = np.array(patch)
                    
                    if self.check_tissue_content(patch_array):
                        batch_patches.append(patch_array)
                        batch_coords.append((x, y))
                        
                        if len(batch_patches) >= self.batch_size:
                            yield batch_patches, batch_coords
                            batch_patches = []
                            batch_coords = []
            
            # Bottom edge
            if height % self.stride != 0:
                y = height - self.patch_size
                for x in range(0, width - self.patch_size + 1, self.stride):
                    location = (int(x * downsample), int(y * downsample))
                    patch = slide.read_region(location, level, (self.patch_size, self.patch_size))
                    patch = patch.convert('RGB')
                    patch_array = np.array(patch)
                    
                    if self.check_tissue_content(patch_array):
                        batch_patches.append(patch_array)
                        batch_coords.append((x, y))
                        
                        if len(batch_patches) >= self.batch_size:
                            yield batch_patches, batch_coords
                            batch_patches = []
                            batch_coords = []
        
        # Yield remaining patches
        if batch_patches:
            yield batch_patches, batch_coords
    
    def process_batch(self, patches: List[np.ndarray]) -> List[List[Dict]]:
        """
        Process a batch of patches through the model.
        
        Args:
            patches: List of patch arrays
            
        Returns:
            List of detections for each patch
        """
        # Run inference
        results = self.model.predict(
            patches,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.patch_size,
            verbose=False,
            device=self.device
        )
        
        # Parse results
        batch_detections = []
        for result in results:
            patch_detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, classes):
                    patch_detections.append({
                        'bbox': box.astype(int),
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': self.class_names[int(cls)]
                    })
            
            batch_detections.append(patch_detections)
        
        return batch_detections
    
    def apply_global_nms(self, detections: List[GlomerulusDetection]) -> List[GlomerulusDetection]:
        """
        Apply NMS across all detections to remove duplicates.
        
        Args:
            detections: All detections from WSI
            
        Returns:
            Filtered detections
        """
        if not detections:
            return []
        
        logger.info(f"Applying NMS to {len(detections)} detections")
        
        # Group by class
        class_groups = {}
        for det in detections:
            if det.class_id not in class_groups:
                class_groups[det.class_id] = []
            class_groups[det.class_id].append(det)
        
        filtered = []
        
        for class_id, class_dets in class_groups.items():
            if not class_dets:
                continue
            
            # Convert to arrays
            boxes = np.array([[d.wsi_x1, d.wsi_y1, d.wsi_x2, d.wsi_y2] for d in class_dets])
            scores = np.array([d.confidence for d in class_dets])
            
            # Apply NMS
            indices = self._nms_numpy(boxes, scores, self.iou_threshold)
            
            # Keep selected
            for idx in indices:
                filtered.append(class_dets[idx])
        
        logger.info(f"After NMS: {len(filtered)} detections")
        return filtered
    
    def _nms_numpy(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """NumPy NMS implementation."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def process_wsi(self, wsi_path: str, output_dir: str = None) -> Dict:
        """
        Process entire WSI for glomeruli detection.
        
        Args:
            wsi_path: Path to WSI file
            output_dir: Directory for results (auto-created if None)
            
        Returns:
            Dictionary with results and statistics
        """
        start_time = time.time()
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path("wsi_inference_results") / Path(wsi_path).stem
        else:
            output_dir = Path(output_dir) / Path(wsi_path).stem
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset statistics
        self.stats = {
            'total_patches': 0,
            'tissue_patches': 0,
            'skipped_patches': 0,
            'total_detections': 0,
            'detections_by_class': {name: 0 for name in self.class_names}
        }
        
        logger.info(f"Processing WSI: {wsi_path}")
        
        # Open WSI
        slide = openslide.OpenSlide(wsi_path)
        wsi_width, wsi_height = slide.dimensions
        
        logger.info(f"WSI dimensions: {wsi_width}x{wsi_height}")
        logger.info(f"Levels: {slide.level_count}")
        
        # Collect all detections
        all_detections = []
        detection_id = 0
        batch_count = 0
        
        # Process patches
        for batch_patches, batch_coords in self.generate_patches_from_wsi(slide):
            batch_count += 1
            
            # Run inference
            batch_results = self.process_batch(batch_patches)
            
            # Process detections
            for patch_detections, (patch_x, patch_y) in zip(batch_results, batch_coords):
                for det in patch_detections:
                    x1, y1, x2, y2 = det['bbox']
                    
                    # Convert to WSI coordinates
                    wsi_x1 = patch_x + x1
                    wsi_y1 = patch_y + y1
                    wsi_x2 = patch_x + x2
                    wsi_y2 = patch_y + y2
                    
                    detection = GlomerulusDetection(
                        detection_id=detection_id,
                        wsi_x1=wsi_x1,
                        wsi_y1=wsi_y1,
                        wsi_x2=wsi_x2,
                        wsi_y2=wsi_y2,
                        center_x=(wsi_x1 + wsi_x2) // 2,
                        center_y=(wsi_y1 + wsi_y2) // 2,
                        width=wsi_x2 - wsi_x1,
                        height=wsi_y2 - wsi_y1,
                        area=(wsi_x2 - wsi_x1) * (wsi_y2 - wsi_y1),
                        confidence=det['confidence'],
                        class_id=det['class_id'],
                        class_name=det['class_name'],
                        patch_x=patch_x,
                        patch_y=patch_y
                    )
                    
                    all_detections.append(detection)
                    detection_id += 1
            
            # Memory cleanup
            if batch_count % 10 == 0:
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                logger.info(f"Processed {batch_count} batches, {len(all_detections)} detections so far")
        
        # Apply global NMS
        filtered_detections = self.apply_global_nms(all_detections)
        
        # Update statistics
        self.stats['total_detections'] = len(filtered_detections)
        for det in filtered_detections:
            self.stats['detections_by_class'][det.class_name] += 1
        
        # Save results
        self._save_results(filtered_detections, output_dir, wsi_path, slide)
        
        # Create visualizations
        self._create_overview_map(filtered_detections, slide, output_dir)
        
        slide.close()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare results
        results = {
            'wsi_path': wsi_path,
            'wsi_dimensions': (wsi_width, wsi_height),
            'processing_time': processing_time,
            'statistics': self.stats,
            'output_directory': str(output_dir),
            'num_detections': len(filtered_detections),
            'detections': filtered_detections
        }
        
        logger.info(f"Processing complete in {processing_time:.2f} seconds")
        logger.info(f"Total detections: {len(filtered_detections)}")
        for class_name, count in self.stats['detections_by_class'].items():
            logger.info(f"  {class_name}: {count}")
        
        return results
    
    def _save_results(self, detections: List[GlomerulusDetection], 
                     output_dir: Path, wsi_path: str, slide: openslide.OpenSlide):
        """Save detection results in multiple formats."""
        
        # Save as CSV
        df_data = []
        for det in detections:
            df_data.append({
                'detection_id': det.detection_id,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'center_x': det.center_x,
                'center_y': det.center_y,
                'width': det.width,
                'height': det.height,
                'area': det.area,
                'x1': det.wsi_x1,
                'y1': det.wsi_y1,
                'x2': det.wsi_x2,
                'y2': det.wsi_y2
            })
        
        df = pd.DataFrame(df_data)
        csv_path = output_dir / 'detections.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
        
        # Save as JSON with proper type conversion
        json_data = {
            'wsi_path': str(wsi_path),
            'wsi_dimensions': list(slide.dimensions),
            'model_config': {
                'patch_size': int(self.patch_size),
                'overlap': int(self.overlap),
                'confidence_threshold': float(self.confidence_threshold),
                'iou_threshold': float(self.iou_threshold)
            },
            'statistics': self.stats,
            'detections': [{
                k: int(v) if isinstance(v, (np.integer, np.int64)) else 
                   float(v) if isinstance(v, (np.floating, np.float64)) else v
                for k, v in asdict(det).items()
            } for det in detections]
        }
        
        json_path = output_dir / 'detections.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved JSON: {json_path}")
        
        # Save summary
        summary_path = output_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"WSI Glomeruli Detection Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"WSI: {wsi_path}\n")
            f.write(f"Dimensions: {slide.dimensions[0]}x{slide.dimensions[1]}\n")
            f.write(f"Total patches: {self.stats['total_patches']}\n")
            f.write(f"Tissue patches: {self.stats['tissue_patches']}\n")
            f.write(f"Skipped patches: {self.stats['skipped_patches']}\n")
            f.write(f"Total detections: {len(detections)}\n")
            f.write(f"\nDetections by class:\n")
            for class_name, count in self.stats['detections_by_class'].items():
                f.write(f"  {class_name}: {count}\n")
        logger.info(f"Saved summary: {summary_path}")
    
    def _create_overview_map(self, detections: List[GlomerulusDetection],
                           slide: openslide.OpenSlide, output_dir: Path):
        """Create overview visualization of detections."""
        
        # Calculate thumbnail size
        wsi_width, wsi_height = slide.dimensions
        max_dim = 4000
        if wsi_width > wsi_height:
            thumb_width = min(max_dim, wsi_width)
            thumb_height = int(thumb_width * wsi_height / wsi_width)
        else:
            thumb_height = min(max_dim, wsi_height)
            thumb_width = int(thumb_height * wsi_width / wsi_height)
        
        scale_x = thumb_width / wsi_width
        scale_y = thumb_height / wsi_height
        
        logger.info(f"Creating overview map ({thumb_width}x{thumb_height})")
        
        # Get thumbnail
        thumbnail = slide.get_thumbnail((thumb_width, thumb_height))
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 20 * thumb_height / thumb_width))
        ax.imshow(thumbnail)
        
        # Plot detections
        colors = {'normal': 'green', 'sclerotic': 'red'}
        for det in detections:
            # Scale coordinates
            x1 = det.wsi_x1 * scale_x
            y1 = det.wsi_y1 * scale_y
            x2 = det.wsi_x2 * scale_x
            y2 = det.wsi_y2 * scale_y
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1,
                edgecolor=colors.get(det.class_name, 'blue'),
                facecolor='none',
                alpha=0.7
            )
            ax.add_patch(rect)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='green', label=f'Normal ({self.stats["detections_by_class"]["normal"]})'),
            Patch(facecolor='none', edgecolor='red', label=f'Sclerotic ({self.stats["detections_by_class"]["sclerotic"]})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        ax.set_title(f'Glomeruli Detections - Total: {len(detections)}', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        overview_path = output_dir / 'overview_map.png'
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved overview map: {overview_path}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='WSI Glomeruli Detection')
    parser.add_argument('--wsi', required=True, help='Path to WSI file')
    parser.add_argument('--model', default='runs/detect/glomeruli_batch8_test/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--output', default='wsi_inference_results',
                       help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--overlap', type=int, default=128,
                       help='Overlap between patches')
    parser.add_argument('--device', default=None,
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return
    
    # Check if WSI exists
    if not Path(args.wsi).exists():
        logger.error(f"WSI not found: {args.wsi}")
        return
    
    # Initialize detector
    detector = WSIGlomeruliDetector(
        model_path=args.model,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        overlap=args.overlap,
        device=args.device
    )
    
    # Process WSI
    results = detector.process_wsi(args.wsi, args.output)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"WSI PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processing time: {results['processing_time']:.2f} seconds")
    print(f"Total detections: {results['num_detections']}")
    print(f"Output saved to: {results['output_directory']}")


if __name__ == "__main__":
    main()