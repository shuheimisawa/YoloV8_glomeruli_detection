#!/usr/bin/env python3
"""
Clean inference test script for glomeruli detection on small WSI sections.
Tests the trained YOLOv8 model with proper memory management and coordinate mapping.
"""

import os
import gc
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
import openslide
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Detection:
    """Store detection information."""
    x1: int  # WSI coordinates
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    patch_origin: Tuple[int, int]  # Where the patch was extracted from


class CleanInferenceEngine:
    """Clean inference engine with proper model loading and memory management."""
    
    def __init__(self, 
                 model_path: str,
                 patch_size: int = 1024,
                 overlap: int = 128,
                 batch_size: int = 4,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained YOLOv8 model weights
            patch_size: Size of patches (must match training)
            overlap: Overlap between patches
            batch_size: Number of patches to process at once
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Class names from training
        self.class_names = ['normal', 'sclerotic']
        
        # Load model
        logging.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        logging.info(f"Model loaded on {device}")
        
        # Verify model configuration
        logging.info(f"Model info: {self.model.model.names}")
        
    def extract_roi_from_wsi(self, 
                            wsi_path: str, 
                            x: int, 
                            y: int, 
                            width: int, 
                            height: int) -> np.ndarray:
        """
        Extract a region of interest from WSI.
        
        Args:
            wsi_path: Path to WSI file
            x, y: Top-left corner of ROI in WSI coordinates
            width, height: Size of ROI
            
        Returns:
            ROI as numpy array
        """
        slide = openslide.OpenSlide(wsi_path)
        roi = slide.read_region((x, y), 0, (width, height))
        roi = roi.convert('RGB')
        roi_array = np.array(roi)
        slide.close()
        return roi_array
    
    def generate_patches(self, 
                        image: np.ndarray, 
                        roi_offset: Tuple[int, int] = (0, 0)) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Generate patches from an image with proper coordinate tracking.
        
        Args:
            image: Input image (ROI or full image)
            roi_offset: Offset of ROI in WSI coordinates
            
        Yields:
            (patch, (x, y)) where x,y are WSI coordinates
        """
        height, width = image.shape[:2]
        patches_batch = []
        coords_batch = []
        
        # Generate patches with stride
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                wsi_coords = (x + roi_offset[0], y + roi_offset[1])
                
                patches_batch.append(patch)
                coords_batch.append(wsi_coords)
                
                # Yield batch when full
                if len(patches_batch) >= self.batch_size:
                    yield patches_batch, coords_batch
                    patches_batch = []
                    coords_batch = []
        
        # Handle edge patches
        # Right edge
        if width % self.stride != 0:
            x = width - self.patch_size
            for y in range(0, height - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                wsi_coords = (x + roi_offset[0], y + roi_offset[1])
                patches_batch.append(patch)
                coords_batch.append(wsi_coords)
                
                if len(patches_batch) >= self.batch_size:
                    yield patches_batch, coords_batch
                    patches_batch = []
                    coords_batch = []
        
        # Bottom edge
        if height % self.stride != 0:
            y = height - self.patch_size
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                wsi_coords = (x + roi_offset[0], y + roi_offset[1])
                patches_batch.append(patch)
                coords_batch.append(wsi_coords)
                
                if len(patches_batch) >= self.batch_size:
                    yield patches_batch, coords_batch
                    patches_batch = []
                    coords_batch = []
        
        # Yield remaining patches
        if patches_batch:
            yield patches_batch, coords_batch
    
    def run_inference_on_batch(self, patches: List[np.ndarray]) -> List[List[Dict]]:
        """
        Run inference on a batch of patches.
        
        Args:
            patches: List of patch images
            
        Returns:
            List of detections for each patch
        """
        # Run inference
        results = self.model.predict(
            patches,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.patch_size,
            verbose=False
        )
        
        # Parse results
        batch_detections = []
        for result in results:
            patch_detections = []
            if result.boxes is not None:
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
    
    def process_roi(self, 
                   wsi_path: str,
                   roi_x: int,
                   roi_y: int,
                   roi_width: int,
                   roi_height: int) -> List[Detection]:
        """
        Process a region of interest from WSI.
        
        Args:
            wsi_path: Path to WSI file
            roi_x, roi_y: Top-left corner of ROI
            roi_width, roi_height: Size of ROI
            
        Returns:
            List of detections in WSI coordinates
        """
        logging.info(f"Processing ROI: ({roi_x}, {roi_y}) size: {roi_width}x{roi_height}")
        
        # Extract ROI
        roi = self.extract_roi_from_wsi(wsi_path, roi_x, roi_y, roi_width, roi_height)
        logging.info(f"ROI extracted: shape={roi.shape}")
        
        all_detections = []
        batch_count = 0
        total_patches = 0
        
        # Process patches
        for patches_batch, coords_batch in self.generate_patches(roi, (roi_x, roi_y)):
            batch_count += 1
            total_patches += len(patches_batch)
            
            logging.info(f"Processing batch {batch_count}: {len(patches_batch)} patches")
            
            # Run inference
            batch_detections = self.run_inference_on_batch(patches_batch)
            
            # Convert to WSI coordinates
            for patch_detections, patch_coords in zip(batch_detections, coords_batch):
                patch_x, patch_y = patch_coords
                
                for det in patch_detections:
                    x1, y1, x2, y2 = det['bbox']
                    
                    # Convert to WSI coordinates
                    detection = Detection(
                        x1=patch_x + x1,
                        y1=patch_y + y1,
                        x2=patch_x + x2,
                        y2=patch_y + y2,
                        confidence=det['confidence'],
                        class_id=det['class_id'],
                        class_name=det['class_name'],
                        patch_origin=patch_coords
                    )
                    all_detections.append(detection)
            
            # Clear memory
            if batch_count % 5 == 0:
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        logging.info(f"Processed {total_patches} patches, found {len(all_detections)} detections")
        
        # Apply NMS to remove duplicates from overlapping patches
        filtered_detections = self.apply_nms(all_detections)
        logging.info(f"After NMS: {len(filtered_detections)} detections")
        
        return filtered_detections
    
    def apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            detections: List of all detections
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Separate by class
        class_detections = {}
        for det in detections:
            if det.class_id not in class_detections:
                class_detections[det.class_id] = []
            class_detections[det.class_id].append(det)
        
        filtered = []
        
        # Apply NMS per class
        for class_id, class_dets in class_detections.items():
            if not class_dets:
                continue
            
            # Convert to numpy arrays
            boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in class_dets])
            scores = np.array([d.confidence for d in class_dets])
            
            # Apply NMS
            indices = self.nms_numpy(boxes, scores, self.iou_threshold)
            
            # Keep selected detections
            for idx in indices:
                filtered.append(class_dets[idx])
        
        return filtered
    
    def nms_numpy(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        NumPy implementation of Non-Maximum Suppression.
        
        Args:
            boxes: Array of boxes (x1, y1, x2, y2)
            scores: Array of confidence scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Indices of boxes to keep
        """
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
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def visualize_detections(self,
                            roi: np.ndarray,
                            detections: List[Detection],
                            roi_offset: Tuple[int, int],
                            save_path: str):
        """
        Visualize detections on ROI.
        
        Args:
            roi: ROI image
            detections: List of detections
            roi_offset: Offset of ROI in WSI
            save_path: Path to save visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.imshow(roi)
        
        # Color map for classes
        colors = {'normal': 'green', 'sclerotic': 'red'}
        
        # Draw detections
        for det in detections:
            # Convert WSI coordinates to ROI coordinates
            x1 = det.x1 - roi_offset[0]
            y1 = det.y1 - roi_offset[1]
            x2 = det.x2 - roi_offset[0]
            y2 = det.y2 - roi_offset[1]
            
            # Skip if outside ROI
            if x1 < 0 or y1 < 0 or x2 > roi.shape[1] or y2 > roi.shape[0]:
                continue
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, 
                edgecolor=colors.get(det.class_name, 'blue'),
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{det.class_name} {det.confidence:.2f}"
            ax.text(x1, y1 - 5, label, 
                   color=colors.get(det.class_name, 'blue'),
                   fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor=colors.get(det.class_name, 'blue'),
                           alpha=0.7))
        
        # Add grid for patch boundaries (optional)
        for i in range(0, roi.shape[1], self.stride):
            ax.axvline(x=i, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
        for i in range(0, roi.shape[0], self.stride):
            ax.axhline(y=i, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
        
        ax.set_title(f"Detections: {len(detections)} glomeruli")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Visualization saved to {save_path}")


def test_small_section():
    """Test inference on a small section of WSI."""
    
    # Configuration
    MODEL_PATH = "runs/detect/glomeruli_batch8_test/weights/best.pt"
    WSI_PATH = "Data/raw_slides/TPATH002.svs"  # Change to your WSI
    OUTPUT_DIR = Path("clean_inference_test_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # ROI settings - small section for testing
    ROI_X = 10000  # Adjust based on your WSI
    ROI_Y = 10000
    ROI_WIDTH = 5000
    ROI_HEIGHT = 5000
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        logging.error(f"Model not found at {MODEL_PATH}")
        return
    
    # Check if WSI exists
    if not Path(WSI_PATH).exists():
        # Try to find a WSI file
        wsi_dir = Path("Data/raw_slides")
        wsi_files = list(wsi_dir.glob("*.svs"))
        if wsi_files:
            WSI_PATH = str(wsi_files[0])
            logging.info(f"Using WSI: {WSI_PATH}")
        else:
            logging.error("No WSI files found")
            return
    
    # Initialize inference engine
    engine = CleanInferenceEngine(
        model_path=MODEL_PATH,
        patch_size=1024,
        overlap=128,
        batch_size=4,
        confidence_threshold=0.3,  # Lower for testing
        iou_threshold=0.5
    )
    
    # Process ROI
    start_time = time.time()
    detections = engine.process_roi(WSI_PATH, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*60}")
    print(f"INFERENCE RESULTS")
    print(f"{'='*60}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Total detections: {len(detections)}")
    
    # Count by class
    class_counts = {}
    for det in detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
    
    print("\nDetections by class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Get ROI for visualization
    roi = engine.extract_roi_from_wsi(WSI_PATH, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)
    
    # Visualize results
    vis_path = OUTPUT_DIR / f"roi_detections_{ROI_X}_{ROI_Y}.png"
    engine.visualize_detections(roi, detections, (ROI_X, ROI_Y), str(vis_path))
    
    # Save detection coordinates
    coords_file = OUTPUT_DIR / f"detections_{ROI_X}_{ROI_Y}.txt"
    with open(coords_file, 'w') as f:
        f.write(f"# Detections for ROI ({ROI_X}, {ROI_Y}) size {ROI_WIDTH}x{ROI_HEIGHT}\n")
        f.write(f"# Format: class_name x1 y1 x2 y2 confidence\n")
        for det in detections:
            f.write(f"{det.class_name} {det.x1} {det.y1} {det.x2} {det.y2} {det.confidence:.3f}\n")
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"  Visualization: {vis_path}")
    print(f"  Coordinates: {coords_file}")


if __name__ == "__main__":
    test_small_section()