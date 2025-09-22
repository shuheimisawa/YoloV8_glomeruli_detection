#!/usr/bin/env python3
"""
Script for preparing YOLO format dataset from QuPath GeoJSON annotations.
Extracts 1024x1024 patches and converts polygons to YOLO bounding boxes.
Handles multiple annotations per patch.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import openslide
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from shapely.geometry import Polygon, Point, box
import warnings
import cv2
warnings.filterwarnings('ignore')


class QuPathYOLOParser:
    """Parser for QuPath GeoJSON annotation files with YOLO format output."""
    
    def __init__(self, geojson_dir: str, svs_dir: str = None):
        self.geojson_dir = Path(geojson_dir)
        self.svs_dir = Path(svs_dir) if svs_dir else None
        self.annotations = {}
        self.class_mapping = {'normal': 0, 'sclerotic': 1}
        
    def parse_geojson_file(self, filepath: Path) -> List[Dict]:
        """Parse a single GeoJSON file and extract glomeruli annotations."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        glomeruli = []
        for feature in data.get('features', []):
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            
            # Check classification
            classification = properties.get('classification', {})
            class_name = classification.get('name', '').lower()
            
            # Filter out partially sclerotic annotations
            if 'partially' in class_name:
                continue
                
            # Only accept normal and sclerotic
            if class_name in ['normal', 'sclerotic'] and geometry.get('type') == 'Polygon':
                coords = geometry.get('coordinates', [])
                if coords:
                    polygon_coords = coords[0] if coords else []
                    
                    if polygon_coords:
                        polygon = Polygon(polygon_coords)
                        centroid = polygon.centroid
                        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
                        
                        glomerulus = {
                            'slide_name': filepath.stem,
                            'class_name': class_name,
                            'class_id': self.class_mapping[class_name],
                            'centroid_x': centroid.x,
                            'centroid_y': centroid.y,
                            'bbox': bounds,
                            'polygon_coords': polygon_coords,
                            'area': polygon.area,
                            'polygon': polygon,
                            'properties': properties
                        }
                        glomeruli.append(glomerulus)
        
        return glomeruli
    
    def parse_all_files(self) -> Dict[str, List[Dict]]:
        """Parse all GeoJSON files in the directory."""
        geojson_files = list(self.geojson_dir.glob('*.geojson'))
        
        for filepath in geojson_files:
            slide_name = filepath.stem
            self.annotations[slide_name] = self.parse_geojson_file(filepath)
            
        return self.annotations
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert annotations to pandas DataFrame."""
        all_glomeruli = []
        for slide_name, glomeruli in self.annotations.items():
            all_glomeruli.extend(glomeruli)
        
        return pd.DataFrame(all_glomeruli)


class YOLOPatchExtractor:
    """Extract 1024x1024 patches and generate YOLO format annotations."""
    
    def __init__(self, svs_dir: str, patch_size: int = 1024, level: int = 0, overlap: float = 0.1):
        self.svs_dir = Path(svs_dir)
        self.patch_size = patch_size
        self.level = level
        self.overlap = overlap
        self.step_size = int(patch_size * (1 - overlap))
        
    def polygon_to_bbox(self, polygon: Polygon) -> Tuple[float, float, float, float]:
        """Convert polygon to bounding box (minx, miny, maxx, maxy)."""
        return polygon.bounds
    
    def bbox_to_yolo(self, bbox: Tuple[float, float, float, float], 
                     patch_x: int, patch_y: int, patch_size: int) -> Tuple[float, float, float, float]:
        """Convert bounding box to YOLO format relative to patch."""
        minx, miny, maxx, maxy = bbox
        
        # Convert to patch-relative coordinates
        rel_minx = minx - patch_x
        rel_miny = miny - patch_y
        rel_maxx = maxx - patch_x
        rel_maxy = maxy - patch_y
        
        # Clamp to patch boundaries
        rel_minx = max(0, min(patch_size, rel_minx))
        rel_miny = max(0, min(patch_size, rel_miny))
        rel_maxx = max(0, min(patch_size, rel_maxx))
        rel_maxy = max(0, min(patch_size, rel_maxy))
        
        # Convert to YOLO format (center_x, center_y, width, height) normalized [0,1]
        width = rel_maxx - rel_minx
        height = rel_maxy - rel_miny
        
        if width <= 0 or height <= 0:
            return None
            
        center_x = (rel_minx + rel_maxx) / 2.0 / patch_size
        center_y = (rel_miny + rel_maxy) / 2.0 / patch_size
        norm_width = width / patch_size
        norm_height = height / patch_size
        
        return (center_x, center_y, norm_width, norm_height)
    
    def get_slide_patches(self, slide_path: Path, annotations: List[Dict]) -> List[Dict]:
        """Generate patch coordinates that contain annotations."""
        try:
            slide = openslide.OpenSlide(str(slide_path))
            slide_width, slide_height = slide.dimensions
            slide.close()
            
            # Create spatial index of annotations
            annotation_boxes = []
            for ann in annotations:
                bbox = ann['bbox']
                annotation_boxes.append(box(bbox[0], bbox[1], bbox[2], bbox[3]))
            
            patches = []
            
            # Generate grid of patches
            for y in range(0, slide_height - self.patch_size + 1, self.step_size):
                for x in range(0, slide_width - self.patch_size + 1, self.step_size):
                    patch_box = box(x, y, x + self.patch_size, y + self.patch_size)
                    
                    # Check if patch intersects with any annotation
                    patch_annotations = []
                    for i, ann_box in enumerate(annotation_boxes):
                        if patch_box.intersects(ann_box):
                            # Check if significant overlap (at least 20% of annotation)
                            intersection = patch_box.intersection(ann_box)
                            if intersection.area / ann_box.area >= 0.2:
                                patch_annotations.append(annotations[i])
                    
                    # Only keep patches that contain annotations
                    if patch_annotations:
                        patches.append({
                            'x': x,
                            'y': y,
                            'width': self.patch_size,
                            'height': self.patch_size,
                            'annotations': patch_annotations
                        })
            
            return patches
            
        except Exception as e:
            print(f"Error processing slide {slide_path}: {e}")
            return []
    
    def extract_patch_with_annotations(self, slide_path: Path, patch_info: Dict, 
                                     output_dir: Path, patch_id: str) -> Optional[Dict]:
        """Extract patch and create YOLO annotation file."""
        try:
            slide = openslide.OpenSlide(str(slide_path))
            
            # Extract patch
            x, y = patch_info['x'], patch_info['y']
            patch = slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
            patch = patch.convert('RGB')
            
            # Save patch image
            patch_filename = f"{patch_id}.jpg"
            patch_path = output_dir / patch_filename
            patch.save(patch_path, quality=95)
            
            # Create YOLO annotation
            yolo_annotations = []
            for ann in patch_info['annotations']:
                bbox = self.polygon_to_bbox(ann['polygon'])
                yolo_bbox = self.bbox_to_yolo(bbox, x, y, self.patch_size)
                
                if yolo_bbox:
                    class_id = ann['class_id']
                    yolo_annotations.append(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
            
            # Save YOLO annotation file
            if yolo_annotations:
                annotation_filename = f"{patch_id}.txt"
                annotation_path = output_dir / annotation_filename
                with open(annotation_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
            
            slide.close()
            
            return {
                'patch_id': patch_id,
                'patch_path': str(patch_path),
                'annotation_path': str(annotation_path) if yolo_annotations else None,
                'slide_name': slide_path.stem,
                'patch_x': x,
                'patch_y': y,
                'num_annotations': len(yolo_annotations),
                'class_distribution': Counter([ann['class_name'] for ann in patch_info['annotations']])
            }
            
        except Exception as e:
            print(f"Error extracting patch {patch_id}: {e}")
            return None
    
    def process_slide(self, slide_name: str, annotations: List[Dict], output_dir: Path) -> List[Dict]:
        """Process a single slide and extract all relevant patches."""
        svs_files = list(self.svs_dir.glob(f"{slide_name}*.svs"))
        
        if not svs_files:
            print(f"Warning: No SVS file found for {slide_name}")
            return []
        
        svs_path = svs_files[0]
        patches_info = self.get_slide_patches(svs_path, annotations)
        
        extracted_patches = []
        for i, patch_info in enumerate(patches_info):
            patch_id = f"{slide_name}_patch_{i:04d}"
            result = self.extract_patch_with_annotations(svs_path, patch_info, output_dir, patch_id)
            if result:
                extracted_patches.append(result)
        
        return extracted_patches


def create_yolo_dataset_splits(df: pd.DataFrame, extractor: YOLOPatchExtractor, 
                              output_base_dir: str, train_ratio: float = 0.7, 
                              val_ratio: float = 0.15, test_ratio: float = 0.15, 
                              random_state: int = 42) -> Dict:
    """Create YOLO dataset with train/val/test splits."""
    
    # Create output directories
    output_base = Path(output_base_dir)
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            (output_base / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Get unique slides
    unique_slides = df['slide_name'].unique()
    
    # Create splits
    train_slides, temp_slides = train_test_split(
        unique_slides, 
        test_size=(val_ratio + test_ratio),
        random_state=random_state
    )
    
    val_slides, test_slides = train_test_split(
        temp_slides,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state
    )
    
    splits = {
        'train': train_slides,
        'val': val_slides,
        'test': test_slides
    }
    
    # Process each split
    all_results = {}
    
    for split_name, slide_names in splits.items():
        print(f"\nProcessing {split_name} split ({len(slide_names)} slides)...")
        split_df = df[df['slide_name'].isin(slide_names)]
        
        output_dir = output_base / split_name / 'images'
        label_dir = output_base / split_name / 'labels'
        
        split_results = []
        
        # Group annotations by slide
        for slide_name in slide_names:
            slide_annotations = split_df[split_df['slide_name'] == slide_name].to_dict('records')
            
            if slide_annotations:
                patches = extractor.process_slide(slide_name, slide_annotations, output_dir)
                
                # Move label files to labels directory
                for patch_info in patches:
                    if patch_info['annotation_path']:
                        label_filename = Path(patch_info['annotation_path']).name
                        new_label_path = label_dir / label_filename
                        os.rename(patch_info['annotation_path'], new_label_path)
                        patch_info['annotation_path'] = str(new_label_path)
                
                split_results.extend(patches)
        
        all_results[split_name] = split_results
        print(f"{split_name}: {len(split_results)} patches extracted")
    
    return all_results


def create_yaml_config(output_dir: str, class_names: List[str]):
    """Create YOLO dataset configuration YAML file."""
    yaml_content = f"""# YOLO dataset configuration
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""
    
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"YOLO config saved to: {yaml_path}")


def analyze_yolo_dataset(results: Dict) -> Dict:
    """Analyze the created YOLO dataset."""
    stats = {}
    
    for split_name, patches in results.items():
        total_patches = len(patches)
        total_annotations = sum(p['num_annotations'] for p in patches)
        
        # Class distribution
        class_counts = Counter()
        for patch in patches:
            for class_name, count in patch['class_distribution'].items():
                class_counts[class_name] += count
        
        # Patches per slide
        slides = Counter(p['slide_name'] for p in patches)
        
        stats[split_name] = {
            'total_patches': total_patches,
            'total_annotations': total_annotations,
            'avg_annotations_per_patch': total_annotations / total_patches if total_patches > 0 else 0,
            'class_distribution': dict(class_counts),
            'slides_count': len(slides),
            'patches_per_slide': dict(slides)
        }
    
    return stats


def print_yolo_dataset_summary(stats: Dict):
    """Print summary of YOLO dataset."""
    print("\n" + "="*60)
    print("YOLO DATASET SUMMARY")
    print("="*60)
    
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Slides: {split_stats['slides_count']}")
        print(f"  Patches: {split_stats['total_patches']}")
        print(f"  Total annotations: {split_stats['total_annotations']}")
        print(f"  Avg annotations per patch: {split_stats['avg_annotations_per_patch']:.1f}")
        
        print(f"  Class distribution:")
        for class_name, count in split_stats['class_distribution'].items():
            percentage = (count / split_stats['total_annotations']) * 100 if split_stats['total_annotations'] > 0 else 0
            print(f"    {class_name}: {count} ({percentage:.1f}%)")


def main():
    """Main function to run the YOLO dataset preparation pipeline."""
    
    # Configuration
    GEOJSON_DIR = "Data/annotations"
    SVS_DIR = "Data/raw_slides"
    OUTPUT_DIR = "yolo_dataset"
    PATCH_SIZE = 1024
    OVERLAP = 0.1  # 10% overlap between patches
    
    print("Starting YOLO dataset preparation pipeline...")
    
    # Step 1: Parse GeoJSON files (excluding partially sclerotic)
    print("\n1. Parsing GeoJSON files...")
    parser = QuPathYOLOParser(GEOJSON_DIR, SVS_DIR)
    annotations = parser.parse_all_files()
    df = parser.get_dataframe()
    
    if df.empty:
        print("No valid annotations found!")
        return
    
    # Print updated statistics
    print(f"\nFiltered dataset statistics:")
    print(f"Total slides: {df['slide_name'].nunique()}")
    print(f"Total annotations: {len(df)}")
    print("Class distribution:")
    for class_name, count in df['class_name'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Step 2: Extract patches and create YOLO dataset
    print(f"\n2. Extracting {PATCH_SIZE}x{PATCH_SIZE} patches with YOLO annotations...")
    extractor = YOLOPatchExtractor(SVS_DIR, patch_size=PATCH_SIZE, overlap=OVERLAP)
    
    results = create_yolo_dataset_splits(df, extractor, OUTPUT_DIR)
    
    # Step 3: Create YOLO config file
    print("\n3. Creating YOLO configuration...")
    class_names = sorted(df['class_name'].unique().tolist())
    create_yaml_config(OUTPUT_DIR, class_names)
    
    # Step 4: Analyze and summarize
    print("\n4. Analyzing dataset...")
    stats = analyze_yolo_dataset(results)
    print_yolo_dataset_summary(stats)
    
    # Save results
    import pickle
    results_path = Path(OUTPUT_DIR) / 'extraction_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({'results': results, 'stats': stats}, f)
    
    print(f"\nDataset preparation complete!")
    print(f"YOLO dataset saved to: {OUTPUT_DIR}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()