#!/usr/bin/env python3
"""
Complete YOLO dataset preparation with better error handling and progress tracking.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
import openslide
from PIL import Image
from shapely.geometry import Polygon, box
import warnings
import traceback
warnings.filterwarnings('ignore')

def parse_all_annotations():
    """Parse all GeoJSON files and create filtered dataframe."""
    geojson_dir = Path('Data/annotations')
    all_glomeruli = []
    
    print("Parsing annotations...")
    for filepath in geojson_dir.glob('*.geojson'):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            slide_annotations = []
            for feature in data.get('features', []):
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})
                
                classification = properties.get('classification', {})
                class_name = classification.get('name', '').lower()
                
                # Filter out partially sclerotic
                if 'partially' in class_name:
                    continue
                    
                if class_name in ['normal', 'sclerotic'] and geometry.get('type') == 'Polygon':
                    coords = geometry.get('coordinates', [])
                    if coords:
                        polygon_coords = coords[0]
                        polygon = Polygon(polygon_coords)
                        
                        slide_annotations.append({
                            'slide_name': filepath.stem,
                            'class_name': class_name,
                            'class_id': 0 if class_name == 'normal' else 1,
                            'polygon': polygon,
                            'bbox': polygon.bounds,
                            'centroid_x': polygon.centroid.x,
                            'centroid_y': polygon.centroid.y,
                            'area': polygon.area
                        })
            
            all_glomeruli.extend(slide_annotations)
            print(f"  {filepath.stem}: {len(slide_annotations)} annotations")
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    return pd.DataFrame(all_glomeruli)

def extract_patches_for_split(df_split, split_name, output_base, patch_size=1024):
    """Extract patches for a specific data split."""
    svs_dir = Path('Data/raw_slides')
    
    split_dir = output_base / split_name
    images_dir = split_dir / 'images'
    labels_dir = split_dir / 'labels'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    patches_created = 0
    total_annotations = 0
    slides_processed = 0
    
    # Group by slide
    for slide_name in df_split['slide_name'].unique():
        slide_df = df_split[df_split['slide_name'] == slide_name]
        annotations = slide_df.to_dict('records')
        
        # Find SVS file
        svs_files = list(svs_dir.glob(f'{slide_name}*.svs'))
        if not svs_files:
            print(f"  Warning: No SVS file for {slide_name}")
            continue
        
        svs_path = svs_files[0]
        
        try:
            slide = openslide.OpenSlide(str(svs_path))
            slide_width, slide_height = slide.dimensions
            
            # Extract patches around each annotation
            slide_patches = 0
            for i, ann in enumerate(annotations):
                # Center patch on annotation
                x = max(0, min(slide_width - patch_size, int(ann['centroid_x'] - patch_size//2)))
                y = max(0, min(slide_height - patch_size, int(ann['centroid_y'] - patch_size//2)))
                
                # Extract patch
                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                patch = patch.convert('RGB')
                
                # Save patch
                patch_id = f"{slide_name}_ann_{i:04d}"
                patch_path = images_dir / f"{patch_id}.jpg"
                patch.save(patch_path, quality=95)
                
                # Create YOLO annotation for this patch
                patch_box = box(x, y, x + patch_size, y + patch_size)
                yolo_annotations = []
                
                # Check all annotations for overlap with this patch
                for check_ann in annotations:
                    ann_box = box(*check_ann['bbox'])
                    if patch_box.intersects(ann_box):
                        # Calculate overlap
                        intersection = patch_box.intersection(ann_box)
                        if intersection.area / ann_box.area >= 0.1:  # 10% overlap minimum
                            
                            # Convert to patch coordinates
                            minx, miny, maxx, maxy = check_ann['bbox']
                            
                            # Relative to patch
                            rel_minx = max(0, minx - x)
                            rel_miny = max(0, miny - y)
                            rel_maxx = min(patch_size, maxx - x)
                            rel_maxy = min(patch_size, maxy - y)
                            
                            if rel_maxx > rel_minx and rel_maxy > rel_miny:
                                # YOLO format
                                width = rel_maxx - rel_minx
                                height = rel_maxy - rel_miny
                                center_x = (rel_minx + rel_maxx) / 2.0 / patch_size
                                center_y = (rel_miny + rel_maxy) / 2.0 / patch_size
                                norm_width = width / patch_size
                                norm_height = height / patch_size
                                
                                class_id = check_ann['class_id']
                                yolo_annotations.append(
                                    f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                                )
                
                # Save YOLO annotation file
                if yolo_annotations:
                    label_path = labels_dir / f"{patch_id}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\\n'.join(yolo_annotations))
                    total_annotations += len(yolo_annotations)
                
                slide_patches += 1
                patches_created += 1
            
            slide.close()
            slides_processed += 1
            print(f"  {split_name}: {slide_name} -> {slide_patches} patches")
            
        except Exception as e:
            print(f"  Error processing {slide_name}: {e}")
    
    print(f"{split_name} complete: {slides_processed} slides, {patches_created} patches, {total_annotations} annotations")
    return {
        'slides': slides_processed,
        'patches': patches_created,
        'annotations': total_annotations
    }

def main():
    """Main function."""
    print("Starting complete YOLO dataset preparation...")
    
    # Parse annotations
    df = parse_all_annotations()
    print(f"\\nTotal annotations: {len(df)}")
    print(f"Slides: {df['slide_name'].nunique()}")
    print("Class distribution:")
    for class_name, count in df['class_name'].value_counts().items():
        print(f"  {class_name}: {count}")
    
    # Create splits
    print("\\nCreating data splits...")
    unique_slides = df['slide_name'].unique()
    
    train_slides, temp_slides = train_test_split(unique_slides, test_size=0.3, random_state=42)
    val_slides, test_slides = train_test_split(temp_slides, test_size=0.5, random_state=42)
    
    train_df = df[df['slide_name'].isin(train_slides)]
    val_df = df[df['slide_name'].isin(val_slides)]
    test_df = df[df['slide_name'].isin(test_slides)]
    
    print(f"Train: {len(train_slides)} slides, {len(train_df)} annotations")
    print(f"Val: {len(val_slides)} slides, {len(val_df)} annotations")
    print(f"Test: {len(test_slides)} slides, {len(test_df)} annotations")
    
    # Create output directory
    output_base = Path('yolo_dataset_complete')
    output_base.mkdir(exist_ok=True)
    
    # Extract patches for each split
    print("\\nExtracting patches...")
    results = {}
    
    print("\\n1. Processing training set...")
    results['train'] = extract_patches_for_split(train_df, 'train', output_base)
    
    print("\\n2. Processing validation set...")
    results['val'] = extract_patches_for_split(val_df, 'val', output_base)
    
    print("\\n3. Processing test set...")
    results['test'] = extract_patches_for_split(test_df, 'test', output_base)
    
    # Create YAML config
    yaml_content = f"""# Complete YOLO dataset configuration
path: {os.path.abspath(output_base)}
train: train/images
val: val/images
test: test/images

# Classes
nc: 2
names: ['normal', 'sclerotic']
"""
    
    yaml_path = output_base / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Print final summary
    print("\\n" + "="*60)
    print("COMPLETE YOLO DATASET SUMMARY")
    print("="*60)
    
    total_slides = 0
    total_patches = 0
    total_annotations = 0
    
    for split_name, stats in results.items():
        print(f"\\n{split_name.upper()}:")
        print(f"  Slides: {stats['slides']}")
        print(f"  Patches: {stats['patches']}")
        print(f"  Annotations: {stats['annotations']}")
        
        total_slides += stats['slides']
        total_patches += stats['patches']
        total_annotations += stats['annotations']
    
    print(f"\\nTOTAL:")
    print(f"  Slides: {total_slides}")
    print(f"  Patches: {total_patches}")
    print(f"  Annotations: {total_annotations}")
    
    print(f"\\nDataset saved to: {output_base}")
    print(f"Config file: {yaml_path}")

if __name__ == "__main__":
    main()