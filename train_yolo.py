#!/usr/bin/env python3
"""
YOLOv8 Training Script for Glomeruli Detection
Trains YOLOv8m on the prepared glomeruli dataset.
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse

def check_dataset(data_yaml_path):
    """Verify dataset structure and configuration."""
    print(f"Checking dataset: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    
    # Check directories exist
    for split in ['train', 'val', 'test']:
        img_dir = base_path / config[split]
        label_dir = base_path / split / 'labels'
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {img_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {label_dir}")
        
        # Count files
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        label_files = list(label_dir.glob('*.txt'))
        
        print(f"  {split}: {len(img_files)} images, {len(label_files)} labels")
    
    print(f"  Classes: {config['nc']} - {config['names']}")
    return config

def setup_training_environment():
    """Set up training environment and check requirements."""
    print("Setting up training environment...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {device} ({memory:.1f}GB)")
    else:
        print("  Using CPU (training will be slow)")
    
    # Check available memory
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    return torch.cuda.is_available()

def create_training_config(epochs=100, imgsz=1024, batch_size=8):
    """Create training configuration."""
    config = {
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'patience': 20,  # Early stopping patience
        'save_period': 10,  # Save checkpoint every N epochs
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'project': 'runs/detect',
        'name': f'glomeruli_yolov8m_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'SGD',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    return config

def train_model(data_yaml_path, model_path='yolov8m.pt', **kwargs):
    """Train YOLOv8 model."""
    print(f"Loading model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Print model info
    print(f"Model: {model.model}")
    
    # Start training
    print("Starting training...")
    print(f"Dataset: {data_yaml_path}")
    print(f"Configuration: {kwargs}")
    
    results = model.train(
        data=data_yaml_path,
        **kwargs
    )
    
    return model, results

def evaluate_model(model, data_yaml_path):
    """Evaluate trained model."""
    print("Evaluating model...")
    
    # Validate on test set
    test_results = model.val(data=data_yaml_path, split='test')
    print("Test Results:")
    print(f"  mAP50: {test_results.box.map50:.4f}")
    print(f"  mAP50-95: {test_results.box.map:.4f}")
    
    # Get class-wise metrics
    if hasattr(test_results.box, 'maps'):
        print("  Class-wise mAP50:")
        class_names = ['normal', 'sclerotic']
        for i, (name, ap) in enumerate(zip(class_names, test_results.box.maps)):
            print(f"    {name}: {ap:.4f}")
    
    return test_results

def plot_training_results(results_dir):
    """Plot training results."""
    results_path = Path(results_dir) / 'results.csv'
    
    if results_path.exists():
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()  # Remove any whitespace
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        if 'train/box_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        if 'train/cls_loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
            axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # mAP plots
        if 'metrics/mAP50(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
            axes[1, 0].set_title('Mean Average Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Precision/Recall
        if 'metrics/precision(B)' in df.columns:
            axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
            axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = Path(results_dir) / 'training_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {plot_path}")
        plt.show()

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for glomeruli detection')
    parser.add_argument('--data', default='yolo_dataset/dataset.yaml', help='Dataset YAML path')
    parser.add_argument('--model', default='yolov8m.pt', help='Model path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=1024, help='Image size')
    parser.add_argument('--device', default='0', help='Device to use')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--name', default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    print("="*60)
    print("YOLOv8 GLOMERULI DETECTION TRAINING")
    print("="*60)
    
    # Check dataset
    try:
        dataset_config = check_dataset(args.data)
    except Exception as e:
        print(f"Dataset error: {e}")
        return
    
    # Setup environment
    cuda_available = setup_training_environment()
    
    # Adjust batch size based on available memory
    if not cuda_available:
        args.batch = min(args.batch, 4)
        print(f"Reduced batch size to {args.batch} for CPU training")
    
    # Create training configuration
    train_config = create_training_config(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch
    )
    
    # Override with command line arguments
    if args.device != '0':
        train_config['device'] = args.device
    if args.workers != 4:
        train_config['workers'] = args.workers
    if args.name:
        train_config['name'] = args.name
    
    # Train model
    try:
        model, results = train_model(args.data, args.model, **train_config)
        
        print("\nTraining completed successfully!")
        
        # Get results directory
        results_dir = model.trainer.save_dir
        print(f"Results saved to: {results_dir}")
        
        # Evaluate model
        test_results = evaluate_model(model, args.data)
        
        # Plot results
        plot_training_results(results_dir)
        
        # Save final model info
        print(f"\nFinal model saved to: {results_dir}/weights/best.pt")
        print(f"Last model saved to: {results_dir}/weights/last.pt")
        
        return model, results_dir
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()