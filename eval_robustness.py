"""
Robustness Evaluation Script

Evaluate models on robustness benchmarks:
- ImageNet-C (corruption robustness)
- ImageNet-R (rendition robustness)
- ImageNet-V2 (distribution shift)
"""

import os
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import pandas as pd

from models import (
    vit_tiny_faoru, vit_small_faoru, vit_base_faoru, vit_large_faoru,
    resnet50_faoru, resnet101_faoru
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate robustness')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--benchmark', type=str, required=True,
                       choices=['imagenet-c', 'imagenet-r', 'imagenet-v2'],
                       help='Robustness benchmark to evaluate')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to benchmark dataset')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of workers')
    parser.add_argument('--output-csv', type=str, default='robustness_results.csv',
                       help='Output CSV file')
    
    return parser.parse_args()


def load_model(checkpoint_path: str) -> nn.Module:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint['config']
    model_name = config['model']['name']
    faoru_config = config['faoru']
    
    model_fn_map = {
        'vit_tiny': vit_tiny_faoru,
        'vit_small': vit_small_faoru,
        'vit_base': vit_base_faoru,
        'vit_large': vit_large_faoru,
        'resnet50': resnet50_faoru,
        'resnet101': resnet101_faoru,
    }
    
    model = model_fn_map[model_name](
        num_classes=config['data']['num_classes'],
        pretrained=False,
        faoru_variant=faoru_config['variant'],
        faoru_transform=faoru_config['transform'],
        faoru_attn=faoru_config.get('attn', True),
        faoru_mlp=faoru_config.get('mlp', True),
        cutoff_ratio=faoru_config.get('cutoff_ratio', 0.3),
        transition_slope=faoru_config.get('transition_slope', 10.0),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def create_transform(input_size: int = 224):
    """Create validation transform"""
    return transforms.Compose([
        transforms.Resize(int(input_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


@torch.no_grad()
def evaluate_corruption(
    model: nn.Module,
    data_dir: str,
    batch_size: int,
    num_workers: int
) -> Dict[str, float]:
    """
    Evaluate on ImageNet-C.
    
    ImageNet-C contains 15 corruption types × 5 severity levels.
    """
    model.eval()
    model.cuda()
    
    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    severities = [1, 2, 3, 4, 5]
    
    results = {}
    
    for corruption in corruption_types:
        corruption_accs = []
        
        for severity in severities:
            corruption_dir = os.path.join(data_dir, corruption, str(severity))
            
            if not os.path.exists(corruption_dir):
                print(f"Warning: {corruption_dir} not found, skipping")
                continue
            
            dataset = datasets.ImageFolder(corruption_dir, transform=create_transform())
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
            correct = 0
            total = 0
            
            for images, targets in tqdm(loader, desc=f'{corruption}-{severity}', leave=False):
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                
                outputs = model(images)
                _, pred = outputs.max(1)
                
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
            
            acc = 100.0 * correct / total
            corruption_accs.append(acc)
            results[f'{corruption}_{severity}'] = acc
            
            print(f"  {corruption}-{severity}: {acc:.2f}%")
        
        # Average over severities for this corruption
        if corruption_accs:
            results[f'{corruption}_avg'] = sum(corruption_accs) / len(corruption_accs)
    
    # Mean Corruption Error (mCE) - average over all corruption×severity combinations
    all_accs = [v for k, v in results.items() if not k.endswith('_avg')]
    if all_accs:
        results['mCA'] = sum(all_accs) / len(all_accs)  # Mean Corruption Accuracy
    
    return results


@torch.no_grad()
def evaluate_rendition(
    model: nn.Module,
    data_dir: str,
    batch_size: int,
    num_workers: int
) -> Dict[str, float]:
    """
    Evaluate on ImageNet-R (30,000 images spanning 200 classes).
    """
    model.eval()
    model.cuda()
    
    dataset = datasets.ImageFolder(data_dir, transform=create_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    correct = 0
    total = 0
    
    for images, targets in tqdm(loader, desc='ImageNet-R'):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        outputs = model(images)
        _, pred = outputs.max(1)
        
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)
    
    acc = 100.0 * correct / total
    
    return {'imagenet_r': acc}


@torch.no_grad()
def evaluate_v2(
    model: nn.Module,
    data_dir: str,
    batch_size: int,
    num_workers: int
) -> Dict[str, float]:
    """
    Evaluate on ImageNet-V2 (three variants: matched-frequency, threshold-0.7, top-images).
    """
    model.eval()
    model.cuda()
    
    results = {}
    
    # Try all three variants
    variants = ['matched-frequency', 'threshold-0.7', 'top-images']
    
    for variant in variants:
        variant_dir = os.path.join(data_dir, variant)
        
        if not os.path.exists(variant_dir):
            print(f"Warning: {variant_dir} not found, skipping")
            continue
        
        dataset = datasets.ImageFolder(variant_dir, transform=create_transform())
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        correct = 0
        total = 0
        
        for images, targets in tqdm(loader, desc=f'ImageNet-V2-{variant}'):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            outputs = model(images)
            _, pred = outputs.max(1)
            
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
        
        acc = 100.0 * correct / total
        results[f'v2_{variant}'] = acc
        
        print(f"  ImageNet-V2-{variant}: {acc:.2f}%")
    
    return results


def main():
    args = parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    
    print(f"\nEvaluating on {args.benchmark}...")
    
    if args.benchmark == 'imagenet-c':
        results = evaluate_corruption(model, args.data_dir, args.batch_size, args.num_workers)
    elif args.benchmark == 'imagenet-r':
        results = evaluate_rendition(model, args.data_dir, args.batch_size, args.num_workers)
    elif args.benchmark == 'imagenet-v2':
        results = evaluate_v2(model, args.data_dir, args.batch_size, args.num_workers)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    # Save results to CSV
    df = pd.DataFrame([results])
    df.to_csv(args.output_csv, index=False)
    
    print(f"\n{'='*50}")
    print(f"Results saved to: {args.output_csv}")
    print(f"{'='*50}")
    
    # Print summary
    for key, value in results.items():
        print(f"  {key}: {value:.2f}%")


if __name__ == '__main__':
    main()
