"""
Evaluation Script for FAORU Models

Evaluate trained models on ImageNet validation set.
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from models import (
    vit_tiny_faoru, vit_small_faoru, vit_base_faoru, vit_large_faoru,
    resnet50_faoru, resnet101_faoru
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FAORU models')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to ImageNet validation set')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--input-size', type=int, default=224,
                       help='Input image size')
    
    return parser.parse_args()


def load_model(checkpoint_path: str) -> nn.Module:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint['config']
    model_name = config['model']['name']
    faoru_config = config['faoru']
    
    # Create model
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
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config


def create_dataloader(data_dir: str, batch_size: int, num_workers: int, input_size: int):
    """Create validation dataloader"""
    transform = transforms.Compose([
        transforms.Resize(int(input_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader):
    """Evaluate model on validation set"""
    model.eval()
    model.cuda()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Evaluating')
    
    for images, targets in pbar:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        
        # Top-1 and Top-5 accuracy
        _, pred = outputs.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        correct_top1 += correct[:1].sum().item()
        correct_top5 += correct[:5].sum().item()
        total += targets.size(0)
        
        # Update progress bar
        top1_acc = 100.0 * correct_top1 / total
        top5_acc = 100.0 * correct_top5 / total
        pbar.set_postfix({
            'Top-1': f'{top1_acc:.2f}%',
            'Top-5': f'{top5_acc:.2f}%'
        })
    
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    
    return top1_acc, top5_acc


def main():
    args = parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint)
    
    print(f"Model: {config['model']['name']}")
    print(f"FAORU variant: {config['faoru']['variant']}")
    print(f"FAORU transform: {config['faoru']['transform']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print(f"\nLoading validation data from {args.data_dir}...")
    dataloader = create_dataloader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.input_size
    )
    
    print(f"\nEvaluating on {len(dataloader.dataset)} images...")
    top1_acc, top5_acc = evaluate(model, dataloader)
    
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
