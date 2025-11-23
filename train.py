"""
Training Script for FAORU Models

Supports distributed training on ImageNet-1K with various architectures.
"""

import os
import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from timm.data import create_transform, Mixup
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from models import (
    vit_tiny_faoru, vit_small_faoru, vit_base_faoru, vit_large_faoru,
    resnet50_faoru, resnet101_faoru
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train FAORU models')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to ImageNet dataset')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    
    # Distributed training
    parser.add_argument('--world-size', type=int, default=1,
                       help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=0,
                       help='Rank of current process')
    parser.add_argument('--dist-url', type=str, default='env://',
                       help='URL for distributed training')
    
    # Override config
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate from config')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Debugging
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (fewer iterations)')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed(args):
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    
    args.distributed = True
    
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create FAORU model from config"""
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
    
    if model_name not in model_fn_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model_fn_map[model_name](
        num_classes=config['data']['num_classes'],
        pretrained=config['model'].get('pretrained', True),
        faoru_variant=faoru_config['variant'],
        faoru_transform=faoru_config['transform'],
        faoru_attn=faoru_config.get('attn', True),
        faoru_mlp=faoru_config.get('mlp', True),
        cutoff_ratio=faoru_config.get('cutoff_ratio', 0.3),
        transition_slope=faoru_config.get('transition_slope', 10.0),
    )
    
    return model


def create_dataloaders(args, config: Dict[str, Any]):
    """Create training and validation dataloaders"""
    data_config = config['data']
    train_config = config['training']
    
    # Training transforms with augmentation
    train_transform = create_transform(
        input_size=data_config['input_size'],
        is_training=True,
        auto_augment=train_config.get('auto_augment', 'rand-m9-mstd0.5-inc1'),
        interpolation='bicubic',
        re_prob=train_config.get('reprob', 0.25),
        re_mode='pixel',
        re_count=1,
    )
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(data_config['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'val'),
        transform=val_transform
    )
    
    # Samplers for distributed training
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Dataloaders
    batch_size = args.batch_size if args.batch_size else train_config['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_config.get('num_workers', 8),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=train_config.get('num_workers', 8),
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mixup_fn: Mixup,
    scaler: GradScaler,
    epoch: int,
    args,
    config: Dict[str, Any]
):
    """Train for one epoch"""
    model.train()
    
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        # Mixup/CutMix
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        
        # Forward with mixed precision
        with autocast(dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        total_loss += loss.item()
        
        if not mixup_fn:  # Only compute accuracy without mixup
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Logging
        if batch_idx % 100 == 0 and (not args.distributed or args.rank == 0):
            elapsed = time.time() - start_time
            throughput = (batch_idx + 1) * images.size(0) * args.world_size / elapsed
            
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Throughput: {throughput:.1f} imgs/s")
        
        if args.debug and batch_idx >= 10:
            break
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    args
):
    """Validate model"""
    model.eval()
    
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for images, targets in val_loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        with autocast(dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        
        # Top-1 and Top-5 accuracy
        _, pred = outputs.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        correct_top1 += correct[:1].sum().item()
        correct_top5 += correct[:5].sum().item()
        total += targets.size(0)
        
        if args.debug and total >= 1000:
            break
    
    avg_loss = total_loss / len(val_loader)
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    
    return avg_loss, top1_acc, top5_acc


def main():
    args = parse_args()
    
    # Setup distributed training
    setup_distributed(args)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.cuda()
    
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, train_sampler = create_dataloaders(args, config)
    
    # Optimizer
    optimizer = create_optimizer_v2(
        model,
        opt=config['training']['optimizer'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched='cosine',
        num_epochs=config['training']['epochs'],
        warmup_epochs=config['training'].get('warmup_epochs', 20),
        warmup_lr=config['training'].get('warmup_lr', 1e-6),
        min_lr=config['training'].get('min_lr', 1e-5),
    )
    
    # Loss function
    mixup_config = config['training'].get('mixup', {})
    
    if mixup_config.get('mixup_alpha', 0) > 0 or mixup_config.get('cutmix_alpha', 0) > 0:
        mixup_fn = Mixup(
            mixup_alpha=mixup_config.get('mixup_alpha', 0.8),
            cutmix_alpha=mixup_config.get('cutmix_alpha', 1.0),
            prob=mixup_config.get('prob', 1.0),
            switch_prob=mixup_config.get('switch_prob', 0.5),
            mode='batch',
            label_smoothing=config['training'].get('label_smoothing', 0.1),
            num_classes=config['data']['num_classes']
        )
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config['training'].get('label_smoothing', 0.1)
        )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_acc = 0.0
    
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            mixup_fn, scaler, epoch, args, config
        )
        
        # Validate
        val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, args)
        
        # Step scheduler
        scheduler.step(epoch)
        
        # Logging
        if not args.distributed or args.rank == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%")
            
            # Save checkpoint
            if val_top1 > best_acc:
                best_acc = val_top1
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'config': config,
                }
                
                torch.save(checkpoint, output_dir / 'best_checkpoint.pth')
                print(f"  Saved best checkpoint (Top-1: {best_acc:.2f}%)")
    
    print(f"\nTraining completed! Best Top-1 Accuracy: {best_acc:.2f}%")
    
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
