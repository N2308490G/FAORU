"""Model package initialization"""

from .vit import (
    ViTWithFAORU,
    vit_tiny_faoru,
    vit_small_faoru,
    vit_base_faoru,
    vit_large_faoru,
    deit3_small_faoru,
    deit3_base_faoru,
)

from .resnet import (
    ResNetWithFAORU,
    resnet50_faoru,
    resnet101_faoru,
    resnet152_faoru,
    resnext50_32x4d_faoru,
    resnext101_64x4d_faoru,
)

__all__ = [
    # ViT models
    'ViTWithFAORU',
    'vit_tiny_faoru',
    'vit_small_faoru',
    'vit_base_faoru',
    'vit_large_faoru',
    'deit3_small_faoru',
    'deit3_base_faoru',
    # ResNet models
    'ResNetWithFAORU',
    'resnet50_faoru',
    'resnet101_faoru',
    'resnet152_faoru',
    'resnext50_32x4d_faoru',
    'resnext101_64x4d_faoru',
]
