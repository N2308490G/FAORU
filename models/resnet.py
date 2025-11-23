"""
ResNet with FAORU Residuals

Implements ResNet variants with frequency-adaptive orthogonal residual updates
integrated into bottleneck blocks.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Type
import timm

from faoru.residual import FAORUResidual


class BottleneckWithFAORU(nn.Module):
    """
    ResNet Bottleneck block with FAORU.
    
    Original: x + conv3x3(conv3x3(conv1x1(x)))
    With FAORU: x + FAORU(conv3x3(conv3x3(conv1x1(x))))
    """
    
    expansion = 4
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
        faoru_variant: str = 'learnable',
        faoru_transform: str = 'fft',
        **faoru_kwargs
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(planes * (base_width / 64.0)) * groups
        
        # Convolution layers
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride,
            padding=dilation, groups=groups, bias=False, dilation=dilation
        )
        self.bn2 = norm_layer(width)
        
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # FAORU residual
        # Note: ResNet operates on spatial features [B, C, H, W]
        # We flatten to [B, H*W, C] for FAORU, then reshape back
        self.faoru = FAORUResidual(
            dim=planes * self.expansion,
            variant=faoru_variant,
            transform_type=faoru_transform,
            **faoru_kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Convolutions
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Downsample skip connection if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Reshape for FAORU: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = out.shape
        out_flat = out.permute(0, 2, 3, 1).reshape(B, H*W, C)
        identity_flat = identity.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Apply FAORU residual
        out_flat = self.faoru(identity_flat, out_flat - identity_flat)
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        out = out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        out = self.relu(out)
        
        return out


class ResNetWithFAORU(nn.Module):
    """
    ResNet with FAORU residuals in bottleneck blocks.
    
    Args:
        base_model: Pre-trained ResNet model name from timm
        faoru_variant: 'learnable', 'piecewise', or 'smooth'
        faoru_transform: 'fft', 'dct', or 'hadamard'
        **faoru_kwargs: Additional arguments for FAORUResidual
    """
    
    def __init__(
        self,
        base_model: str = 'resnet50',
        num_classes: int = 1000,
        pretrained: bool = True,
        faoru_variant: str = 'learnable',
        faoru_transform: str = 'fft',
        **faoru_kwargs
    ):
        super().__init__()
        
        # Load base ResNet from timm
        self.resnet = timm.create_model(
            base_model,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Integrate FAORU into bottleneck blocks
        for name, module in self.resnet.named_modules():
            if isinstance(module, timm.models.resnet.Bottleneck):
                # Get block dimensions
                channels = module.conv3.out_channels
                
                # Add FAORU residual
                module.faoru = FAORUResidual(
                    dim=channels,
                    variant=faoru_variant,
                    transform_type=faoru_transform,
                    **faoru_kwargs
                )
                
                # Replace forward method
                module.forward = self._create_bottleneck_forward(module)
    
    def _create_bottleneck_forward(self, block):
        """
        Create modified forward function for bottleneck with FAORU.
        """
        original_forward = block.forward
        
        def forward(x):
            identity = x
            
            # Convolution path
            out = block.conv1(x)
            out = block.bn1(out)
            out = block.act1(out)
            
            out = block.conv2(out)
            out = block.bn2(out)
            out = block.act2(out)
            
            out = block.conv3(out)
            out = block.bn3(out)
            
            # Downsample identity if needed
            if block.downsample is not None:
                identity = block.downsample(x)
            
            # Apply FAORU residual
            # Reshape: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = out.shape
            out_flat = out.permute(0, 2, 3, 1).reshape(B, H*W, C)
            identity_flat = identity.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            # FAORU: identity + FAORU(out - identity)
            out_flat = block.faoru(identity_flat, out_flat - identity_flat)
            
            # Reshape back: [B, H*W, C] -> [B, C, H, W]
            out = out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            out = block.act3(out)
            
            return out
        
        return forward
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, 3, H, W] input images
        
        Returns:
            [B, num_classes] logits
        """
        return self.resnet(x)
    
    def get_num_params(self) -> Tuple[int, int]:
        """
        Returns:
            (total_params, faoru_params)
        """
        total = sum(p.numel() for p in self.parameters())
        
        faoru_params = 0
        for module in self.resnet.modules():
            if hasattr(module, 'faoru'):
                faoru_params += sum(p.numel() for p in module.faoru.parameters())
        
        return total, faoru_params


def resnet50_faoru(**kwargs) -> ResNetWithFAORU:
    """ResNet-50 with FAORU (25.6M params)"""
    return ResNetWithFAORU('resnet50', **kwargs)


def resnet101_faoru(**kwargs) -> ResNetWithFAORU:
    """ResNet-101 with FAORU (44.5M params)"""
    return ResNetWithFAORU('resnet101', **kwargs)


def resnet152_faoru(**kwargs) -> ResNetWithFAORU:
    """ResNet-152 with FAORU (60.2M params)"""
    return ResNetWithFAORU('resnet152', **kwargs)


def resnext50_32x4d_faoru(**kwargs) -> ResNetWithFAORU:
    """ResNeXt-50-32x4d with FAORU (25.0M params)"""
    return ResNetWithFAORU('resnext50_32x4d', **kwargs)


def resnext101_64x4d_faoru(**kwargs) -> ResNetWithFAORU:
    """ResNeXt-101-64x4d with FAORU (83.5M params)"""
    return ResNetWithFAORU('resnext101_64x4d', **kwargs)


# Example usage
if __name__ == '__main__':
    # Create ResNet-50 with FAORU
    model = resnet50_faoru(
        pretrained=True,
        faoru_variant='learnable',
        faoru_transform='fft'
    )
    
    # Count parameters
    total_params, faoru_params = model.get_num_params()
    print(f"Total params: {total_params:,}")
    print(f"FAORU params: {faoru_params:,} ({faoru_params/total_params*100:.2f}%)")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Verify FAORU integration
    faoru_count = 0
    for name, module in model.resnet.named_modules():
        if hasattr(module, 'faoru'):
            faoru_count += 1
    print(f"Number of FAORU modules: {faoru_count}")
