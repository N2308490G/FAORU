"""
Vision Transformer with FAORU Residuals

Implements ViT variants with frequency-adaptive orthogonal residual updates
integrated into attention and MLP blocks.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import timm

from faoru.residual import FAORUResidual


class ViTWithFAORU(nn.Module):
    """
    Vision Transformer with FAORU residuals.
    
    Integrates FAORU into:
    - Attention blocks (x + FAORU(Attention(LN(x))))
    - MLP blocks (x + FAORU(MLP(LN(x))))
    
    Args:
        base_model: Pre-trained ViT model from timm
        faoru_variant: 'learnable', 'piecewise', or 'smooth'
        faoru_transform: 'fft', 'dct', or 'hadamard'
        faoru_attn: Whether to apply FAORU to attention residuals
        faoru_mlp: Whether to apply FAORU to MLP residuals
        **faoru_kwargs: Additional arguments for FAORUResidual
    """
    
    def __init__(
        self,
        base_model: str = 'vit_base_patch16_224',
        num_classes: int = 1000,
        pretrained: bool = True,
        faoru_variant: str = 'learnable',
        faoru_transform: str = 'fft',
        faoru_attn: bool = True,
        faoru_mlp: bool = True,
        **faoru_kwargs
    ):
        super().__init__()
        
        # Load base ViT from timm
        self.vit = timm.create_model(
            base_model,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Get embedding dimension
        self.embed_dim = self.vit.embed_dim
        
        # Integrate FAORU into transformer blocks
        for i, block in enumerate(self.vit.blocks):
            # Original block structure:
            # x = x + attn(norm1(x))
            # x = x + mlp(norm2(x))
            
            if faoru_attn:
                # Wrap attention residual with FAORU
                block.faoru_attn = FAORUResidual(
                    dim=self.embed_dim,
                    variant=faoru_variant,
                    transform_type=faoru_transform,
                    **faoru_kwargs
                )
            else:
                block.faoru_attn = None
            
            if faoru_mlp:
                # Wrap MLP residual with FAORU
                block.faoru_mlp = FAORUResidual(
                    dim=self.embed_dim,
                    variant=faoru_variant,
                    transform_type=faoru_transform,
                    **faoru_kwargs
                )
            else:
                block.faoru_mlp = None
            
            # Replace forward method
            block.forward = self._create_block_forward(
                block, 
                has_faoru_attn=faoru_attn,
                has_faoru_mlp=faoru_mlp
            )
    
    def _create_block_forward(self, block, has_faoru_attn, has_faoru_mlp):
        """
        Create modified forward function for ViT block with FAORU.
        """
        def forward(x):
            # Attention residual
            attn_out = block.attn(block.norm1(x))
            
            if has_faoru_attn:
                x = block.faoru_attn(x, attn_out)
            else:
                x = x + block.drop_path1(attn_out)
            
            # MLP residual
            mlp_out = block.mlp(block.norm2(x))
            
            if has_faoru_mlp:
                x = block.faoru_mlp(x, mlp_out)
            else:
                x = x + block.drop_path2(mlp_out)
            
            return x
        
        return forward
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, 3, H, W] input images
        
        Returns:
            [B, num_classes] logits
        """
        return self.vit(x)
    
    def get_num_params(self) -> Tuple[int, int]:
        """
        Returns:
            (total_params, faoru_params)
        """
        total = sum(p.numel() for p in self.parameters())
        
        faoru_params = 0
        for block in self.vit.blocks:
            if hasattr(block, 'faoru_attn') and block.faoru_attn is not None:
                faoru_params += sum(p.numel() for p in block.faoru_attn.parameters())
            if hasattr(block, 'faoru_mlp') and block.faoru_mlp is not None:
                faoru_params += sum(p.numel() for p in block.faoru_mlp.parameters())
        
        return total, faoru_params


def vit_tiny_faoru(**kwargs) -> ViTWithFAORU:
    """ViT-Tiny with FAORU (5.7M params)"""
    return ViTWithFAORU('vit_tiny_patch16_224', **kwargs)


def vit_small_faoru(**kwargs) -> ViTWithFAORU:
    """ViT-Small with FAORU (22M params)"""
    return ViTWithFAORU('vit_small_patch16_224', **kwargs)


def vit_base_faoru(**kwargs) -> ViTWithFAORU:
    """ViT-Base with FAORU (86M params)"""
    return ViTWithFAORU('vit_base_patch16_224', **kwargs)


def vit_large_faoru(**kwargs) -> ViTWithFAORU:
    """ViT-Large with FAORU (304M params)"""
    return ViTWithFAORU('vit_large_patch16_224', **kwargs)


def deit3_small_faoru(**kwargs) -> ViTWithFAORU:
    """DeiT-III-Small with FAORU"""
    return ViTWithFAORU('deit3_small_patch16_224', **kwargs)


def deit3_base_faoru(**kwargs) -> ViTWithFAORU:
    """DeiT-III-Base with FAORU"""
    return ViTWithFAORU('deit3_base_patch16_224', **kwargs)


# Example usage
if __name__ == '__main__':
    # Create ViT-Base with FAORU
    model = vit_base_faoru(
        pretrained=True,
        faoru_variant='learnable',
        faoru_transform='fft',
        faoru_attn=True,
        faoru_mlp=True
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
    for i, block in enumerate(model.vit.blocks):
        has_attn = hasattr(block, 'faoru_attn') and block.faoru_attn is not None
        has_mlp = hasattr(block, 'faoru_mlp') and block.faoru_mlp is not None
        print(f"Block {i}: FAORU-Attn={has_attn}, FAORU-MLP={has_mlp}")
