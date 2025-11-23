"""
FAORU Residual Connection Module

Implements the three-stage frequency-adaptive orthogonal residual update:
1. Frequency Transform (FFT/DCT/Hadamard)
2. Per-frequency orthogonalization with adaptive strengths λ_k
3. Inverse transform back to spatial domain

Reference:
    FAORU: Frequency-Adaptive Orthogonal Residual Updates for Modern Vision Networks
    TPAMI 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

from .variants import PiecewiseConstantLambda, SmoothTransitionLambda, LearnableLambda
from .transforms import RealFFT, DCT, HadamardTransform


class FAORUResidual(nn.Module):
    """
    Frequency-Adaptive Orthogonal Residual Update module.
    
    Performs residual updates in a frequency-transformed space with per-component
    orthogonalization controlled by learnable or hand-crafted strengths λ_k.
    
    Args:
        dim (int): Feature dimension D
        variant (str): Lambda parameterization: 'piecewise', 'smooth', or 'learnable'
        transform (str): Transform type: 'fft', 'dct', or 'hadamard'
        epsilon (float): Numerical stability constant for alpha computation
        cutoff_ratio (float): For piecewise/smooth variants, frequency cutoff (0-1)
        lambda_low (float): For piecewise variant, λ for low frequencies
        lambda_high (float): For piecewise variant, λ for high frequencies
        
    Example:
        >>> residual = FAORUResidual(dim=768, variant='learnable')
        >>> x = torch.randn(32, 196, 768)  # [B, S, D]
        >>> f_x = torch.randn(32, 196, 768)  # Residual branch output
        >>> output = residual(x, f_x)  # x + FAORU(f_x)
    """
    
    def __init__(
        self,
        dim: int,
        variant: Literal['piecewise', 'smooth', 'learnable'] = 'learnable',
        transform: Literal['fft', 'dct', 'hadamard'] = 'fft',
        epsilon: float = 1e-6,
        cutoff_ratio: float = 0.3,
        lambda_low: float = 0.3,
        lambda_high: float = 1.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.epsilon = epsilon
        self.variant = variant
        self.transform_type = transform
        
        # Initialize frequency transform
        if transform == 'fft':
            self.transform = RealFFT(dim)
        elif transform == 'dct':
            self.transform = DCT(dim)
        elif transform == 'hadamard':
            self.transform = HadamardTransform(dim)
        else:
            raise ValueError(f"Unknown transform: {transform}")
        
        # Initialize lambda parameterization
        freq_dim = self.transform.get_freq_dim()
        
        if variant == 'piecewise':
            self.lambda_module = PiecewiseConstantLambda(
                freq_dim, cutoff_ratio, lambda_low, lambda_high
            )
        elif variant == 'smooth':
            self.lambda_module = SmoothTransitionLambda(
                freq_dim, cutoff_ratio, lambda_low, lambda_high
            )
        elif variant == 'learnable':
            self.lambda_module = LearnableLambda(freq_dim)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    
    def compute_alpha_k(
        self, 
        X_omega: torch.Tensor, 
        F_omega: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-frequency projection coefficients α_k.
        
        For each frequency k:
            α_k = Re(<F[k], X[k]>) / (||X[k]||² + ε)
        
        where <·,·> aggregates over batch and spatial dimensions.
        
        Args:
            X_omega: Input in frequency domain, shape [B, S, freq_dim, 2] (complex as real pairs)
            F_omega: Residual in frequency domain, shape [B, S, freq_dim, 2]
            
        Returns:
            alpha_k: Projection coefficients, shape [freq_dim]
        """
        # Convert to complex
        X_complex = torch.view_as_complex(X_omega)  # [B, S, freq_dim]
        F_complex = torch.view_as_complex(F_omega)  # [B, S, freq_dim]
        
        # Compute numerator: Re(<F, X>) = Re(sum(F * conj(X)))
        numerator = torch.real(torch.sum(F_complex * torch.conj(X_complex), dim=[0, 1]))  # [freq_dim]
        
        # Compute denominator: ||X||² = sum(|X|²)
        denominator = torch.sum(torch.abs(X_complex) ** 2, dim=[0, 1]) + self.epsilon  # [freq_dim]
        
        # α_k = numerator / denominator
        alpha_k = numerator / denominator  # [freq_dim]
        
        return alpha_k
    
    def forward(
        self, 
        x: torch.Tensor, 
        f_x: torch.Tensor,
        return_freq_analysis: bool = False
    ) -> torch.Tensor:
        """
        Forward pass: x + FAORU(f_x)
        
        Stages:
            1. FFT: Transform x and f_x to frequency domain
            2. Orthogonalization: F_⊥[k] = F[k] - λ_k * α_k * X[k] for each k
            3. IFFT: Transform F_⊥ back to spatial domain
            4. Residual: return x + f_⊥
        
        Args:
            x: Input tensor, shape [B, S, D]
            f_x: Residual branch output, shape [B, S, D]
            return_freq_analysis: If True, return (output, freq_stats) for analysis
            
        Returns:
            output: x + FAORU(f_x), shape [B, S, D]
            freq_stats (optional): Dict with frequency-domain statistics
        """
        B, S, D = x.shape
        assert D == self.dim, f"Feature dim mismatch: {D} vs {self.dim}"
        
        # Stage 1: Frequency Transform
        X_omega = self.transform.forward(x)  # [B, S, freq_dim, 2]
        F_omega = self.transform.forward(f_x)  # [B, S, freq_dim, 2]
        
        # Stage 2: Per-frequency orthogonalization
        # Compute α_k for each frequency
        alpha_k = self.compute_alpha_k(X_omega, F_omega)  # [freq_dim]
        
        # Get λ_k for each frequency
        lambda_k = self.lambda_module()  # [freq_dim]
        
        # Convert to complex for computation
        X_complex = torch.view_as_complex(X_omega)  # [B, S, freq_dim]
        F_complex = torch.view_as_complex(F_omega)  # [B, S, freq_dim]
        
        # F_⊥[k] = F[k] - λ_k * α_k * X[k]
        # Broadcast: alpha_k [freq_dim] * lambda_k [freq_dim] -> [freq_dim]
        # Then broadcast to [B, S, freq_dim]
        correction = (lambda_k * alpha_k).unsqueeze(0).unsqueeze(0)  # [1, 1, freq_dim]
        F_perp_complex = F_complex - correction * X_complex  # [B, S, freq_dim]
        
        # Convert back to real pairs
        F_perp_omega = torch.view_as_real(F_perp_complex)  # [B, S, freq_dim, 2]
        
        # Stage 3: Inverse Transform
        f_perp = self.transform.inverse(F_perp_omega)  # [B, S, D]
        
        # Stage 4: Residual connection
        output = x + f_perp
        
        if return_freq_analysis:
            # Compute statistics for analysis
            with torch.no_grad():
                energy_before = torch.sum(torch.abs(F_complex) ** 2, dim=[0, 1])
                energy_after = torch.sum(torch.abs(F_perp_complex) ** 2, dim=[0, 1])
                correlation_before = torch.real(
                    torch.sum(F_complex * torch.conj(X_complex), dim=[0, 1])
                )
                correlation_after = torch.real(
                    torch.sum(F_perp_complex * torch.conj(X_complex), dim=[0, 1])
                )
                
                freq_stats = {
                    'lambda_k': lambda_k.detach().cpu(),
                    'alpha_k': alpha_k.detach().cpu(),
                    'energy_before': energy_before.detach().cpu(),
                    'energy_after': energy_after.detach().cpu(),
                    'correlation_before': correlation_before.detach().cpu(),
                    'correlation_after': correlation_after.detach().cpu(),
                }
            
            return output, freq_stats
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, variant={self.variant}, '
            f'transform={self.transform_type}, epsilon={self.epsilon}'
        )


class FAORUTransformerBlock(nn.Module):
    """
    Example Transformer block using FAORU residuals.
    
    Standard: x = x + attn(norm(x))
    FAORU:    x = FAORU_residual(x, attn(norm(x)))
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        faoru_variant: str = 'learnable',
        faoru_transform: str = 'fft',
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        
        # FAORU residuals
        self.residual1 = FAORUResidual(dim, faoru_variant, faoru_transform)
        self.residual2 = FAORUResidual(dim, faoru_variant, faoru_transform)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with FAORU residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = self.residual1(x, attn_out)
        
        # MLP with FAORU residual
        mlp_out = self.mlp(self.norm2(x))
        x = self.residual2(x, mlp_out)
        
        return x
