"""
Lambda (λ_k) Parameterizations

Three variants for frequency-adaptive strengths:
1. Piecewise Constant (FAORU-PC): Step function at cutoff
2. Smooth Transition (FAORU-ST): Sigmoid transition
3. Learnable (FAORU-L): End-to-end learned parameters
"""

import torch
import torch.nn as nn
from typing import Tuple


class PiecewiseConstantLambda(nn.Module):
    """
    Piecewise constant λ_k (FAORU-PC).
    
    λ_k = {
        lambda_low   if k/D < cutoff_ratio
        lambda_high  if k/D >= cutoff_ratio
    }
    
    Args:
        freq_dim: Number of frequency components
        cutoff_ratio: Cutoff at normalized frequency (default: 0.3)
        lambda_low: Value for low frequencies (default: 0.3)
        lambda_high: Value for high frequencies (default: 1.0)
    """
    
    def __init__(
        self,
        freq_dim: int,
        cutoff_ratio: float = 0.3,
        lambda_low: float = 0.3,
        lambda_high: float = 1.0
    ):
        super().__init__()
        
        self.freq_dim = freq_dim
        self.cutoff_idx = int(cutoff_ratio * freq_dim)
        
        # Fixed values (not learnable)
        lambda_values = torch.ones(freq_dim) * lambda_high
        lambda_values[:self.cutoff_idx] = lambda_low
        
        self.register_buffer('lambda_k', lambda_values)
    
    def forward(self) -> torch.Tensor:
        return self.lambda_k


class SmoothTransitionLambda(nn.Module):
    """
    Smooth transition λ_k with sigmoid (FAORU-ST).
    
    λ_k = lambda_low + (lambda_high - lambda_low) * σ(slope * (k/D - cutoff_ratio))
    
    Args:
        freq_dim: Number of frequency components
        cutoff_ratio: Transition center (default: 0.3)
        lambda_low: Minimum value (default: 0.3)
        lambda_high: Maximum value (default: 1.0)
        slope: Sigmoid slope (default: 20.0)
    """
    
    def __init__(
        self,
        freq_dim: int,
        cutoff_ratio: float = 0.3,
        lambda_low: float = 0.3,
        lambda_high: float = 1.0,
        slope: float = 20.0
    ):
        super().__init__()
        
        self.freq_dim = freq_dim
        
        # Compute normalized frequencies
        normalized_freq = torch.arange(freq_dim, dtype=torch.float32) / freq_dim
        
        # Smooth transition: sigma(slope * (f - cutoff))
        sigmoid_input = slope * (normalized_freq - cutoff_ratio)
        transition = torch.sigmoid(sigmoid_input)
        
        # Scale to [lambda_low, lambda_high]
        lambda_values = lambda_low + (lambda_high - lambda_low) * transition
        
        self.register_buffer('lambda_k', lambda_values)
    
    def forward(self) -> torch.Tensor:
        return self.lambda_k


class LearnableLambda(nn.Module):
    """
    Learnable λ_k parameters (FAORU-L).
    
    λ_k = σ(w_k) where w_k are learnable parameters.
    Sigmoid ensures λ_k ∈ [0, 1].
    
    Args:
        freq_dim: Number of frequency components
        init_strategy: Initialization ('uniform', 'smooth', 'random')
    """
    
    def __init__(
        self,
        freq_dim: int,
        init_strategy: str = 'smooth'
    ):
        super().__init__()
        
        self.freq_dim = freq_dim
        
        # Initialize learnable parameters
        if init_strategy == 'uniform':
            # All frequencies start at λ = 0.5 (w = 0)
            init_weights = torch.zeros(freq_dim)
        
        elif init_strategy == 'smooth':
            # Initialize with smooth transition from 0.3 to 1.0
            # This mimics FAORU-ST but allows learning
            normalized_freq = torch.arange(freq_dim, dtype=torch.float32) / freq_dim
            sigmoid_input = 20.0 * (normalized_freq - 0.3)
            init_lambda = 0.3 + 0.7 * torch.sigmoid(sigmoid_input)
            # Inverse sigmoid to get weights
            init_weights = torch.log(init_lambda / (1.0 - init_lambda + 1e-8))
        
        elif init_strategy == 'random':
            # Random initialization
            init_weights = torch.randn(freq_dim) * 0.1
        
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")
        
        # Learnable parameters
        self.weights = nn.Parameter(init_weights)
    
    def forward(self) -> torch.Tensor:
        """
        Returns λ_k = σ(weights) ∈ [0, 1]
        """
        return torch.sigmoid(self.weights)
    
    def get_statistics(self) -> dict:
        """
        Get statistics of learned λ_k for analysis.
        """
        with torch.no_grad():
            lambda_k = self.forward()
            
            return {
                'mean': lambda_k.mean().item(),
                'std': lambda_k.std().item(),
                'min': lambda_k.min().item(),
                'max': lambda_k.max().item(),
                'lambda_k': lambda_k.cpu().numpy()
            }


def create_lambda_module(
    variant: str,
    freq_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create lambda module.
    
    Args:
        variant: 'piecewise', 'smooth', or 'learnable'
        freq_dim: Number of frequency components
        **kwargs: Additional arguments passed to the module
        
    Returns:
        Lambda module
    """
    if variant == 'piecewise':
        return PiecewiseConstantLambda(freq_dim, **kwargs)
    elif variant == 'smooth':
        return SmoothTransitionLambda(freq_dim, **kwargs)
    elif variant == 'learnable':
        return LearnableLambda(freq_dim, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
