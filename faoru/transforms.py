"""
Frequency Transforms

Implements orthonormal transforms along the feature dimension:
1. Real-valued FFT (rFFT)
2. Discrete Cosine Transform (DCT)
3. Hadamard Transform
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class RealFFT(nn.Module):
    """
    Real-valued FFT along feature dimension.
    
    Uses torch.fft.rfft which exploits conjugate symmetry:
    - Input: [B, S, D] (real)
    - Output: [B, S, D//2+1, 2] (complex as real pairs)
    
    The inverse automatically maintains real outputs via conjugate symmetry.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.freq_dim = dim // 2 + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rFFT along last dimension.
        
        Args:
            x: Real tensor [B, S, D]
            
        Returns:
            X_omega: Complex tensor as real pairs [B, S, D//2+1, 2]
        """
        # torch.fft.rfft returns complex tensor
        X_complex = torch.fft.rfft(x, dim=-1, norm='ortho')  # [B, S, D//2+1]
        
        # Convert to real pairs for easier manipulation
        X_omega = torch.view_as_real(X_complex)  # [B, S, D//2+1, 2]
        
        return X_omega
    
    def inverse(self, X_omega: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse rFFT.
        
        Args:
            X_omega: Complex tensor as real pairs [B, S, D//2+1, 2]
            
        Returns:
            x: Real tensor [B, S, D]
        """
        # Convert from real pairs to complex
        X_complex = torch.view_as_complex(X_omega)  # [B, S, D//2+1]
        
        # Inverse FFT
        x = torch.fft.irfft(X_complex, n=self.dim, dim=-1, norm='ortho')  # [B, S, D]
        
        return x
    
    def get_freq_dim(self) -> int:
        return self.freq_dim


class DCT(nn.Module):
    """
    Discrete Cosine Transform (Type-II) along feature dimension.
    
    DCT is a real-to-real transform, often used in compression (JPEG, MP3).
    More efficient than FFT for certain hardware.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Precompute DCT basis matrix
        # DCT-II: X[k] = sum_n x[n] * cos(π * k * (n + 0.5) / D)
        n = torch.arange(dim, dtype=torch.float32)
        k = torch.arange(dim, dtype=torch.float32).unsqueeze(1)
        
        dct_matrix = torch.cos(torch.pi * k * (n + 0.5) / dim)
        
        # Orthonormalization
        dct_matrix[0, :] *= np.sqrt(1.0 / dim)
        dct_matrix[1:, :] *= np.sqrt(2.0 / dim)
        
        self.register_buffer('dct_matrix', dct_matrix)
        self.register_buffer('idct_matrix', dct_matrix.T)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DCT along last dimension.
        
        Args:
            x: Real tensor [B, S, D]
            
        Returns:
            X_dct: Real tensor [B, S, D, 2] (padded to match FFT interface)
        """
        # Apply DCT via matrix multiplication
        X_dct_real = torch.matmul(x, self.dct_matrix.T)  # [B, S, D]
        
        # Pad with zeros to match complex format [B, S, D, 2]
        X_dct = torch.stack([X_dct_real, torch.zeros_like(X_dct_real)], dim=-1)
        
        return X_dct
    
    def inverse(self, X_dct: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse DCT.
        
        Args:
            X_dct: Real tensor [B, S, D, 2] (only real part used)
            
        Returns:
            x: Real tensor [B, S, D]
        """
        # Extract real part
        X_dct_real = X_dct[..., 0]  # [B, S, D]
        
        # Apply inverse DCT
        x = torch.matmul(X_dct_real, self.idct_matrix.T)  # [B, S, D]
        
        return x
    
    def get_freq_dim(self) -> int:
        return self.dim


class HadamardTransform(nn.Module):
    """
    Hadamard Transform (Walsh-Hadamard) along feature dimension.
    
    Fast transform with O(D log D) complexity, useful for efficient hardware.
    Requires D to be a power of 2.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        
        # Check if dim is power of 2
        if not (dim & (dim - 1) == 0 and dim != 0):
            raise ValueError(f"Hadamard transform requires dim to be power of 2, got {dim}")
        
        self.dim = dim
        
        # Precompute Hadamard matrix
        hadamard_matrix = self._compute_hadamard_matrix(dim)
        
        self.register_buffer('hadamard_matrix', hadamard_matrix)
    
    def _compute_hadamard_matrix(self, n: int) -> torch.Tensor:
        """
        Recursively compute Hadamard matrix.
        
        H_1 = [1]
        H_n = [H_{n/2}  H_{n/2}]
              [H_{n/2} -H_{n/2}]
        """
        if n == 1:
            return torch.tensor([[1.0]])
        else:
            H_half = self._compute_hadamard_matrix(n // 2)
            H = torch.cat([
                torch.cat([H_half, H_half], dim=1),
                torch.cat([H_half, -H_half], dim=1)
            ], dim=0)
            return H
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Hadamard transform.
        
        Args:
            x: Real tensor [B, S, D]
            
        Returns:
            X_had: Real tensor [B, S, D, 2] (padded to match FFT interface)
        """
        # Normalize
        X_had_real = torch.matmul(x, self.hadamard_matrix.T) / np.sqrt(self.dim)
        
        # Pad to match complex format
        X_had = torch.stack([X_had_real, torch.zeros_like(X_had_real)], dim=-1)
        
        return X_had
    
    def inverse(self, X_had: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse Hadamard transform.
        
        Note: Hadamard transform is self-inverse (up to scaling).
        
        Args:
            X_had: Real tensor [B, S, D, 2]
            
        Returns:
            x: Real tensor [B, S, D]
        """
        X_had_real = X_had[..., 0]
        
        # Inverse is the same as forward (self-inverse property)
        x = torch.matmul(X_had_real, self.hadamard_matrix.T) / np.sqrt(self.dim)
        
        return x
    
    def get_freq_dim(self) -> int:
        return self.dim


def create_transform(
    transform_type: str,
    dim: int
) -> nn.Module:
    """
    Factory function to create frequency transform.
    
    Args:
        transform_type: 'fft', 'dct', or 'hadamard'
        dim: Feature dimension
        
    Returns:
        Transform module
    """
    if transform_type == 'fft':
        return RealFFT(dim)
    elif transform_type == 'dct':
        return DCT(dim)
    elif transform_type == 'hadamard':
        return HadamardTransform(dim)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
