"""
Unit tests for FAORU residual module
"""

import torch
import pytest

from faoru.residual import FAORUResidual, compute_alpha_k
from faoru.variants import PiecewiseConstantLambda, SmoothTransitionLambda, LearnableLambda
from faoru.transforms import RealFFT, DCT, HadamardTransform


class TestAlphaComputation:
    """Test alpha_k computation"""
    
    def test_alpha_k_shape(self):
        """Test output shape of compute_alpha_k"""
        B, S, D = 4, 196, 385  # D//2+1 for rFFT
        F = torch.randn(B, S, D, dtype=torch.complex64)
        X = torch.randn(B, S, D, dtype=torch.complex64)
        
        alpha_k = compute_alpha_k(F, X)
        
        assert alpha_k.shape == (B, S, D)
        assert alpha_k.dtype == torch.float32  # Real-valued output
    
    def test_alpha_k_orthogonal(self):
        """Test alpha_k for orthogonal F and X"""
        B, S, D = 2, 100, 385
        
        # Create orthogonal F and X (inner product = 0)
        F = torch.randn(B, S, D, dtype=torch.complex64)
        X = torch.randn(B, S, D, dtype=torch.complex64)
        X = X - (F.real * X.real + F.imag * X.imag).unsqueeze(-1) * F  # Make orthogonal
        
        alpha_k = compute_alpha_k(F, X)
        
        # Should be close to zero
        assert torch.allclose(alpha_k, torch.zeros_like(alpha_k), atol=1e-3)
    
    def test_alpha_k_parallel(self):
        """Test alpha_k for parallel F and X"""
        B, S, D = 2, 100, 385
        
        X = torch.randn(B, S, D, dtype=torch.complex64)
        F = 2.0 * X  # F parallel to X
        
        alpha_k = compute_alpha_k(F, X)
        
        # Should be close to 2.0
        assert torch.allclose(alpha_k, torch.full_like(alpha_k, 2.0), rtol=1e-2)


class TestFAORUResidual:
    """Test FAORU residual module"""
    
    @pytest.mark.parametrize('variant', ['learnable', 'piecewise', 'smooth'])
    @pytest.mark.parametrize('transform_type', ['fft', 'dct', 'hadamard'])
    def test_faoru_shape(self, variant, transform_type):
        """Test output shape for different variants and transforms"""
        if transform_type == 'hadamard':
            dim = 512  # Power of 2 for Hadamard
        else:
            dim = 768
        
        residual = FAORUResidual(dim=dim, variant=variant, transform_type=transform_type)
        
        x = torch.randn(2, 196, dim)
        f_x = torch.randn(2, 196, dim)
        
        output = residual(x, f_x)
        
        assert output.shape == (2, 196, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_faoru_zero_residual(self):
        """Test FAORU with zero residual returns identity"""
        residual = FAORUResidual(dim=768, variant='learnable')
        
        x = torch.randn(2, 196, 768)
        f_x = torch.zeros(2, 196, 768)
        
        output = residual(x, f_x)
        
        # With zero residual, output should approximate identity
        assert torch.allclose(output, x, atol=1e-3)
    
    def test_faoru_gradient_flow(self):
        """Test gradient flow through FAORU"""
        residual = FAORUResidual(dim=768, variant='learnable')
        residual.train()
        
        x = torch.randn(2, 196, 768, requires_grad=True)
        f_x = torch.randn(2, 196, 768, requires_grad=True)
        
        output = residual(x, f_x)
        loss = output.sum()
        loss.backward()
        
        # Gradients should exist
        assert x.grad is not None
        assert f_x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(f_x.grad).any()
    
    def test_faoru_frequency_analysis(self):
        """Test frequency analysis return"""
        residual = FAORUResidual(dim=768, variant='learnable')
        
        x = torch.randn(2, 196, 768)
        f_x = torch.randn(2, 196, 768)
        
        output, freq_analysis = residual(x, f_x, return_freq_analysis=True)
        
        assert 'F' in freq_analysis
        assert 'X' in freq_analysis
        assert 'alpha_k' in freq_analysis
        assert 'lambda_k' in freq_analysis
        assert 'F_perp' in freq_analysis


class TestLambdaVariants:
    """Test lambda parameterizations"""
    
    def test_piecewise_constant(self):
        """Test piecewise constant lambda"""
        lambda_module = PiecewiseConstantLambda(freq_dim=385, cutoff_ratio=0.3)
        
        lambda_k = lambda_module()
        
        assert lambda_k.shape == (385,)
        
        # Check cutoff at 30%
        cutoff_idx = int(385 * 0.3)
        assert torch.all(lambda_k[:cutoff_idx] == 1.0)
        assert torch.all(lambda_k[cutoff_idx:] == 0.0)
    
    def test_smooth_transition(self):
        """Test smooth transition lambda"""
        lambda_module = SmoothTransitionLambda(freq_dim=385, cutoff_ratio=0.3, slope=10.0)
        
        lambda_k = lambda_module()
        
        assert lambda_k.shape == (385,)
        
        # Should be smooth (values between 0 and 1)
        assert torch.all((lambda_k >= 0) & (lambda_k <= 1))
        
        # Should be monotonically decreasing
        assert torch.all(lambda_k[:-1] >= lambda_k[1:])
    
    def test_learnable_lambda(self):
        """Test learnable lambda"""
        lambda_module = LearnableLambda(freq_dim=385)
        
        lambda_k = lambda_module()
        
        assert lambda_k.shape == (385,)
        assert torch.all((lambda_k >= 0) & (lambda_k <= 1))
        
        # Check gradient flow
        loss = lambda_k.sum()
        loss.backward()
        
        assert lambda_module.weights.grad is not None


class TestFrequencyTransforms:
    """Test frequency transforms"""
    
    def test_rfft_forward_inverse(self):
        """Test rFFT forward and inverse"""
        transform = RealFFT(dim=768)
        
        x = torch.randn(2, 196, 768)
        F = transform.forward(x)
        x_recon = transform.inverse(F)
        
        assert F.shape == (2, 196, 385)  # 768//2+1
        assert x_recon.shape == x.shape
        assert torch.allclose(x, x_recon, atol=1e-4)
    
    def test_dct_forward_inverse(self):
        """Test DCT forward and inverse"""
        transform = DCT(dim=768)
        
        x = torch.randn(2, 196, 768)
        F = transform.forward(x)
        x_recon = transform.inverse(F)
        
        assert F.shape == x.shape
        assert torch.allclose(x, x_recon, atol=1e-4)
    
    def test_hadamard_forward_inverse(self):
        """Test Hadamard forward and inverse"""
        transform = HadamardTransform(dim=512)  # Must be power of 2
        
        x = torch.randn(2, 196, 512)
        F = transform.forward(x)
        x_recon = transform.inverse(F)
        
        assert F.shape == x.shape
        assert torch.allclose(x, x_recon, atol=1e-4)
    
    def test_hadamard_non_power_of_2(self):
        """Test Hadamard with non-power-of-2 dimension"""
        with pytest.raises(ValueError):
            transform = HadamardTransform(dim=768)  # Not power of 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
