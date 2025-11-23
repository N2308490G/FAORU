"""
FAORU: Frequency-Adaptive Orthogonal Residual Updates

Core implementation of the FAORU residual mechanism.
"""

__version__ = "1.0.0"
__author__ = "FAORU Team"

from .residual import FAORUResidual
from .variants import (
    PiecewiseConstantLambda,
    SmoothTransitionLambda,
    LearnableLambda
)
from .transforms import (
    RealFFT,
    DCT,
    HadamardTransform
)

__all__ = [
    'FAORUResidual',
    'PiecewiseConstantLambda',
    'SmoothTransitionLambda',
    'LearnableLambda',
    'RealFFT',
    'DCT',
    'HadamardTransform'
]
