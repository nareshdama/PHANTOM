# PHANTOM — Core module package
# Proportional-Navigation Heuristic Adaptive Missile-Tracking Nullification

from core.injection import (
    InjectionProfile,
    NoInjection,
    StepInjection,
    LinearRampInjection,
    PiecewiseInjection,
    AdaptiveInjection,
)
from core.simulator import PhantomSimulator

__all__ = [
    "InjectionProfile",
    "NoInjection",
    "StepInjection",
    "LinearRampInjection",
    "PiecewiseInjection",
    "AdaptiveInjection",
    "PhantomSimulator",
]
