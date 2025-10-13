"""
Validation Framework for Star Tracker Radiometry Pipeline

This package provides comprehensive validation tools for the star tracker simulation,
including attitude determination accuracy, star identification performance, 
astrometric precision, photometric calibration, and noise characterization.

Modules:
    metrics: Core validation metrics and error calculations
    monte_carlo: Monte Carlo framework for statistical validation
    attitude_validation: Attitude determination accuracy validation
    identification_validation: Star identification performance validation  
    astrometric_validation: Astrometric precision validation
    photometric_validation: Photometric calibration validation
    noise_validation: Noise characterization and sensitivity analysis
    reporting: Results aggregation and report generation
"""

__version__ = "1.0.0"
__author__ = "Star Tracker Validation Team"

# Import key classes for convenience
from .metrics import *
from .monte_carlo import MonteCarloValidator, ValidationScenario, ValidationResult

__all__ = [
    'MonteCarloValidator',
    'ValidationScenario', 
    'ValidationResult',
    'attitude_error_angle',
    'quaternion_component_errors', 
    'identification_rate',
    'astrometric_residuals',
    'centroid_rms',
    'calculate_snr'
]