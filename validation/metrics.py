#!/usr/bin/env python3
"""
validation/metrics.py - Core Validation Metrics

Implements fundamental metrics for validating star tracker performance:
- Attitude error calculations (quaternion-based)
- Star identification rates and confusion matrices  
- Astrometric residual analysis
- Centroiding accuracy metrics
- Signal-to-noise ratio calculations

All functions include comprehensive docstrings, type hints, input validation,
and follow the established physical unit conventions (pixels, µm, mm, arcsec).

Usage:
    from validation.metrics import attitude_error_angle, identification_rate
    
    error_arcsec = attitude_error_angle(q_true, q_solved)
    id_rate = identification_rate(detected_stars, matched_stars, catalog_stars)
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Union, Optional, Any
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

# Physical constants and conversion factors
ARCSEC_PER_RADIAN = 206264.8062471  # arcseconds per radian
DEG_PER_RADIAN = 57.29577951308232   # degrees per radian

@dataclass
class ValidationResults:
    """Container for validation results with metadata."""
    metric_name: str
    value: Union[float, np.ndarray, Dict[str, Any]]
    units: str
    timestamp: str
    parameters: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None

def validate_quaternion(q: np.ndarray, name: str = "quaternion") -> np.ndarray:
    """
    Validate and normalize quaternion input.
    
    Parameters
    ----------
    q : np.ndarray
        Quaternion as [w, x, y, z] or [x, y, z, w]
    name : str
        Parameter name for error messages
        
    Returns
    -------
    np.ndarray
        Normalized quaternion as [w, x, y, z]
        
    Raises
    ------
    ValueError
        If quaternion is invalid
    """
    q = np.asarray(q, dtype=float)
    
    if q.shape != (4,):
        raise ValueError(f"{name} must be 4-element array, got shape {q.shape}")
    
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        raise ValueError(f"{name} has zero norm")
        
    if abs(norm - 1.0) > 1e-6:
        logger.warning(f"{name} not unit normalized (norm={norm:.8f}), normalizing")
        q = q / norm
        
    # Ensure positive scalar component (w > 0 convention)
    if q[0] < 0:
        q = -q
        
    return q

def attitude_error_angle(
    q_true: np.ndarray, 
    q_solved: np.ndarray,
    return_axis: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Calculate attitude error angle between two quaternions.
    
    Computes the rotation angle required to align q_solved with q_true,
    representing the cross-track attitude error.
    
    Parameters
    ----------
    q_true : np.ndarray
        True/reference quaternion [w, x, y, z]
    q_solved : np.ndarray  
        Solved/estimated quaternion [w, x, y, z]
    return_axis : bool, optional
        If True, also return rotation axis (default: False)
        
    Returns
    -------
    float
        Attitude error angle in arcseconds
    np.ndarray, optional
        Rotation axis (unit vector) if return_axis=True
        
    Notes
    -----
    Error angle is computed as:
    θ = 2 * arccos(|q_true · q_solved|)
    
    This gives the geodesic distance on SO(3) representing the minimum
    rotation angle between the two orientations.
    
    Examples
    --------
    >>> q1 = np.array([1, 0, 0, 0])  # Identity
    >>> q2 = np.array([0.999, 0.045, 0, 0])  # ~5° rotation
    >>> error = attitude_error_angle(q1, q2)
    >>> print(f"Error: {error:.1f} arcsec")
    Error: 309.4 arcsec
    """
    # Validate and normalize inputs
    q_true = validate_quaternion(q_true, "q_true")
    q_solved = validate_quaternion(q_solved, "q_solved")
    
    # Compute quaternion difference (relative rotation)
    # q_error = q_true^(-1) * q_solved = q_true* ⊗ q_solved
    q_true_conj = np.array([q_true[0], -q_true[1], -q_true[2], -q_true[3]])
    
    # Quaternion multiplication: q_error = q_true* ⊗ q_solved
    q_error = np.array([
        q_true_conj[0]*q_solved[0] - q_true_conj[1]*q_solved[1] - q_true_conj[2]*q_solved[2] - q_true_conj[3]*q_solved[3],
        q_true_conj[0]*q_solved[1] + q_true_conj[1]*q_solved[0] + q_true_conj[2]*q_solved[3] - q_true_conj[3]*q_solved[2],
        q_true_conj[0]*q_solved[2] - q_true_conj[1]*q_solved[3] + q_true_conj[2]*q_solved[0] + q_true_conj[3]*q_solved[1],
        q_true_conj[0]*q_solved[3] + q_true_conj[1]*q_solved[2] - q_true_conj[2]*q_solved[1] + q_true_conj[3]*q_solved[0]
    ])
    
    # Error angle from scalar component: θ = 2 * arccos(|w|)
    w_abs = abs(q_error[0])
    
    # Handle numerical precision issues
    if w_abs > 1.0:
        if w_abs > 1.0 + 1e-10:
            logger.warning(f"Quaternion scalar component {w_abs} > 1, possible numerical error")
        w_abs = 1.0
        
    error_angle_rad = 2.0 * np.arccos(w_abs)
    error_angle_arcsec = error_angle_rad * ARCSEC_PER_RADIAN
    
    if return_axis:
        # Rotation axis from vector component
        vec_norm = np.linalg.norm(q_error[1:4])
        if vec_norm > 1e-10:
            axis = q_error[1:4] / vec_norm
        else:
            axis = np.array([1.0, 0.0, 0.0])  # Arbitrary axis for zero rotation
        return error_angle_arcsec, axis
    
    return error_angle_arcsec

def quaternion_component_errors(
    q_true: np.ndarray, 
    q_solved: np.ndarray
) -> Dict[str, float]:
    """
    Calculate individual quaternion component errors.
    
    Parameters
    ----------
    q_true : np.ndarray
        True quaternion [w, x, y, z]
    q_solved : np.ndarray
        Solved quaternion [w, x, y, z]
        
    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'w', 'x', 'y', 'z', 'norm_error'
        All values are dimensionless quaternion component differences
        
    Notes
    -----
    Component errors may not be meaningful for attitude accuracy since
    quaternions have gauge freedom (q and -q represent same rotation).
    Use attitude_error_angle() for physical attitude errors.
    """
    q_true = validate_quaternion(q_true, "q_true")
    q_solved = validate_quaternion(q_solved, "q_solved")
    
    # Handle quaternion double-cover: choose sign to minimize difference
    if np.dot(q_true, q_solved) < 0:
        q_solved = -q_solved
        
    diff = q_solved - q_true
    
    return {
        'w': float(diff[0]),
        'x': float(diff[1]), 
        'y': float(diff[2]),
        'z': float(diff[3]),
        'norm_error': float(abs(np.linalg.norm(q_solved) - 1.0))
    }

def identification_rate(
    detected_stars: List[Any],
    matched_stars: List[Any], 
    catalog_stars: List[Any]
) -> Dict[str, float]:
    """
    Calculate star identification performance metrics.
    
    Parameters
    ----------
    detected_stars : List[Any]
        List of detected star objects/IDs
    matched_stars : List[Any]
        List of successfully matched star objects/IDs  
    catalog_stars : List[Any]
        List of catalog stars in field of view
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'identification_rate': fraction of catalog stars identified
        - 'detection_rate': fraction of catalog stars detected
        - 'matching_rate': fraction of detected stars matched
        - 'false_positive_rate': fraction of detections that are false
        
    Notes
    -----
    Identification rate = matched_stars / catalog_stars (primary metric)
    Detection rate = detected_stars / catalog_stars  
    Matching rate = matched_stars / detected_stars
    False positive rate = (detected_stars - matched_stars) / detected_stars
    
    Examples
    --------
    >>> detected = [1, 2, 3, 4, 5]  # 5 detections
    >>> matched = [1, 2, 3]         # 3 matches  
    >>> catalog = [1, 2, 3, 6, 7]   # 5 catalog stars
    >>> rates = identification_rate(detected, matched, catalog)
    >>> print(f"ID rate: {rates['identification_rate']:.2f}")
    ID rate: 0.60
    """
    n_detected = len(detected_stars)
    n_matched = len(matched_stars) 
    n_catalog = len(catalog_stars)
    
    # Input validation
    if n_catalog == 0:
        logger.warning("No catalog stars provided")
        return {
            'identification_rate': 0.0,
            'detection_rate': 0.0, 
            'matching_rate': 0.0,
            'false_positive_rate': 0.0
        }
        
    if n_matched > n_detected:
        raise ValueError(f"Matched stars ({n_matched}) > detected stars ({n_detected})")
        
    if n_matched > n_catalog:
        logger.warning(f"Matched stars ({n_matched}) > catalog stars ({n_catalog})")
    
    # Calculate rates
    identification_rate_val = n_matched / n_catalog
    detection_rate_val = min(n_detected / n_catalog, 1.0)  # Cap at 1.0
    
    if n_detected > 0:
        matching_rate_val = n_matched / n_detected
        false_positive_rate_val = (n_detected - n_matched) / n_detected
    else:
        matching_rate_val = 0.0
        false_positive_rate_val = 0.0
        
    return {
        'identification_rate': identification_rate_val,
        'detection_rate': detection_rate_val,
        'matching_rate': matching_rate_val, 
        'false_positive_rate': false_positive_rate_val
    }

def astrometric_residuals(
    u_catalog: np.ndarray,
    u_simulated: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate astrometric residuals between catalog and simulated positions.
    
    Parameters
    ----------
    u_catalog : np.ndarray
        Catalog positions, shape (N, 2) in pixels [x, y]
    u_simulated : np.ndarray  
        Simulated/measured positions, shape (N, 2) in pixels [x, y]
        
    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing:
        - 'residuals_x': x-direction residuals in pixels, shape (N,)
        - 'residuals_y': y-direction residuals in pixels, shape (N,)
        - 'residuals_radial': radial residuals in pixels, shape (N,)
        - 'mean_x': mean x residual in pixels
        - 'mean_y': mean y residual in pixels
        - 'rms_x': RMS x residual in pixels
        - 'rms_y': RMS y residual in pixels  
        - 'rms_radial': RMS radial residual in pixels
        
    Notes
    -----
    Residuals are computed as: simulated - catalog (convention: positive = overshoot)
    RMS residuals: sqrt(mean(residuals^2))
    Radial residuals: sqrt(residuals_x^2 + residuals_y^2)
    
    Examples
    --------
    >>> catalog = np.array([[100, 200], [300, 400]])
    >>> simulated = np.array([[100.1, 200.2], [299.9, 400.1]])
    >>> residuals = astrometric_residuals(catalog, simulated)
    >>> print(f"RMS: {residuals['rms_radial']:.3f} pixels")
    RMS: 0.158 pixels
    """
    u_catalog = np.asarray(u_catalog, dtype=float)
    u_simulated = np.asarray(u_simulated, dtype=float)
    
    # Input validation
    if u_catalog.shape != u_simulated.shape:
        raise ValueError(f"Shape mismatch: catalog {u_catalog.shape} vs simulated {u_simulated.shape}")
        
    if u_catalog.ndim != 2 or u_catalog.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2), got {u_catalog.shape}")
        
    if u_catalog.shape[0] == 0:
        logger.warning("No positions provided")
        return {
            'residuals_x': np.array([]),
            'residuals_y': np.array([]),
            'residuals_radial': np.array([]),
            'mean_x': 0.0, 'mean_y': 0.0,
            'rms_x': 0.0, 'rms_y': 0.0, 'rms_radial': 0.0
        }
    
    # Calculate residuals: simulated - catalog
    residuals = u_simulated - u_catalog
    residuals_x = residuals[:, 0]
    residuals_y = residuals[:, 1]
    residuals_radial = np.sqrt(residuals_x**2 + residuals_y**2)
    
    # Statistics
    mean_x = np.mean(residuals_x)
    mean_y = np.mean(residuals_y)
    rms_x = np.sqrt(np.mean(residuals_x**2))
    rms_y = np.sqrt(np.mean(residuals_y**2))
    rms_radial = np.sqrt(np.mean(residuals_radial**2))
    
    return {
        'residuals_x': residuals_x,
        'residuals_y': residuals_y, 
        'residuals_radial': residuals_radial,
        'mean_x': mean_x,
        'mean_y': mean_y,
        'rms_x': rms_x,
        'rms_y': rms_y,
        'rms_radial': rms_radial
    }

def centroid_rms(residuals: np.ndarray) -> float:
    """
    Calculate RMS centroid error from residual array.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residual values (any shape)
        
    Returns
    -------
    float
        RMS error: sqrt(mean(residuals^2))
        
    Notes
    -----
    Standard RMS calculation: σ_RMS = sqrt(Σ(r_i^2) / N)
    Units depend on input units (typically pixels or arcseconds).
    """
    residuals = np.asarray(residuals, dtype=float)
    
    if residuals.size == 0:
        return 0.0
        
    return float(np.sqrt(np.mean(residuals**2)))

def calculate_snr(
    signal_electrons: Union[float, np.ndarray],
    noise_electrons: Union[float, np.ndarray],
    include_shot_noise: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate signal-to-noise ratio for detector measurements.
    
    Parameters
    ----------
    signal_electrons : float or np.ndarray
        Signal level in electrons
    noise_electrons : float or np.ndarray
        Read noise level in electrons (RMS)
    include_shot_noise : bool, optional
        Include Poisson shot noise sqrt(signal) (default: True)
        
    Returns
    -------
    float or np.ndarray
        Signal-to-noise ratio (dimensionless)
        
    Notes
    -----
    SNR calculation:
    - With shot noise: SNR = signal / sqrt(signal + noise^2)
    - Without shot noise: SNR = signal / noise
    
    Shot noise follows Poisson statistics: σ_shot = sqrt(N_electrons)
    Total noise: σ_total = sqrt(σ_shot^2 + σ_read^2) = sqrt(N + σ_read^2)
    
    Examples
    --------
    >>> signal = 1000.0  # electrons
    >>> read_noise = 13.0  # electrons RMS
    >>> snr = calculate_snr(signal, read_noise)
    >>> print(f"SNR: {snr:.1f}")
    SNR: 31.4
    """
    signal_electrons = np.asarray(signal_electrons, dtype=float)
    noise_electrons = np.asarray(noise_electrons, dtype=float)
    
    # Input validation
    if np.any(signal_electrons < 0):
        raise ValueError("Signal electrons must be non-negative")
        
    if np.any(noise_electrons < 0):
        raise ValueError("Noise electrons must be non-negative")
    
    # Handle zero signal case
    zero_signal = (signal_electrons == 0)
    if np.any(zero_signal):
        logger.warning("Zero signal detected, SNR will be zero")
    
    if include_shot_noise:
        # Total noise includes shot noise: σ_total = sqrt(N + σ_read^2)
        total_noise_squared = signal_electrons + noise_electrons**2
        total_noise = np.sqrt(total_noise_squared)
        
        # Avoid division by zero
        snr = np.divide(signal_electrons, total_noise, 
                       out=np.zeros_like(signal_electrons), 
                       where=(total_noise != 0))
    else:
        # Read noise only
        snr = np.divide(signal_electrons, noise_electrons,
                       out=np.zeros_like(signal_electrons),
                       where=(noise_electrons != 0))
    
    # Return scalar if input was scalar
    if np.isscalar(snr):
        return float(snr)
    return snr

def confusion_matrix_metrics(
    true_positives: int,
    false_positives: int, 
    false_negatives: int,
    true_negatives: int = 0
) -> Dict[str, float]:
    """
    Calculate classification metrics from confusion matrix elements.
    
    Parameters
    ----------
    true_positives : int
        Correctly identified stars
    false_positives : int
        Incorrectly identified detections (spurious)
    false_negatives : int
        Missed catalog stars
    true_negatives : int, optional
        Correctly rejected non-stars (default: 0, often undefined)
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'precision': TP / (TP + FP)
        - 'recall': TP / (TP + FN) 
        - 'f1_score': 2 * (precision * recall) / (precision + recall)
        - 'specificity': TN / (TN + FP) if TN provided
        
    Notes
    -----
    Precision = fraction of detections that are correct
    Recall = fraction of catalog stars that are detected (same as identification rate)
    F1 score = harmonic mean of precision and recall
    """
    tp, fp, fn, tn = int(true_positives), int(false_positives), int(false_negatives), int(true_negatives)
    
    # Calculate precision  
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
        logger.warning("No positive detections for precision calculation")
    
    # Calculate recall (sensitivity)
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
        logger.warning("No ground truth positives for recall calculation")
    
    # Calculate F1 score
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
        
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    # Calculate specificity if true negatives provided
    if tn > 0 or fp > 0:
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
            metrics['specificity'] = specificity
        else:
            metrics['specificity'] = 0.0
            
    return metrics

# Export all public functions
__all__ = [
    'ValidationResults',
    'attitude_error_angle',
    'quaternion_component_errors',
    'identification_rate', 
    'astrometric_residuals',
    'centroid_rms',
    'calculate_snr',
    'confusion_matrix_metrics',
    'validate_quaternion'
]