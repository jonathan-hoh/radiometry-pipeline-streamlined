#!/usr/bin/env python3
"""
test_metrics.py - Unit tests for validation metrics

Comprehensive test suite for all validation metrics functions including:
- Quaternion attitude error calculations
- Star identification performance metrics
- Astrometric residual analysis
- Signal-to-noise ratio calculations
- Edge cases and error handling

Run with:
    PYTHONPATH=. python -m pytest validation/tests/test_metrics.py -v
"""

import numpy as np
import pytest
import logging
from typing import Dict, Any

# Import the metrics module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.metrics import (
    attitude_error_angle,
    quaternion_component_errors,
    identification_rate,
    astrometric_residuals,
    centroid_rms,
    calculate_snr,
    confusion_matrix_metrics,
    validate_quaternion,
    ValidationResults
)

class TestAttitudeErrorAngle:
    """Test attitude error angle calculations."""
    
    def test_identity_quaternions(self):
        """Test error between identical quaternions is zero."""
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        error = attitude_error_angle(q_id, q_id)
        assert abs(error) < 1e-10, f"Identity error should be ~0, got {error}"
        
    def test_known_rotation_angle(self):
        """Test error calculation for known rotation."""
        # 90° rotation about z-axis
        q1 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        q2 = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2])  # 90° about z
        
        error = attitude_error_angle(q1, q2)
        expected = 90.0 * 3600.0  # 90 degrees in arcseconds
        
        assert abs(error - expected) < 1.0, f"Expected {expected:.1f} arcsec, got {error:.1f}"
        
    def test_small_angle_approximation(self):
        """Test small angle cases for numerical precision."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        # Small rotation (1 arcsecond ≈ 4.85e-6 radians)
        small_angle_rad = 1.0 / 206264.8062471  # 1 arcsecond
        q2 = np.array([np.cos(small_angle_rad/2), np.sin(small_angle_rad/2), 0.0, 0.0])
        
        error = attitude_error_angle(q1, q2)
        assert abs(error - 1.0) < 0.01, f"Expected ~1 arcsec, got {error:.3f}"
        
    def test_quaternion_double_cover(self):
        """Test that q and -q give same rotation error."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.9, 0.1, 0.1, 0.4])
        q2 = q2 / np.linalg.norm(q2)  # Normalize
        
        error1 = attitude_error_angle(q1, q2)
        error2 = attitude_error_angle(q1, -q2)  # Opposite quaternion
        
        assert abs(error1 - error2) < 1e-10, "Quaternion double-cover not handled"
        
    def test_return_axis(self):
        """Test rotation axis calculation."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2])  # 90° about z
        
        error, axis = attitude_error_angle(q1, q2, return_axis=True)
        
        assert len(axis) == 3, "Axis should be 3D vector"
        assert abs(np.linalg.norm(axis) - 1.0) < 1e-10, "Axis should be unit vector"
        assert abs(axis[2] - 1.0) < 1e-6, "Should be z-axis rotation"
        
    def test_invalid_quaternions(self):
        """Test error handling for invalid quaternions."""
        with pytest.raises(ValueError):
            attitude_error_angle(np.array([1, 2, 3]), np.array([1, 0, 0, 0]))
            
        with pytest.raises(ValueError):
            attitude_error_angle(np.array([0, 0, 0, 0]), np.array([1, 0, 0, 0]))

class TestQuaternionComponentErrors:
    """Test quaternion component error calculations."""
    
    def test_identical_quaternions(self):
        """Test zero errors for identical quaternions."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        errors = quaternion_component_errors(q, q)
        
        for component in ['w', 'x', 'y', 'z']:
            assert abs(errors[component]) < 1e-15
        assert errors['norm_error'] < 1e-15
        
    def test_component_differences(self):
        """Test component error calculations."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.9, 0.1, 0.0, 0.0])
        q2 = q2 / np.linalg.norm(q2)  # Normalize
        
        errors = quaternion_component_errors(q1, q2)
        
        assert 'w' in errors and 'x' in errors and 'y' in errors and 'z' in errors
        assert 'norm_error' in errors
        
    def test_double_cover_handling(self):
        """Test that double-cover is handled (sign selection)."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.9, 0.1, 0.1, 0.4])
        q2 = q2 / np.linalg.norm(q2)
        
        errors1 = quaternion_component_errors(q1, q2)
        errors2 = quaternion_component_errors(q1, -q2)
        
        # Errors should be similar after sign selection
        assert (abs(errors1['w'] - errors2['w']) < 1e-10 or 
                abs(errors1['w'] + errors2['w']) < 1e-10)

class TestIdentificationRate:
    """Test star identification rate calculations."""
    
    def test_perfect_identification(self):
        """Test perfect identification scenario."""
        detected = [1, 2, 3, 4, 5]
        matched = [1, 2, 3, 4, 5]
        catalog = [1, 2, 3, 4, 5]
        
        rates = identification_rate(detected, matched, catalog)
        
        assert rates['identification_rate'] == 1.0
        assert rates['detection_rate'] == 1.0
        assert rates['matching_rate'] == 1.0
        assert rates['false_positive_rate'] == 0.0
        
    def test_partial_identification(self):
        """Test partial identification scenario."""
        detected = [1, 2, 3, 4, 99]  # 5 detections, 1 false positive
        matched = [1, 2, 3]          # 3 correct matches
        catalog = [1, 2, 3, 4, 5]    # 5 catalog stars
        
        rates = identification_rate(detected, matched, catalog)
        
        assert rates['identification_rate'] == 0.6  # 3/5
        assert rates['detection_rate'] == 0.8       # 4/5 (assuming 4 real detections)
        assert rates['matching_rate'] == 0.6        # 3/5
        assert rates['false_positive_rate'] == 0.4   # 2/5
        
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        rates = identification_rate([], [], [])
        
        for rate in rates.values():
            assert rate == 0.0
            
        # Empty catalog case
        rates = identification_rate([1, 2], [1], [])
        assert rates['identification_rate'] == 0.0
        
    def test_more_matches_than_detections(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError):
            identification_rate([1, 2], [1, 2, 3], [1, 2, 3, 4])

class TestAstrometricResiduals:
    """Test astrometric residual calculations."""
    
    def test_zero_residuals(self):
        """Test identical positions give zero residuals."""
        positions = np.array([[100, 200], [300, 400]])
        residuals = astrometric_residuals(positions, positions)
        
        assert residuals['mean_x'] == 0.0
        assert residuals['mean_y'] == 0.0
        assert residuals['rms_x'] == 0.0
        assert residuals['rms_y'] == 0.0
        assert residuals['rms_radial'] == 0.0
        
    def test_known_residuals(self):
        """Test residual calculations with known offsets."""
        catalog = np.array([[0, 0], [10, 0]])
        simulated = np.array([[1, 0], [11, 2]])  # +1 in x, +2 in y for second star
        
        residuals = astrometric_residuals(catalog, simulated)
        
        assert residuals['mean_x'] == 1.0
        assert residuals['mean_y'] == 1.0
        assert residuals['rms_x'] == 1.0
        np.testing.assert_almost_equal(residuals['rms_y'], np.sqrt(2), decimal=10)
        
    def test_shape_validation(self):
        """Test input shape validation."""
        with pytest.raises(ValueError):
            astrometric_residuals(np.array([[1, 2]]), np.array([[1, 2, 3]]))
            
        with pytest.raises(ValueError):
            astrometric_residuals(np.array([1, 2]), np.array([1, 2]))
            
    def test_empty_arrays(self):
        """Test handling of empty position arrays."""
        empty = np.array([]).reshape(0, 2)
        residuals = astrometric_residuals(empty, empty)
        
        assert len(residuals['residuals_x']) == 0
        assert residuals['rms_radial'] == 0.0

class TestCentroidRMS:
    """Test centroid RMS calculations."""
    
    def test_known_rms(self):
        """Test RMS calculation with known values."""
        residuals = np.array([3, 4])  # Should give RMS = 5
        rms = centroid_rms(residuals)
        assert abs(rms - 5.0) < 1e-15
        
    def test_zero_residuals(self):
        """Test RMS of zero residuals."""
        residuals = np.array([0, 0, 0])
        rms = centroid_rms(residuals)
        assert rms == 0.0
        
    def test_empty_array(self):
        """Test empty array handling."""
        rms = centroid_rms(np.array([]))
        assert rms == 0.0
        
    def test_multidimensional_array(self):
        """Test multidimensional array handling."""
        residuals = np.array([[1, 2], [3, 4]])
        rms = centroid_rms(residuals)
        expected = np.sqrt(np.mean([1, 4, 9, 16]))
        assert abs(rms - expected) < 1e-15

class TestCalculateSNR:
    """Test signal-to-noise ratio calculations."""
    
    def test_high_signal_snr(self):
        """Test high signal case where shot noise dominates."""
        signal = 10000.0  # electrons
        read_noise = 13.0  # electrons
        
        snr = calculate_snr(signal, read_noise, include_shot_noise=True)
        expected = signal / np.sqrt(signal + read_noise**2)
        
        assert abs(snr - expected) < 1e-10
        
    def test_low_signal_snr(self):
        """Test low signal case where read noise dominates.""" 
        signal = 100.0
        read_noise = 50.0
        
        snr = calculate_snr(signal, read_noise, include_shot_noise=True)
        expected = signal / np.sqrt(signal + read_noise**2)
        
        assert abs(snr - expected) < 1e-10
        
    def test_no_shot_noise(self):
        """Test SNR calculation without shot noise."""
        signal = 1000.0
        read_noise = 13.0
        
        snr = calculate_snr(signal, read_noise, include_shot_noise=False)
        expected = signal / read_noise
        
        assert abs(snr - expected) < 1e-15
        
    def test_zero_signal(self):
        """Test zero signal handling."""
        snr = calculate_snr(0.0, 13.0)
        assert snr == 0.0
        
    def test_array_inputs(self):
        """Test array inputs."""
        signals = np.array([100, 1000, 10000])
        read_noise = 13.0
        
        snrs = calculate_snr(signals, read_noise)
        assert len(snrs) == 3
        assert all(snrs >= 0)
        
    def test_negative_inputs(self):
        """Test error handling for negative inputs."""
        with pytest.raises(ValueError):
            calculate_snr(-100, 13.0)
            
        with pytest.raises(ValueError):
            calculate_snr(100, -13.0)

class TestConfusionMatrixMetrics:
    """Test confusion matrix metrics calculations."""
    
    def test_perfect_classification(self):
        """Test perfect classification metrics."""
        metrics = confusion_matrix_metrics(
            true_positives=10,
            false_positives=0,
            false_negatives=0,
            true_negatives=5
        )
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['specificity'] == 1.0
        
    def test_realistic_classification(self):
        """Test realistic classification scenario."""
        metrics = confusion_matrix_metrics(
            true_positives=8,
            false_positives=2,
            false_negatives=1,
            true_negatives=3
        )
        
        expected_precision = 8 / (8 + 2)  # 0.8
        expected_recall = 8 / (8 + 1)     # 8/9
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        expected_specificity = 3 / (3 + 2)  # 0.6
        
        assert abs(metrics['precision'] - expected_precision) < 1e-15
        assert abs(metrics['recall'] - expected_recall) < 1e-15
        assert abs(metrics['f1_score'] - expected_f1) < 1e-15
        assert abs(metrics['specificity'] - expected_specificity) < 1e-15
        
    def test_edge_cases(self):
        """Test edge cases for confusion matrix."""
        # No positive detections
        metrics = confusion_matrix_metrics(0, 0, 5, 10)
        assert metrics['precision'] == 0.0
        
        # No ground truth positives
        metrics = confusion_matrix_metrics(0, 5, 0, 10)
        assert metrics['recall'] == 0.0

class TestValidateQuaternion:
    """Test quaternion validation function."""
    
    def test_valid_quaternion(self):
        """Test validation of valid quaternions."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        q_validated = validate_quaternion(q)
        
        np.testing.assert_array_almost_equal(q, q_validated)
        
    def test_normalization(self):
        """Test automatic normalization."""
        q = np.array([2.0, 0.0, 0.0, 0.0])  # Not normalized
        q_validated = validate_quaternion(q)
        
        assert abs(np.linalg.norm(q_validated) - 1.0) < 1e-15
        assert q_validated[0] == 1.0  # Should be normalized to [1,0,0,0]
        
    def test_sign_convention(self):
        """Test positive scalar component convention."""
        q = np.array([-1.0, 0.0, 0.0, 0.0])  # Negative scalar
        q_validated = validate_quaternion(q)
        
        assert q_validated[0] > 0  # Should flip sign
        
    def test_invalid_shapes(self):
        """Test invalid quaternion shapes."""
        with pytest.raises(ValueError):
            validate_quaternion(np.array([1, 2, 3]))
            
        with pytest.raises(ValueError):
            validate_quaternion(np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))
            
    def test_zero_quaternion(self):
        """Test zero quaternion handling."""
        with pytest.raises(ValueError):
            validate_quaternion(np.array([0, 0, 0, 0]))

class TestValidationResults:
    """Test ValidationResults dataclass."""
    
    def test_creation(self):
        """Test ValidationResults creation."""
        result = ValidationResults(
            metric_name="test_metric",
            value=1.23,
            units="arcsec",
            timestamp="2024-01-01T00:00:00",
            parameters={"param1": "value1"}
        )
        
        assert result.metric_name == "test_metric"
        assert result.value == 1.23
        assert result.units == "arcsec"
        assert result.success == True  # Default value
        assert result.error_message is None  # Default value

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])