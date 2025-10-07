#!/usr/bin/env python3
"""
Test script for focal length perturbation analysis.
Quick validation to ensure the perturbation framework works correctly.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Import our modules
from src.core.perturbation_model import (
    create_focal_length_scenario,
    PerturbationAnalyzer,
    ParameterVariation,
    DistributionType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_parameter_variation():
    """Test basic parameter variation functionality."""
    logger.info("Testing parameter variation...")
    
    # Test normal distribution
    param = ParameterVariation(
        name="test_param",
        nominal_value=25.0,
        distribution=DistributionType.NORMAL,
        parameters={'mean': 25.0, 'std': 0.5},
        units="mm"
    )
    
    # Generate samples
    samples = param.sample(size=1000)
    
    # Check statistics
    sample_mean = np.mean(samples)
    sample_std = np.std(samples)
    
    assert abs(sample_mean - 25.0) < 0.1, f"Mean test failed: {sample_mean}"
    assert abs(sample_std - 0.5) < 0.1, f"Std test failed: {sample_std}"
    
    logger.info(f"✓ Parameter variation test passed (mean: {sample_mean:.2f}, std: {sample_std:.2f})")

def test_focal_length_scenario():
    """Test focal length scenario creation."""
    logger.info("Testing focal length scenario creation...")
    
    scenario = create_focal_length_scenario(
        nominal_focal_length=25.0,
        thermal_variation=0.5,
        temperature_range=(-20, 60)
    )
    
    assert scenario.name == "Focal Length Thermal Variation"
    assert len(scenario.parameters) == 1
    assert scenario.parameters[0].name == "focal_length"
    assert scenario.parameters[0].nominal_value == 25.0
    
    # Test sampling
    samples = scenario.sample_all(size=100)
    assert 'focal_length' in samples
    assert len(samples['focal_length']) == 100
    
    logger.info("✓ Focal length scenario test passed")

def test_mock_simulation():
    """Test mock simulation function for perturbation analysis."""
    logger.info("Testing mock simulation function...")
    
    def mock_simulation_function(param_values, **kwargs):
        """Mock simulation that returns predictable results."""
        focal_length = param_values.get('focal_length', 25.0)
        
        # Simple linear relationship: attitude error = 2 * (focal_length - 25) + noise
        base_error = 5.0  # arcsec
        sensitivity = 2.0  # arcsec/mm
        noise = np.random.normal(0, 0.5)
        
        attitude_error = base_error + sensitivity * (focal_length - 25.0) + noise
        
        return {
            'simulation_success': True,
            'focal_length_mm': focal_length,
            'attitude_error_arcsec': abs(attitude_error),
            'sensitivity_test': sensitivity
        }
    
    # Create scenario and analyzer
    scenario = create_focal_length_scenario(
        nominal_focal_length=25.0,
        thermal_variation=0.6  # ±0.6mm variation
    )
    
    analyzer = PerturbationAnalyzer(scenario)
    
    # Run small Monte Carlo
    results_df = analyzer.run_monte_carlo(
        simulation_function=mock_simulation_function,
        n_trials=20
    )
    
    # Check results
    successful_trials = results_df[results_df['simulation_success'] == True]
    assert len(successful_trials) > 15, "Too many failed trials in mock simulation"
    
    # Test sensitivity analysis
    sensitivity = analyzer.analyze_sensitivity('attitude_error_arcsec')
    focal_sensitivity = sensitivity.get('focal_length', {})
    
    # Should have strong correlation since we designed it that way
    correlation = focal_sensitivity.get('correlation', 0)
    assert abs(correlation) > 0.5, f"Expected strong correlation, got {correlation}"
    
    logger.info(f"✓ Mock simulation test passed (correlation: {correlation:.3f})")

def create_test_visualization():
    """Create a simple test visualization."""
    logger.info("Creating test visualization...")
    
    # Generate synthetic test data
    np.random.seed(42)
    focal_lengths = np.random.normal(25.0, 0.2, 200)
    attitude_errors = 5.0 + 1.5 * (focal_lengths - 25.0) + np.random.normal(0, 0.3, 200)
    attitude_errors = np.abs(attitude_errors)  # Take absolute value
    
    # Create simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of focal lengths
    ax1.hist(focal_lengths, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Focal Length (mm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Focal Length Distribution\n(Test Data)')
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot of relationship
    ax2.scatter(focal_lengths, attitude_errors, alpha=0.6, s=30)
    
    # Fit trend line
    coeffs = np.polyfit(focal_lengths, attitude_errors, 1)
    trend_x = np.linspace(focal_lengths.min(), focal_lengths.max(), 100)
    trend_y = np.polyval(coeffs, trend_x)
    ax2.plot(trend_x, trend_y, 'r--', linewidth=2, 
             label=f'Trend: {coeffs[0]:.2f} arcsec/mm')
    
    ax2.set_xlabel('Focal Length (mm)')
    ax2.set_ylabel('Attitude Error (arcsec)')
    ax2.set_title('Focal Length vs Attitude Error\n(Test Correlation)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save test plot
    test_output_dir = Path(__file__).parent.parent.parent / "outputs" / "test_perturbation"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(test_output_dir / 'perturbation_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Test visualization saved to {test_output_dir}")

def main():
    """Run all perturbation analysis tests."""
    print("=" * 60)
    print("PERTURBATION ANALYSIS VALIDATION TESTS")
    print("=" * 60)
    print()
    
    try:
        # Run tests
        test_parameter_variation()
        test_focal_length_scenario()
        test_mock_simulation()
        create_test_visualization()
        
        print()
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("The perturbation analysis framework is ready for use!")
        print()
        print("To run the full focal length analysis:")
        print("PYTHONPATH=. python tools/specialized/focal_length_perturbation_analysis.py")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)