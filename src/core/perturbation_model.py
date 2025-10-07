#!/usr/bin/env python3
"""
Perturbation Model Module
Handles parameter variations and uncertainty propagation for Monte Carlo analysis.

This module provides a framework for systematically studying how small changes in
system parameters (e.g., focal length, sensor characteristics) propagate through
the star tracker pipeline to affect final attitude accuracy.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributionType(Enum):
    """Supported probability distributions for parameter variations."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    EXPONENTIAL = "exponential"
    CONSTANT = "constant"

@dataclass
class ParameterVariation:
    """Defines how a single parameter varies during Monte Carlo analysis."""
    
    name: str
    nominal_value: float
    distribution: DistributionType
    parameters: Dict[str, float]
    units: str = ""
    description: str = ""
    
    def __post_init__(self):
        """Validate parameter configuration."""
        if self.distribution == DistributionType.NORMAL:
            required_params = ['std', 'mean']
            # If mean not specified, use nominal_value
            if 'mean' not in self.parameters:
                self.parameters['mean'] = self.nominal_value
        elif self.distribution == DistributionType.UNIFORM:
            required_params = ['min', 'max']
        elif self.distribution == DistributionType.TRIANGULAR:
            required_params = ['min', 'max', 'mode']
        elif self.distribution == DistributionType.EXPONENTIAL:
            required_params = ['scale']
        elif self.distribution == DistributionType.CONSTANT:
            required_params = []
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution}")
        
        # Check required parameters are present
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter '{param}' for {self.distribution.value} distribution")
    
    def sample(self, size: int = 1) -> Union[float, np.ndarray]:
        """Generate random samples from the parameter distribution."""
        if self.distribution == DistributionType.NORMAL:
            return np.random.normal(
                self.parameters['mean'], 
                self.parameters['std'], 
                size
            )
        elif self.distribution == DistributionType.UNIFORM:
            return np.random.uniform(
                self.parameters['min'], 
                self.parameters['max'], 
                size
            )
        elif self.distribution == DistributionType.TRIANGULAR:
            return np.random.triangular(
                self.parameters['min'],
                self.parameters['mode'], 
                self.parameters['max'], 
                size
            )
        elif self.distribution == DistributionType.EXPONENTIAL:
            return np.random.exponential(self.parameters['scale'], size)
        elif self.distribution == DistributionType.CONSTANT:
            return np.full(size, self.nominal_value)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get theoretical statistics for the distribution."""
        stats = {'nominal': self.nominal_value}
        
        if self.distribution == DistributionType.NORMAL:
            stats['mean'] = self.parameters['mean']
            stats['std'] = self.parameters['std']
            stats['variance'] = self.parameters['std'] ** 2
        elif self.distribution == DistributionType.UNIFORM:
            a, b = self.parameters['min'], self.parameters['max']
            stats['mean'] = (a + b) / 2
            stats['std'] = np.sqrt((b - a) ** 2 / 12)
            stats['variance'] = (b - a) ** 2 / 12
        elif self.distribution == DistributionType.TRIANGULAR:
            a, c, b = self.parameters['min'], self.parameters['mode'], self.parameters['max']
            stats['mean'] = (a + c + b) / 3
            stats['variance'] = (a**2 + b**2 + c**2 - a*b - a*c - b*c) / 18
            stats['std'] = np.sqrt(stats['variance'])
        elif self.distribution == DistributionType.CONSTANT:
            stats['mean'] = self.nominal_value
            stats['std'] = 0.0
            stats['variance'] = 0.0
        
        return stats

@dataclass
class PerturbationScenario:
    """Defines a complete perturbation scenario with multiple parameter variations."""
    
    name: str
    description: str
    parameters: List[ParameterVariation] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_parameter(self, param: ParameterVariation):
        """Add a parameter variation to the scenario."""
        self.parameters.append(param)
        logger.info(f"Added parameter variation: {param.name} ({param.distribution.value})")
    
    def sample_all(self, size: int = 1) -> Dict[str, Union[float, np.ndarray]]:
        """Generate samples for all parameters in the scenario."""
        samples = {}
        for param in self.parameters:
            samples[param.name] = param.sample(size)
            if size == 1:
                samples[param.name] = float(samples[param.name])
        return samples
    
    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names."""
        return [param.name for param in self.parameters]
    
    def get_nominal_values(self) -> Dict[str, float]:
        """Get nominal values for all parameters."""
        return {param.name: param.nominal_value for param in self.parameters}
    
    def to_dict(self) -> Dict:
        """Export scenario to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': [
                {
                    'name': p.name,
                    'nominal_value': p.nominal_value,
                    'distribution': p.distribution.value,
                    'parameters': p.parameters,
                    'units': p.units,
                    'description': p.description
                }
                for p in self.parameters
            ],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PerturbationScenario':
        """Create scenario from dictionary."""
        scenario = cls(data['name'], data['description'])
        scenario.metadata = data.get('metadata', {})
        
        for param_data in data['parameters']:
            param = ParameterVariation(
                name=param_data['name'],
                nominal_value=param_data['nominal_value'],
                distribution=DistributionType(param_data['distribution']),
                parameters=param_data['parameters'],
                units=param_data.get('units', ''),
                description=param_data.get('description', '')
            )
            scenario.add_parameter(param)
        
        return scenario

class PerturbationAnalyzer:
    """Manages and analyzes Monte Carlo perturbation studies."""
    
    def __init__(self, scenario: PerturbationScenario):
        """Initialize analyzer with a perturbation scenario."""
        self.scenario = scenario
        self.results = None
        self.trial_data = None
        
        logger.info(f"Initialized perturbation analyzer for scenario: {scenario.name}")
        logger.info(f"Parameters to vary: {scenario.get_parameter_names()}")
    
    def run_monte_carlo(
        self, 
        simulation_function: Callable, 
        n_trials: int = 1000,
        **sim_kwargs
    ) -> pd.DataFrame:
        """
        Run Monte Carlo analysis with parameter variations.
        
        Args:
            simulation_function: Function that takes parameter dict and returns results
            n_trials: Number of Monte Carlo trials
            **sim_kwargs: Additional arguments to pass to simulation function
            
        Returns:
            DataFrame with trial results including parameter values and outputs
        """
        logger.info(f"Starting Monte Carlo analysis: {n_trials} trials")
        
        # Initialize results storage
        trial_results = []
        
        for trial in range(n_trials):
            if trial % (n_trials // 10) == 0:
                logger.info(f"Progress: {trial}/{n_trials} trials ({100*trial/n_trials:.0f}%)")
            
            # Sample parameter values for this trial
            param_values = self.scenario.sample_all(size=1)
            
            try:
                # Run simulation with perturbed parameters
                sim_results = simulation_function(param_values, **sim_kwargs)
                
                # Store trial data
                trial_data = {
                    'trial': trial,
                    **param_values,  # Parameter values
                    **sim_results    # Simulation results
                }
                trial_results.append(trial_data)
                
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                # Store failed trial with NaN results
                trial_data = {
                    'trial': trial,
                    **param_values,
                    'simulation_success': False,
                    'error_message': str(e)
                }
                trial_results.append(trial_data)
        
        # Convert to DataFrame
        self.trial_data = pd.DataFrame(trial_results)
        logger.info(f"Monte Carlo analysis completed: {len(self.trial_data)} trials")
        
        return self.trial_data
    
    def analyze_sensitivity(self, output_parameter: str) -> Dict[str, float]:
        """
        Analyze sensitivity of output to input parameter variations.
        
        Args:
            output_parameter: Name of the output parameter to analyze
            
        Returns:
            Dictionary of sensitivity coefficients
        """
        if self.trial_data is None:
            raise ValueError("No trial data available. Run Monte Carlo analysis first.")
        
        # Filter out failed trials
        valid_trials = self.trial_data.dropna(subset=[output_parameter])
        
        if len(valid_trials) == 0:
            raise ValueError(f"No valid trials with output parameter: {output_parameter}")
        
        sensitivity_coefficients = {}
        
        # Calculate sensitivity for each input parameter
        for param_name in self.scenario.get_parameter_names():
            if param_name in valid_trials.columns:
                # Calculate correlation and linear sensitivity
                param_values = valid_trials[param_name].values
                output_values = valid_trials[output_parameter].values
                
                # Linear correlation coefficient
                correlation = np.corrcoef(param_values, output_values)[0, 1]
                
                # Linear regression slope (sensitivity coefficient)
                if np.std(param_values) > 0:
                    sensitivity = np.cov(param_values, output_values)[0, 1] / np.var(param_values)
                else:
                    sensitivity = 0.0
                
                # Normalized sensitivity (% change in output per % change in input)
                nominal_param = self.scenario.get_nominal_values()[param_name]
                nominal_output = np.mean(output_values)
                
                if nominal_param != 0 and nominal_output != 0:
                    normalized_sensitivity = (sensitivity * nominal_param) / nominal_output
                else:
                    normalized_sensitivity = 0.0
                
                sensitivity_coefficients[param_name] = {
                    'correlation': correlation,
                    'sensitivity': sensitivity,
                    'normalized_sensitivity': normalized_sensitivity,
                    'param_std': np.std(param_values),
                    'output_std': np.std(output_values)
                }
        
        return sensitivity_coefficients
    
    def get_summary_statistics(self, parameter: str) -> Dict[str, float]:
        """Get summary statistics for a parameter across all trials."""
        if self.trial_data is None:
            raise ValueError("No trial data available. Run Monte Carlo analysis first.")
        
        if parameter not in self.trial_data.columns:
            raise ValueError(f"Parameter '{parameter}' not found in trial data")
        
        data = self.trial_data[parameter].dropna()
        
        return {
            'count': len(data),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'q95': float(np.percentile(data, 95)),
            'q99': float(np.percentile(data, 99))
        }
    
    def export_results(self, filepath: str):
        """Export trial results to CSV file."""
        if self.trial_data is None:
            raise ValueError("No trial data available to export")
        
        self.trial_data.to_csv(filepath, index=False)
        logger.info(f"Results exported to: {filepath}")
    
    def export_scenario(self, filepath: str):
        """Export scenario definition to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.scenario.to_dict(), f, indent=2)
        logger.info(f"Scenario exported to: {filepath}")

def create_focal_length_scenario(
    nominal_focal_length: float = 25.0,
    thermal_variation: float = 0.5,
    temperature_range: Tuple[float, float] = (-20, 60)
) -> PerturbationScenario:
    """
    Create a focal length perturbation scenario based on thermal effects.
    
    Args:
        nominal_focal_length: Nominal focal length in mm
        thermal_variation: Maximum focal length change in mm over temperature range
        temperature_range: Operating temperature range in Celsius
        
    Returns:
        Configured perturbation scenario
    """
    # Calculate coefficient of thermal expansion effect
    # Typical aluminum CTE ≈ 23 ppm/°C
    # For a 25mm focal length over 80°C range: ΔL ≈ 25 * 23e-6 * 80 ≈ 0.046mm
    # But optical systems can have additional effects, so use provided variation
    
    temp_span = temperature_range[1] - temperature_range[0]
    
    scenario = PerturbationScenario(
        name="Focal Length Thermal Variation",
        description=f"Focal length variations due to thermal expansion over {temperature_range[0]}°C to {temperature_range[1]}°C",
        metadata={
            'nominal_focal_length_mm': nominal_focal_length,
            'thermal_variation_mm': thermal_variation,
            'temperature_range_C': temperature_range,
            'assumed_mechanism': 'thermal_expansion'
        }
    )
    
    # Define focal length variation as normal distribution
    # Use 3-sigma = thermal_variation (99.7% of values within range)
    focal_length_std = thermal_variation / 3.0
    
    focal_length_param = ParameterVariation(
        name="focal_length",
        nominal_value=nominal_focal_length,
        distribution=DistributionType.NORMAL,
        parameters={
            'mean': nominal_focal_length,
            'std': focal_length_std
        },
        units="mm",
        description=f"Focal length with thermal variation (±{thermal_variation}mm, 3σ)"
    )
    
    scenario.add_parameter(focal_length_param)
    
    logger.info(f"Created focal length scenario:")
    logger.info(f"  Nominal: {nominal_focal_length} mm")
    logger.info(f"  Variation: ±{thermal_variation} mm (3σ)")
    logger.info(f"  Standard deviation: {focal_length_std:.3f} mm")
    
    return scenario

def create_comprehensive_scenario(
    focal_length_variation: float = 0.5,
    pixel_pitch_variation: float = 0.1,
    read_noise_variation: float = 2.0
) -> PerturbationScenario:
    """
    Create a comprehensive perturbation scenario with multiple parameters.
    (For future expansion beyond just focal length)
    
    Args:
        focal_length_variation: Focal length variation in mm (3σ)
        pixel_pitch_variation: Pixel pitch variation in µm (3σ)
        read_noise_variation: Read noise variation in electrons (3σ)
        
    Returns:
        Multi-parameter perturbation scenario
    """
    scenario = PerturbationScenario(
        name="Comprehensive System Variation",
        description="Multiple parameter variations including optical and sensor effects",
        metadata={
            'analysis_type': 'comprehensive',
            'parameters_varied': ['focal_length', 'pixel_pitch', 'read_noise']
        }
    )
    
    # Focal length variation
    focal_length_param = ParameterVariation(
        name="focal_length",
        nominal_value=25.0,
        distribution=DistributionType.NORMAL,
        parameters={'mean': 25.0, 'std': focal_length_variation / 3.0},
        units="mm",
        description="Focal length with thermal variation"
    )
    scenario.add_parameter(focal_length_param)
    
    # Pixel pitch variation (manufacturing tolerance)
    pixel_pitch_param = ParameterVariation(
        name="pixel_pitch",
        nominal_value=5.5,
        distribution=DistributionType.NORMAL,
        parameters={'mean': 5.5, 'std': pixel_pitch_variation / 3.0},
        units="µm",
        description="Pixel pitch manufacturing variation"
    )
    scenario.add_parameter(pixel_pitch_param)
    
    # Read noise variation (temperature effects)
    read_noise_param = ParameterVariation(
        name="read_noise",
        nominal_value=13.0,
        distribution=DistributionType.NORMAL,
        parameters={'mean': 13.0, 'std': read_noise_variation / 3.0},
        units="e⁻",
        description="Read noise temperature variation"
    )
    scenario.add_parameter(read_noise_param)
    
    return scenario