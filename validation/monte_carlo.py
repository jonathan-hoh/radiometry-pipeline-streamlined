#!/usr/bin/env python3
"""
validation/monte_carlo.py - Monte Carlo Framework for Validation

Implements Monte Carlo statistical validation framework with:
- Random attitude generation (uniform quaternion sampling)
- Structured test scenario generation  
- Parallel execution with checkpointing
- Results aggregation and statistical analysis

Supports interruption/resume capability for long-running validation campaigns.

Usage:
    from validation.monte_carlo import MonteCarloValidator
    
    validator = MonteCarloValidator(config_path="validation/config/validation_config.yaml")
    results = validator.run_validation_campaign(scenario_list, n_workers=8)
"""

import numpy as np
import multiprocessing as mp
import logging
import time
import pickle
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ValidationScenario:
    """Container for validation scenario definition."""
    scenario_id: str
    description: str
    parameters: Dict[str, Any]
    n_trials: int = 100
    timeout_seconds: int = 300
    priority: str = "medium"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationScenario':
        """Create from dictionary."""
        return cls(**data)

@dataclass 
class ValidationResult:
    """Container for individual validation result."""
    scenario_id: str
    trial_id: int
    success: bool
    execution_time: float
    results: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

@dataclass
class CampaignResults:
    """Container for complete validation campaign results."""
    campaign_id: str
    scenarios: List[ValidationScenario]
    results: List[ValidationResult]
    statistics: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    
class MonteCarloValidator:
    """
    Monte Carlo validation framework with parallel execution and checkpointing.
    
    Manages large-scale validation campaigns with automatic progress saving,
    resumption capability, and statistical analysis of results.
    """
    
    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo validator.
        
        Parameters
        ----------
        config_path : str or Path, optional
            Path to validation configuration YAML file
        checkpoint_dir : str or Path, optional
            Directory for checkpoint files (default: validation/checkpoints)
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.config = self._load_config(config_path)
        self.checkpoint_dir = Path(checkpoint_dir or "validation/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        if random_seed is not None:
            self.random_seed = random_seed
        else:
            self.random_seed = self.config.get('monte_carlo', {}).get('random_seed', 42)
            
        np.random.seed(self.random_seed)
        
        # Initialize campaign tracking
        self.current_campaign_id = None
        self.checkpoint_interval = self.config.get('monte_carlo', {}).get('checkpoint_interval', 100)
        
        logger.info(f"MonteCarloValidator initialized with seed={self.random_seed}")
        
    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default configuration
            return {
                'monte_carlo': {
                    'n_samples': 1000,
                    'random_seed': 42,
                    'parallel_workers': mp.cpu_count(),
                    'checkpoint_interval': 100
                },
                'validation_thresholds': {
                    'attitude_error_arcsec': 1.0,
                    'identification_rate_min': 0.95
                },
                'performance': {
                    'max_runtime_minutes': 30
                }
            }
            
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_random_attitudes(self, n_samples: int) -> np.ndarray:
        """
        Generate random attitudes with uniform distribution on SO(3).
        
        Uses Shoemake method for uniform quaternion sampling on 4-sphere.
        
        Parameters
        ----------
        n_samples : int
            Number of random attitudes to generate
            
        Returns
        -------
        np.ndarray
            Random quaternions, shape (n_samples, 4) as [w, x, y, z]
            
        Notes
        -----
        Implements Shoemake's uniform quaternion sampling:
        1. Generate 3 uniform random numbers u1, u2, u3 ∈ [0,1)
        2. Compute quaternion components using trigonometric functions
        3. Results in uniform distribution on SO(3) rotation group
        
        References
        ----------
        Shoemake, K. "Uniform random rotations." Graphics Gems III (1992): 124-132.
        """
        logger.info(f"Generating {n_samples} random attitudes using Shoemake method")
        
        # Generate random numbers for Shoemake algorithm
        u = np.random.uniform(0, 1, (n_samples, 3))
        
        # Shoemake uniform quaternion sampling
        sqrt_1_minus_u1 = np.sqrt(1 - u[:, 0])
        sqrt_u1 = np.sqrt(u[:, 0])
        theta1 = 2 * np.pi * u[:, 1]
        theta2 = 2 * np.pi * u[:, 2]
        
        quaternions = np.zeros((n_samples, 4))
        quaternions[:, 0] = sqrt_1_minus_u1 * np.cos(theta1)  # w
        quaternions[:, 1] = sqrt_1_minus_u1 * np.sin(theta1)  # x
        quaternions[:, 2] = sqrt_u1 * np.cos(theta2)          # y
        quaternions[:, 3] = sqrt_u1 * np.sin(theta2)          # z
        
        # Verify quaternions are normalized (should be by construction)
        norms = np.linalg.norm(quaternions, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-15):
            logger.warning("Random quaternions not perfectly normalized, fixing")
            quaternions = quaternions / norms[:, np.newaxis]
            
        return quaternions
    
    def generate_test_scenarios(
        self,
        n_samples: int,
        ra_range: Tuple[float, float] = (0, 360),
        dec_range: Tuple[float, float] = (-90, 90),
        roll_range: Tuple[float, float] = (0, 360),
        grid_spacing: Optional[float] = None
    ) -> List[ValidationScenario]:
        """
        Generate structured test scenarios with systematic attitude coverage.
        
        Parameters
        ----------
        n_samples : int
            Number of test scenarios to generate
        ra_range : tuple of float
            Right ascension range in degrees (min, max)
        dec_range : tuple of float  
            Declination range in degrees (min, max)
        roll_range : tuple of float
            Roll angle range in degrees (min, max)
        grid_spacing : float, optional
            If provided, use grid sampling instead of random
            
        Returns
        -------
        List[ValidationScenario]
            List of test scenarios with attitude parameters
        """
        logger.info(f"Generating {n_samples} test scenarios")
        
        scenarios = []
        
        if grid_spacing is not None:
            # Structured grid sampling
            ra_points = np.arange(ra_range[0], ra_range[1], grid_spacing)
            dec_points = np.arange(dec_range[0], dec_range[1], grid_spacing)
            roll_points = np.arange(roll_range[0], roll_range[1], grid_spacing)
            
            # Create all combinations (may exceed n_samples)
            count = 0
            for ra in ra_points:
                for dec in dec_points:
                    for roll in roll_points:
                        if count >= n_samples:
                            break
                            
                        scenario = ValidationScenario(
                            scenario_id=f"grid_{count:04d}",
                            description=f"Grid point RA={ra:.1f}, Dec={dec:.1f}, Roll={roll:.1f}",
                            parameters={
                                'attitude_ra_deg': float(ra),
                                'attitude_dec_deg': float(dec), 
                                'attitude_roll_deg': float(roll),
                                'sampling_method': 'grid'
                            },
                            tags=['grid_sampling', 'systematic']
                        )
                        scenarios.append(scenario)
                        count += 1
                        
                    if count >= n_samples:
                        break
                if count >= n_samples:
                    break
        else:
            # Random sampling within ranges
            for i in range(n_samples):
                ra = np.random.uniform(ra_range[0], ra_range[1])
                dec = np.random.uniform(dec_range[0], dec_range[1])
                roll = np.random.uniform(roll_range[0], roll_range[1])
                
                scenario = ValidationScenario(
                    scenario_id=f"random_{i:04d}",
                    description=f"Random attitude RA={ra:.1f}, Dec={dec:.1f}, Roll={roll:.1f}",
                    parameters={
                        'attitude_ra_deg': float(ra),
                        'attitude_dec_deg': float(dec),
                        'attitude_roll_deg': float(roll),
                        'sampling_method': 'random'
                    },
                    tags=['random_sampling', 'monte_carlo']
                )
                scenarios.append(scenario)
                
        logger.info(f"Generated {len(scenarios)} test scenarios")
        return scenarios
    
    def run_parallel(
        self,
        scenario_list: List[ValidationScenario],
        validation_function: Callable[[ValidationScenario], ValidationResult],
        n_workers: Optional[int] = None
    ) -> List[ValidationResult]:
        """
        Execute validation scenarios in parallel with progress tracking.
        
        Parameters
        ----------
        scenario_list : List[ValidationScenario]
            Scenarios to execute
        validation_function : Callable
            Function that takes ValidationScenario and returns ValidationResult
        n_workers : int, optional
            Number of parallel workers (default: from config)
            
        Returns
        -------
        List[ValidationResult]
            Results from all scenarios
        """
        if n_workers is None:
            n_workers = self.config.get('monte_carlo', {}).get('parallel_workers', mp.cpu_count())
            
        logger.info(f"Running {len(scenario_list)} scenarios with {n_workers} workers")
        
        results = []
        completed_count = 0
        start_time = time.time()
        
        # Check for existing checkpoint
        checkpoint_file = self._get_checkpoint_file()
        if checkpoint_file.exists():
            logger.info(f"Found checkpoint file: {checkpoint_file}")
            results, completed_count = self._load_checkpoint(checkpoint_file)
            logger.info(f"Resuming from checkpoint with {completed_count} completed scenarios")
        
        # Skip already completed scenarios
        remaining_scenarios = scenario_list[completed_count:]
        
        if not remaining_scenarios:
            logger.info("All scenarios already completed")
            return results
            
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all remaining scenarios
            future_to_scenario = {
                executor.submit(validation_function, scenario): scenario 
                for scenario in remaining_scenarios
            }
            
            try:
                for future in as_completed(future_to_scenario):
                    scenario = future_to_scenario[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        completed_count += 1
                        
                        # Progress logging
                        elapsed = time.time() - start_time
                        progress = completed_count / len(scenario_list)
                        eta = elapsed / progress - elapsed if progress > 0 else 0
                        
                        if completed_count % 10 == 0 or completed_count == len(scenario_list):
                            logger.info(
                                f"Progress: {completed_count}/{len(scenario_list)} "
                                f"({progress*100:.1f}%) - ETA: {eta:.1f}s"
                            )
                        
                        # Checkpoint saving
                        if completed_count % self.checkpoint_interval == 0:
                            self._save_checkpoint(results, completed_count)
                            
                    except Exception as e:
                        logger.error(f"Scenario {scenario.scenario_id} failed: {e}")
                        # Create failed result
                        failed_result = ValidationResult(
                            scenario_id=scenario.scenario_id,
                            trial_id=0,
                            success=False,
                            execution_time=0.0,
                            results={},
                            error_message=str(e)
                        )
                        results.append(failed_result)
                        completed_count += 1
                        
            except KeyboardInterrupt:
                logger.info("Interrupted by user, saving checkpoint...")
                self._save_checkpoint(results, completed_count)
                raise
                
        # Final checkpoint and cleanup
        self._save_checkpoint(results, completed_count)
        total_time = time.time() - start_time
        logger.info(f"Completed {len(scenario_list)} scenarios in {total_time:.1f}s")
        
        return results
    
    def aggregate_results(self, result_list: List[ValidationResult]) -> Dict[str, Any]:
        """
        Aggregate validation results into statistical summary.
        
        Parameters
        ----------
        result_list : List[ValidationResult]
            Individual validation results
            
        Returns
        -------
        Dict[str, Any]
            Statistical summary including success rates, performance metrics,
            error distributions, and confidence intervals
        """
        logger.info(f"Aggregating results from {len(result_list)} trials")
        
        if not result_list:
            return {'error': 'No results to aggregate'}
            
        # Basic statistics
        total_trials = len(result_list)
        successful_trials = sum(1 for r in result_list if r.success)
        success_rate = successful_trials / total_trials
        
        # Execution time statistics
        execution_times = [r.execution_time for r in result_list if r.success]
        
        # Extract metric values (assumes consistent result structure)
        metric_values = {}
        for result in result_list:
            if result.success and result.results:
                for metric_name, value in result.results.items():
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    if isinstance(value, (int, float, np.number)):
                        metric_values[metric_name].append(float(value))
        
        # Calculate statistics for each metric
        metric_statistics = {}
        for metric_name, values in metric_values.items():
            if values:
                values_array = np.array(values)
                metric_statistics[metric_name] = {
                    'count': len(values),
                    'mean': float(np.mean(values_array)),
                    'median': float(np.median(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'percentile_5': float(np.percentile(values_array, 5)),
                    'percentile_95': float(np.percentile(values_array, 95)),
                    'percentile_99': float(np.percentile(values_array, 99))
                }
        
        # Error analysis
        failed_results = [r for r in result_list if not r.success]
        error_summary = {}
        if failed_results:
            error_messages = [r.error_message for r in failed_results if r.error_message]
            error_summary = {
                'total_failures': len(failed_results),
                'failure_rate': len(failed_results) / total_trials,
                'common_errors': self._analyze_error_patterns(error_messages)
            }
        
        # Performance analysis
        performance_summary = {}
        if execution_times:
            execution_array = np.array(execution_times)
            performance_summary = {
                'mean_execution_time': float(np.mean(execution_array)),
                'median_execution_time': float(np.median(execution_array)),
                'total_execution_time': float(np.sum(execution_array))
            }
        
        # Validation threshold checks
        threshold_analysis = self._check_validation_thresholds(metric_statistics)
        
        aggregated_results = {
            'summary': {
                'total_trials': total_trials,
                'successful_trials': successful_trials,
                'success_rate': success_rate,
                'timestamp': datetime.utcnow().isoformat()
            },
            'metrics': metric_statistics,
            'errors': error_summary,
            'performance': performance_summary,
            'threshold_analysis': threshold_analysis,
            'configuration': {
                'random_seed': self.random_seed,
                'checkpoint_interval': self.checkpoint_interval
            }
        }
        
        return aggregated_results
    
    def _check_validation_thresholds(self, metric_statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Check metrics against validation thresholds."""
        thresholds = self.config.get('validation_thresholds', {})
        threshold_results = {}
        
        for threshold_name, threshold_value in thresholds.items():
            # Map threshold names to metric names
            metric_mapping = {
                'attitude_error_arcsec': 'attitude_error_angle',
                'identification_rate_min': 'identification_rate',
                'astrometric_rms_pixels': 'astrometric_rms',
                'centroiding_accuracy_pixels': 'centroid_rms'
            }
            
            metric_name = metric_mapping.get(threshold_name)
            if metric_name in metric_statistics:
                metric_stats = metric_statistics[metric_name]
                
                if 'min' in threshold_name:
                    # Minimum threshold (e.g., identification rate)
                    passed = metric_stats['mean'] >= threshold_value
                    threshold_results[threshold_name] = {
                        'threshold': threshold_value,
                        'actual_mean': metric_stats['mean'],
                        'passed': passed,
                        'margin': metric_stats['mean'] - threshold_value
                    }
                else:
                    # Maximum threshold (e.g., error limits)
                    passed = metric_stats['mean'] <= threshold_value
                    threshold_results[threshold_name] = {
                        'threshold': threshold_value,
                        'actual_mean': metric_stats['mean'],
                        'passed': passed,
                        'margin': threshold_value - metric_stats['mean']
                    }
        
        return threshold_results
    
    def _analyze_error_patterns(self, error_messages: List[str]) -> Dict[str, int]:
        """Analyze common error patterns."""
        error_patterns = {}
        for message in error_messages:
            if message:
                # Simple pattern matching - can be enhanced
                if 'timeout' in message.lower():
                    error_patterns['timeout'] = error_patterns.get('timeout', 0) + 1
                elif 'memory' in message.lower():
                    error_patterns['memory'] = error_patterns.get('memory', 0) + 1
                elif 'convergence' in message.lower():
                    error_patterns['convergence'] = error_patterns.get('convergence', 0) + 1
                else:
                    error_patterns['other'] = error_patterns.get('other', 0) + 1
        return error_patterns
    
    def _get_checkpoint_file(self) -> Path:
        """Get checkpoint file path for current campaign."""
        if self.current_campaign_id is None:
            self.current_campaign_id = f"campaign_{int(time.time())}"
        return self.checkpoint_dir / f"{self.current_campaign_id}_checkpoint.pkl"
    
    def _save_checkpoint(self, results: List[ValidationResult], completed_count: int):
        """Save current progress to checkpoint file."""
        checkpoint_file = self._get_checkpoint_file()
        checkpoint_data = {
            'results': results,
            'completed_count': completed_count,
            'timestamp': datetime.utcnow().isoformat(),
            'random_seed': self.random_seed
        }
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.debug(f"Saved checkpoint: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, checkpoint_file: Path) -> Tuple[List[ValidationResult], int]:
        """Load progress from checkpoint file."""
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            results = checkpoint_data['results']
            completed_count = checkpoint_data['completed_count']
            
            # Verify random seed consistency
            saved_seed = checkpoint_data.get('random_seed')
            if saved_seed != self.random_seed:
                logger.warning(
                    f"Random seed mismatch: current={self.random_seed}, "
                    f"checkpoint={saved_seed}"
                )
            
            return results, completed_count
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return [], 0
    
    def cleanup_checkpoints(self, keep_latest: int = 3):
        """Clean up old checkpoint files."""
        checkpoint_files = list(self.checkpoint_dir.glob("*_checkpoint.pkl"))
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Remove old checkpoint files
        for checkpoint_file in checkpoint_files[keep_latest:]:
            try:
                checkpoint_file.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {checkpoint_file}: {e}")

# Utility functions for quaternion/attitude generation
def euler_to_quaternion(ra_deg: float, dec_deg: float, roll_deg: float) -> np.ndarray:
    """
    Convert Euler angles (RA, Dec, Roll) to quaternion.
    
    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees  
    roll_deg : float
        Roll angle in degrees
        
    Returns
    -------
    np.ndarray
        Quaternion [w, x, y, z]
    """
    # Convert to radians
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg) 
    roll = np.radians(roll_deg)
    
    # Euler angle to quaternion conversion (ZYX order)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cd = np.cos(dec * 0.5)
    sd = np.sin(dec * 0.5)
    ca = np.cos(ra * 0.5)
    sa = np.sin(ra * 0.5)
    
    w = cr * cd * ca + sr * sd * sa
    x = sr * cd * ca - cr * sd * sa
    y = cr * sd * ca + sr * cd * sa
    z = cr * cd * sa - sr * sd * ca
    
    return np.array([w, x, y, z])

# Export public classes and functions
__all__ = [
    'MonteCarloValidator',
    'ValidationScenario', 
    'ValidationResult',
    'CampaignResults',
    'euler_to_quaternion'
]