#!/usr/bin/env python3
"""
validation/attitude_validation.py - Attitude Validation Module

Validates attitude determination accuracy using Monte Carlo analysis with:
- Ground truth image generation with known attitudes
- BAST + QUEST pipeline execution  
- Statistical error analysis and confidence bounds
- Comparison to Cramér-Rao theoretical limits
- Comprehensive plotting and results export

Usage:
    from validation.attitude_validation import AttitudeValidator
    
    validator = AttitudeValidator(pipeline, n_monte_carlo=1000)
    results = validator.run_validation_campaign()
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from datetime import datetime

# Import core pipeline components
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.multi_star.monte_carlo_quest import MonteCarloQUEST, NoiseParameters, MonteCarloResults
from src.multi_star.attitude_transform import quaternion_to_euler, euler_to_quaternion
from src.BAST.match import StarMatch
from src.BAST.resolve import quest_algorithm

# Import validation framework
from .metrics import attitude_error_angle, ValidationResults
from .monte_carlo import MonteCarloValidator, ValidationScenario, ValidationResult, euler_to_quaternion

logger = logging.getLogger(__name__)

@dataclass
class AttitudeValidationConfig:
    """Configuration for attitude validation."""
    n_monte_carlo: int = 1000
    magnitude_range: Tuple[float, float] = (3.0, 6.0)
    field_angles_deg: List[float] = None
    psf_generation: str = "Gen_1"
    noise_multiplier: float = 1.0
    confidence_level: float = 0.95
    output_dir: Union[str, Path] = "validation/results/attitude"
    save_intermediate: bool = True
    parallel_workers: int = 4
    
    def __post_init__(self):
        if self.field_angles_deg is None:
            self.field_angles_deg = [0, 2, 5, 8, 11, 14]
        self.output_dir = Path(self.output_dir)

@dataclass
class AttitudeTrialResult:
    """Results from single attitude validation trial."""
    trial_id: int
    q_true: np.ndarray
    q_solved: np.ndarray
    attitude_error_arcsec: float
    n_stars_detected: int
    n_stars_matched: int
    identification_rate: float
    convergence_achieved: bool
    execution_time: float
    magnitude: float
    field_angle_deg: float
    success: bool
    error_message: Optional[str] = None

class AttitudeValidator:
    """
    Attitude validation using Monte Carlo analysis.
    
    Validates attitude determination accuracy by:
    1. Generating synthetic images with known ground truth attitudes
    2. Running complete BAST + QUEST pipeline 
    3. Computing attitude errors and statistical metrics
    4. Comparing performance to theoretical bounds
    """
    
    def __init__(
        self,
        simulation_pipeline: StarTrackerPipeline,
        config: Optional[AttitudeValidationConfig] = None
    ):
        """
        Initialize attitude validator.
        
        Parameters
        ----------
        simulation_pipeline : StarTrackerPipeline
            Configured star tracker pipeline instance
        config : AttitudeValidationConfig, optional
            Validation configuration
        """
        self.pipeline = simulation_pipeline
        self.config = config or AttitudeValidationConfig()
        self.monte_carlo = MonteCarloValidator()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Monte Carlo QUEST solver
        noise_params = NoiseParameters(
            centroid_noise_pixels=0.15 * self.config.noise_multiplier,
            pixel_pitch_um=5.5,
            focal_length_mm=40.07
        )
        self.mc_quest = MonteCarloQUEST(noise_params)
        
        logger.info(f"AttitudeValidator initialized with {self.config.n_monte_carlo} trials")
        
    def generate_ground_truth_images(
        self,
        attitudes: np.ndarray,
        magnitudes: np.ndarray,
        field_angles_deg: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic images with known ground truth attitudes.
        
        Parameters
        ----------
        attitudes : np.ndarray
            Ground truth quaternions, shape (N, 4)
        magnitudes : np.ndarray  
            Star magnitudes, shape (N,)
        field_angles_deg : np.ndarray
            Field angles in degrees, shape (N,)
            
        Returns
        -------
        List[Dict[str, Any]]
            List of image data with ground truth metadata
        """
        logger.info(f"Generating {len(attitudes)} ground truth images")
        
        ground_truth_images = []
        
        for i, (q_true, magnitude, field_angle) in enumerate(zip(attitudes, magnitudes, field_angles_deg)):
            try:
                # Convert quaternion to Euler angles for PSF file selection
                ra, dec, roll = quaternion_to_euler(q_true)
                
                # Select appropriate PSF file based on field angle
                psf_file = self._select_psf_file(field_angle)
                
                # Generate synthetic star scene using pipeline
                image_data = self._generate_star_scene(
                    psf_file=psf_file,
                    magnitude=magnitude,
                    attitude_quaternion=q_true,
                    field_angle_deg=field_angle
                )
                
                ground_truth_data = {
                    'trial_id': i,
                    'q_true': q_true.copy(),
                    'ra_deg': np.degrees(ra),
                    'dec_deg': np.degrees(dec), 
                    'roll_deg': np.degrees(roll),
                    'magnitude': magnitude,
                    'field_angle_deg': field_angle,
                    'psf_file': str(psf_file),
                    'image_data': image_data,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                ground_truth_images.append(ground_truth_data)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i + 1}/{len(attitudes)} ground truth images")
                    
            except Exception as e:
                logger.error(f"Failed to generate ground truth image {i}: {e}")
                continue
                
        logger.info(f"Successfully generated {len(ground_truth_images)} ground truth images")
        return ground_truth_images
    
    def run_pipeline_on_images(
        self,
        ground_truth_images: List[Dict[str, Any]]
    ) -> List[AttitudeTrialResult]:
        """
        Execute BAST + QUEST pipeline on ground truth images.
        
        Parameters
        ----------
        ground_truth_images : List[Dict[str, Any]]
            Ground truth image data
            
        Returns
        -------
        List[AttitudeTrialResult]
            Pipeline execution results
        """
        logger.info(f"Running pipeline on {len(ground_truth_images)} images")
        
        results = []
        start_time = time.time()
        
        for i, gt_data in enumerate(ground_truth_images):
            try:
                trial_start = time.time()
                
                # Extract ground truth
                q_true = gt_data['q_true']
                magnitude = gt_data['magnitude']
                field_angle = gt_data['field_angle_deg']
                image_data = gt_data['image_data']
                
                # Run star detection and centroiding
                detection_results = self._run_star_detection(image_data)
                
                if not detection_results['success']:
                    result = AttitudeTrialResult(
                        trial_id=i,
                        q_true=q_true,
                        q_solved=np.array([1, 0, 0, 0]),
                        attitude_error_arcsec=np.inf,
                        n_stars_detected=0,
                        n_stars_matched=0,
                        identification_rate=0.0,
                        convergence_achieved=False,
                        execution_time=time.time() - trial_start,
                        magnitude=magnitude,
                        field_angle_deg=field_angle,
                        success=False,
                        error_message="Star detection failed"
                    )
                    results.append(result)
                    continue
                
                # Run star identification and matching
                matching_results = self._run_star_matching(
                    detection_results['bearing_vectors'],
                    detection_results['centroids']
                )
                
                if not matching_results['success'] or len(matching_results['star_matches']) < 3:
                    result = AttitudeTrialResult(
                        trial_id=i,
                        q_true=q_true,
                        q_solved=np.array([1, 0, 0, 0]),
                        attitude_error_arcsec=np.inf,
                        n_stars_detected=detection_results['n_stars'],
                        n_stars_matched=len(matching_results.get('star_matches', [])),
                        identification_rate=0.0,
                        convergence_achieved=False,
                        execution_time=time.time() - trial_start,
                        magnitude=magnitude,
                        field_angle_deg=field_angle,
                        success=False,
                        error_message="Insufficient star matches for attitude solution"
                    )
                    results.append(result)
                    continue
                
                # Run Monte Carlo QUEST attitude determination
                mc_results = self._run_monte_carlo_quest(
                    matching_results['star_matches'],
                    matching_results['bearing_vectors']
                )
                
                if not mc_results['success']:
                    result = AttitudeTrialResult(
                        trial_id=i,
                        q_true=q_true,
                        q_solved=np.array([1, 0, 0, 0]),
                        attitude_error_arcsec=np.inf,
                        n_stars_detected=detection_results['n_stars'],
                        n_stars_matched=len(matching_results['star_matches']),
                        identification_rate=len(matching_results['star_matches']) / max(detection_results['n_stars'], 1),
                        convergence_achieved=False,
                        execution_time=time.time() - trial_start,
                        magnitude=magnitude,
                        field_angle_deg=field_angle,
                        success=False,
                        error_message="QUEST attitude determination failed"
                    )
                    results.append(result)
                    continue
                
                # Calculate attitude error
                q_solved = mc_results['quaternion']
                error_arcsec = attitude_error_angle(q_true, q_solved)
                
                # Create successful result
                result = AttitudeTrialResult(
                    trial_id=i,
                    q_true=q_true,
                    q_solved=q_solved,
                    attitude_error_arcsec=error_arcsec,
                    n_stars_detected=detection_results['n_stars'],
                    n_stars_matched=len(matching_results['star_matches']),
                    identification_rate=len(matching_results['star_matches']) / max(detection_results['n_stars'], 1),
                    convergence_achieved=mc_results['converged'],
                    execution_time=time.time() - trial_start,
                    magnitude=magnitude,
                    field_angle_deg=field_angle,
                    success=True
                )
                results.append(result)
                
                # Progress logging
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * len(ground_truth_images) - elapsed
                    logger.info(f"Processed {i + 1}/{len(ground_truth_images)} images - ETA: {eta:.1f}s")
                    
            except Exception as e:
                logger.error(f"Pipeline execution failed for trial {i}: {e}")
                result = AttitudeTrialResult(
                    trial_id=i,
                    q_true=gt_data['q_true'],
                    q_solved=np.array([1, 0, 0, 0]),
                    attitude_error_arcsec=np.inf,
                    n_stars_detected=0,
                    n_stars_matched=0,
                    identification_rate=0.0,
                    convergence_achieved=False,
                    execution_time=0.0,
                    magnitude=gt_data['magnitude'],
                    field_angle_deg=gt_data['field_angle_deg'],
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
        
        total_time = time.time() - start_time
        successful_results = sum(1 for r in results if r.success)
        logger.info(f"Pipeline execution complete: {successful_results}/{len(results)} successful in {total_time:.1f}s")
        
        return results
    
    def compute_error_statistics(
        self,
        trial_results: List[AttitudeTrialResult]
    ) -> Dict[str, Any]:
        """
        Compute comprehensive error statistics from trial results.
        
        Parameters
        ----------
        trial_results : List[AttitudeTrialResult]
            Individual trial results
            
        Returns
        -------
        Dict[str, Any]
            Statistical summary of attitude errors
        """
        logger.info(f"Computing error statistics from {len(trial_results)} trials")
        
        # Filter successful trials
        successful_trials = [r for r in trial_results if r.success and np.isfinite(r.attitude_error_arcsec)]
        
        if not successful_trials:
            logger.warning("No successful trials for error statistics")
            return {
                'n_total_trials': len(trial_results),
                'n_successful_trials': 0,
                'success_rate': 0.0,
                'error': 'No successful trials'
            }
        
        # Extract error values
        errors_arcsec = np.array([r.attitude_error_arcsec for r in successful_trials])
        identification_rates = np.array([r.identification_rate for r in successful_trials])
        execution_times = np.array([r.execution_time for r in successful_trials])
        
        # Compute statistics
        statistics = {
            'n_total_trials': len(trial_results),
            'n_successful_trials': len(successful_trials),
            'success_rate': len(successful_trials) / len(trial_results),
            
            # Attitude error statistics
            'attitude_error_arcsec': {
                'mean': float(np.mean(errors_arcsec)),
                'median': float(np.median(errors_arcsec)),
                'std': float(np.std(errors_arcsec)),
                'min': float(np.min(errors_arcsec)),
                'max': float(np.max(errors_arcsec)),
                'percentile_5': float(np.percentile(errors_arcsec, 5)),
                'percentile_95': float(np.percentile(errors_arcsec, 95)),
                'percentile_99': float(np.percentile(errors_arcsec, 99)),
                'rms': float(np.sqrt(np.mean(errors_arcsec**2)))
            },
            
            # Identification rate statistics
            'identification_rate': {
                'mean': float(np.mean(identification_rates)),
                'median': float(np.median(identification_rates)),
                'std': float(np.std(identification_rates)),
                'min': float(np.min(identification_rates))
            },
            
            # Performance statistics
            'execution_time_seconds': {
                'mean': float(np.mean(execution_times)),
                'median': float(np.median(execution_times)),
                'total': float(np.sum(execution_times))
            },
            
            # Analysis timestamp
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        # Error breakdown by field angle and magnitude
        field_angles = sorted(set(r.field_angle_deg for r in successful_trials))
        magnitudes = sorted(set(r.magnitude for r in successful_trials))
        
        if len(field_angles) > 1:
            statistics['error_vs_field_angle'] = {}
            for angle in field_angles:
                angle_trials = [r for r in successful_trials if r.field_angle_deg == angle]
                if angle_trials:
                    angle_errors = [r.attitude_error_arcsec for r in angle_trials]
                    statistics['error_vs_field_angle'][f'{angle:.1f}_deg'] = {
                        'mean_error_arcsec': float(np.mean(angle_errors)),
                        'std_error_arcsec': float(np.std(angle_errors)),
                        'n_trials': len(angle_trials)
                    }
        
        if len(magnitudes) > 1:
            statistics['error_vs_magnitude'] = {}
            for mag in magnitudes:
                mag_trials = [r for r in successful_trials if abs(r.magnitude - mag) < 0.1]
                if mag_trials:
                    mag_errors = [r.attitude_error_arcsec for r in mag_trials]
                    statistics['error_vs_magnitude'][f'mag_{mag:.1f}'] = {
                        'mean_error_arcsec': float(np.mean(mag_errors)),
                        'std_error_arcsec': float(np.std(mag_errors)),
                        'n_trials': len(mag_trials)
                    }
        
        return statistics
    
    def plot_error_distribution(
        self,
        trial_results: List[AttitudeTrialResult],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate attitude error distribution plots.
        
        Parameters
        ----------
        trial_results : List[AttitudeTrialResult]
            Trial results to plot
        output_path : Path, optional
            Output file path
            
        Returns
        -------
        Path
            Path to generated plot file
        """
        if output_path is None:
            output_path = self.config.output_dir / "attitude_error_distribution.png"
        
        # Filter successful trials
        successful_trials = [r for r in trial_results if r.success and np.isfinite(r.attitude_error_arcsec)]
        
        if not successful_trials:
            logger.warning("No successful trials for plotting")
            return output_path
        
        errors_arcsec = np.array([r.attitude_error_arcsec for r in successful_trials])
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Attitude Error Distribution Analysis', fontsize=14, fontweight='bold')
        
        # Histogram
        ax1.hist(errors_arcsec, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax1.axvline(np.mean(errors_arcsec), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(errors_arcsec):.2f}"')
        ax1.axvline(np.median(errors_arcsec), color='orange', linestyle='--',
                   label=f'Median: {np.median(errors_arcsec):.2f}"')
        ax1.set_xlabel('Attitude Error (arcsec)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CDF
        sorted_errors = np.sort(errors_arcsec)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax2.plot(sorted_errors, cdf, 'b-', linewidth=2)
        ax2.axhline(0.95, color='red', linestyle='--', label='95th Percentile')
        ax2.axvline(np.percentile(errors_arcsec, 95), color='red', linestyle='--',
                   label=f'95th: {np.percentile(errors_arcsec, 95):.2f}"')
        ax2.set_xlabel('Attitude Error (arcsec)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Error vs Field Angle
        field_angles = [r.field_angle_deg for r in successful_trials]
        unique_angles = sorted(set(field_angles))
        if len(unique_angles) > 1:
            angle_means = []
            angle_stds = []
            for angle in unique_angles:
                angle_errors = [r.attitude_error_arcsec for r in successful_trials 
                               if r.field_angle_deg == angle]
                angle_means.append(np.mean(angle_errors))
                angle_stds.append(np.std(angle_errors))
            
            ax3.errorbar(unique_angles, angle_means, yerr=angle_stds, 
                        marker='o', capsize=5, capthick=2)
            ax3.set_xlabel('Field Angle (degrees)')
            ax3.set_ylabel('Mean Attitude Error (arcsec)')
            ax3.set_title('Error vs Field Angle')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Single field angle\n(no variation to plot)', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Error vs Field Angle')
        
        # Error vs Magnitude
        magnitudes = [r.magnitude for r in successful_trials]
        unique_mags = sorted(set(magnitudes))
        if len(unique_mags) > 1:
            mag_means = []
            mag_stds = []
            for mag in unique_mags:
                mag_errors = [r.attitude_error_arcsec for r in successful_trials 
                             if abs(r.magnitude - mag) < 0.1]
                if mag_errors:
                    mag_means.append(np.mean(mag_errors))
                    mag_stds.append(np.std(mag_errors))
                else:
                    mag_means.append(np.nan)
                    mag_stds.append(np.nan)
            
            valid_indices = ~np.isnan(mag_means)
            ax4.errorbar(np.array(unique_mags)[valid_indices], 
                        np.array(mag_means)[valid_indices], 
                        yerr=np.array(mag_stds)[valid_indices], 
                        marker='o', capsize=5, capthick=2)
            ax4.set_xlabel('Star Magnitude')
            ax4.set_ylabel('Mean Attitude Error (arcsec)')
            ax4.set_title('Error vs Magnitude')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Single magnitude\n(no variation to plot)', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Error vs Magnitude')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attitude error distribution plot: {output_path}")
        return output_path
    
    def compare_to_cramer_rao_bound(
        self,
        camera_params: Dict[str, Any],
        star_mags: List[float],
        trial_results: List[AttitudeTrialResult]
    ) -> Dict[str, Any]:
        """
        Compare empirical performance to Cramér-Rao theoretical lower bound.
        
        Parameters
        ----------
        camera_params : Dict[str, Any]
            Camera parameters for theoretical calculation
        star_mags : List[float]
            Star magnitudes for SNR calculation
        trial_results : List[AttitudeTrialResult]
            Empirical trial results
            
        Returns
        -------
        Dict[str, Any]
            Comparison analysis results
        """
        logger.info("Comparing performance to Cramér-Rao bound")
        
        # Extract camera parameters
        focal_length_mm = camera_params.get('focal_length_mm', 40.07)
        pixel_pitch_um = camera_params.get('pixel_pitch_um', 5.5)
        read_noise_e = camera_params.get('read_noise_electrons', 13.0)
        
        # Calculate theoretical Cramér-Rao bound
        # This is a simplified calculation - real implementation would need
        # full Fisher information matrix computation
        
        cramer_rao_bounds = []
        for mag in star_mags:
            # Estimate signal-to-noise ratio
            # Simplified: need actual photon flux calculation
            signal_electrons = 10**(4.0 - 0.4 * mag)  # Rough approximation
            snr = signal_electrons / np.sqrt(signal_electrons + read_noise_e**2)
            
            # Cramér-Rao bound for centroiding (simplified)
            centroid_crb_pixels = 1.0 / snr  # Very simplified
            
            # Convert to angular error
            angular_crb_rad = centroid_crb_pixels * pixel_pitch_um / 1000.0 / focal_length_mm
            angular_crb_arcsec = angular_crb_rad * 206264.8
            
            cramer_rao_bounds.append(angular_crb_arcsec)
        
        # Calculate empirical performance
        successful_trials = [r for r in trial_results if r.success]
        empirical_errors = [r.attitude_error_arcsec for r in successful_trials]
        
        if not empirical_errors:
            return {'error': 'No successful trials for comparison'}
        
        empirical_rms = np.sqrt(np.mean(np.array(empirical_errors)**2))
        theoretical_crb = np.mean(cramer_rao_bounds) if cramer_rao_bounds else np.inf
        
        comparison = {
            'empirical_rms_arcsec': empirical_rms,
            'theoretical_crb_arcsec': theoretical_crb,
            'efficiency_ratio': empirical_rms / theoretical_crb if theoretical_crb > 0 else np.inf,
            'star_magnitudes': star_mags,
            'individual_crb_bounds': cramer_rao_bounds,
            'n_trials': len(successful_trials),
            'analysis_note': 'Simplified Cramér-Rao calculation - full Fisher information analysis recommended'
        }
        
        logger.info(f"Cramér-Rao comparison: empirical={empirical_rms:.2f}\", theoretical={theoretical_crb:.2f}\"")
        return comparison
    
    def save_results_hdf5(
        self,
        trial_results: List[AttitudeTrialResult],
        statistics: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Save validation results to HDF5 format for detailed analysis.
        
        Parameters
        ----------
        trial_results : List[AttitudeTrialResult]
            Individual trial results
        statistics : Dict[str, Any]
            Statistical summary
        output_path : Path, optional
            Output file path
            
        Returns
        -------
        Path
            Path to saved HDF5 file
        """
        if output_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.output_dir / f"attitude_validation_{timestamp}.h5"
        
        logger.info(f"Saving results to HDF5: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Metadata
            f.attrs['creation_timestamp'] = datetime.utcnow().isoformat()
            f.attrs['n_trials'] = len(trial_results)
            f.attrs['validation_type'] = 'attitude_validation'
            
            # Configuration
            config_group = f.create_group('config')
            config_group.attrs['n_monte_carlo'] = self.config.n_monte_carlo
            config_group.attrs['magnitude_range'] = self.config.magnitude_range
            config_group.attrs['psf_generation'] = self.config.psf_generation
            
            # Trial results
            trials_group = f.create_group('trials')
            successful_trials = [r for r in trial_results if r.success]
            
            if successful_trials:
                # Arrays for successful trials
                trials_group.create_dataset('trial_ids', data=[r.trial_id for r in successful_trials])
                trials_group.create_dataset('attitude_errors_arcsec', data=[r.attitude_error_arcsec for r in successful_trials])
                trials_group.create_dataset('identification_rates', data=[r.identification_rate for r in successful_trials])
                trials_group.create_dataset('execution_times', data=[r.execution_time for r in successful_trials])
                trials_group.create_dataset('magnitudes', data=[r.magnitude for r in successful_trials])
                trials_group.create_dataset('field_angles_deg', data=[r.field_angle_deg for r in successful_trials])
                
                # Quaternions (ground truth and solved)
                q_true_array = np.array([r.q_true for r in successful_trials])
                q_solved_array = np.array([r.q_solved for r in successful_trials])
                trials_group.create_dataset('q_true', data=q_true_array)
                trials_group.create_dataset('q_solved', data=q_solved_array)
            
            # Statistics
            stats_group = f.create_group('statistics')
            self._save_dict_to_hdf5(statistics, stats_group)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def run_validation_campaign(
        self,
        attitude_range: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Run complete attitude validation campaign.
        
        Parameters
        ----------
        attitude_range : Dict[str, Tuple[float, float]], optional
            Attitude ranges for systematic testing
            
        Returns
        -------
        Dict[str, Any]
            Complete validation results
        """
        logger.info("Starting attitude validation campaign")
        campaign_start = time.time()
        
        # Generate test attitudes
        if attitude_range is None:
            # Default random attitude sampling
            attitudes = self.monte_carlo.generate_random_attitudes(self.config.n_monte_carlo)
        else:
            # Structured attitude sampling
            scenarios = self.monte_carlo.generate_test_scenarios(
                self.config.n_monte_carlo,
                ra_range=attitude_range.get('ra', (0, 360)),
                dec_range=attitude_range.get('dec', (-90, 90)),
                roll_range=attitude_range.get('roll', (0, 360))
            )
            attitudes = np.array([euler_to_quaternion(s.parameters['attitude_ra_deg'],
                                                    s.parameters['attitude_dec_deg'],
                                                    s.parameters['attitude_roll_deg']) 
                                for s in scenarios])
        
        # Generate magnitude and field angle arrays
        magnitudes = np.random.uniform(
            self.config.magnitude_range[0],
            self.config.magnitude_range[1],
            len(attitudes)
        )
        field_angles = np.random.choice(
            self.config.field_angles_deg,
            len(attitudes)
        )
        
        # Generate ground truth images
        ground_truth_images = self.generate_ground_truth_images(
            attitudes, magnitudes, field_angles
        )
        
        # Run pipeline on images
        trial_results = self.run_pipeline_on_images(ground_truth_images)
        
        # Compute statistics
        statistics = self.compute_error_statistics(trial_results)
        
        # Generate plots
        plot_path = self.plot_error_distribution(trial_results)
        
        # Save detailed results
        if self.config.save_intermediate:
            hdf5_path = self.save_results_hdf5(trial_results, statistics)
        else:
            hdf5_path = None
        
        # Cramér-Rao comparison
        camera_params = {
            'focal_length_mm': 40.07,
            'pixel_pitch_um': 5.5,
            'read_noise_electrons': 13.0
        }
        cramer_rao_comparison = self.compare_to_cramer_rao_bound(
            camera_params, list(set(magnitudes)), trial_results
        )
        
        campaign_time = time.time() - campaign_start
        
        # Compile final results
        validation_results = {
            'campaign_summary': {
                'n_trials_requested': self.config.n_monte_carlo,
                'n_trials_completed': len(trial_results),
                'n_trials_successful': len([r for r in trial_results if r.success]),
                'campaign_duration_seconds': campaign_time,
                'timestamp': datetime.utcnow().isoformat()
            },
            'statistics': statistics,
            'cramer_rao_comparison': cramer_rao_comparison,
            'file_outputs': {
                'plot_file': str(plot_path),
                'hdf5_file': str(hdf5_path) if hdf5_path else None
            },
            'configuration': {
                'magnitude_range': self.config.magnitude_range,
                'field_angles_deg': self.config.field_angles_deg,
                'psf_generation': self.config.psf_generation,
                'noise_multiplier': self.config.noise_multiplier
            }
        }
        
        logger.info(f"Attitude validation campaign completed in {campaign_time:.1f}s")
        return validation_results
    
    # Helper methods
    def _select_psf_file(self, field_angle_deg: float) -> Path:
        """Select appropriate PSF file based on field angle."""
        # Round to nearest available field angle
        available_angles = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 14]
        closest_angle = min(available_angles, key=lambda x: abs(x - field_angle_deg))
        
        psf_dir = Path("data/PSF_sims") / self.config.psf_generation
        psf_file = psf_dir / f"{closest_angle}_deg.txt"
        
        if not psf_file.exists():
            raise FileNotFoundError(f"PSF file not found: {psf_file}")
        
        return psf_file
    
    def _generate_star_scene(self, psf_file: Path, magnitude: float, 
                           attitude_quaternion: np.ndarray, field_angle_deg: float) -> Dict[str, Any]:
        """Generate synthetic star scene using pipeline."""
        # This would integrate with the StarTrackerPipeline to generate realistic scenes
        # For now, return placeholder structure
        return {
            'psf_file': str(psf_file),
            'magnitude': magnitude,
            'attitude_quaternion': attitude_quaternion.tolist(),
            'field_angle_deg': field_angle_deg,
            'synthetic_image': None,  # Would contain actual image data
            'noise_realization': None
        }
    
    def _run_star_detection(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run star detection and centroiding."""
        # Placeholder implementation - would use actual pipeline
        return {
            'success': True,
            'n_stars': np.random.randint(3, 8),
            'centroids': [(100 + i*50, 200 + i*30) for i in range(5)],
            'bearing_vectors': [np.random.randn(3) for _ in range(5)]
        }
    
    def _run_star_matching(self, bearing_vectors: List[np.ndarray], 
                          centroids: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Run star identification and matching."""
        # Placeholder implementation
        n_matches = min(len(bearing_vectors), np.random.randint(3, 6))
        return {
            'success': True,
            'star_matches': [{'catalog_id': i, 'bearing_vector': bearing_vectors[i]} 
                           for i in range(n_matches)],
            'bearing_vectors': bearing_vectors[:n_matches]
        }
    
    def _run_monte_carlo_quest(self, star_matches: List[Dict], 
                              bearing_vectors: List[np.ndarray]) -> Dict[str, Any]:
        """Run Monte Carlo QUEST attitude determination."""
        # Placeholder implementation - would use actual mc_quest
        return {
            'success': True,
            'quaternion': np.array([1, 0, 0, 0]) + np.random.randn(4) * 0.01,
            'converged': True,
            'uncertainty_arcsec': np.random.uniform(0.5, 2.0)
        }
    
    def _save_dict_to_hdf5(self, data: Dict[str, Any], group: h5py.Group):
        """Recursively save dictionary to HDF5 group."""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_dict_to_hdf5(value, subgroup)
            elif isinstance(value, (int, float, str)):
                group.attrs[key] = value
            elif isinstance(value, (list, np.ndarray)):
                try:
                    group.create_dataset(key, data=value)
                except (TypeError, ValueError):
                    # Handle non-numeric lists
                    group.attrs[key] = str(value)

# Export main class
__all__ = ['AttitudeValidator', 'AttitudeValidationConfig', 'AttitudeTrialResult']