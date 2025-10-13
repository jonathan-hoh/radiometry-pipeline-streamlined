#!/usr/bin/env python3
"""
validation/noise_validation.py - Noise Characterization and Validation

Comprehensive noise analysis and centroiding sensitivity validation:
- Multiple noise source injection (read noise, shot noise, dark current)
- Centroiding error vs SNR scaling validation
- Noise statistics verification (Gaussian/Poisson characteristics)
- Performance degradation analysis under various noise conditions
- Environmental noise sensitivity (temperature, radiation effects)

Usage:
    from validation.noise_validation import NoiseValidator
    
    validator = NoiseValidator(pipeline, detector_model)
    results = validator.run_noise_characterization()
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from datetime import datetime
from scipy import stats
from scipy.optimize import curve_fit

# Import core pipeline components
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.core.psf_photon_simulation import simulate_psf_with_poisson_noise
from src.BAST.identify import calculate_centroid

# Import validation framework
from .metrics import calculate_snr, centroid_rms, ValidationResults
from .monte_carlo import MonteCarloValidator

logger = logging.getLogger(__name__)

@dataclass
class NoiseConfig:
    """Configuration for noise validation."""
    snr_range: Tuple[float, float] = (3.0, 1000.0)  # SNR test range
    n_snr_points: int = 20  # Number of SNR test points
    n_trials_per_snr: int = 100  # Trials per SNR point
    noise_types: List[str] = None  # Types to test: read, shot, dark, thermal
    read_noise_levels: List[float] = None  # Read noise levels (electrons RMS)
    dark_current_levels: List[float] = None  # Dark current (electrons/pixel/sec)
    temperature_range: Tuple[float, float] = (-40, 20)  # Temperature range (°C)
    magnitude_test_points: List[float] = None  # Magnitudes for scaling tests
    validate_noise_statistics: bool = True
    centroiding_algorithm: str = "moment_based"  # Algorithm to test
    output_dir: Union[str, Path] = "validation/results/noise"
    save_noise_maps: bool = True
    theoretical_comparison: bool = True
    
    def __post_init__(self):
        if self.noise_types is None:
            self.noise_types = ["read", "shot", "dark", "combined"]
        if self.read_noise_levels is None:
            self.read_noise_levels = [5.0, 10.0, 13.0, 20.0, 30.0]  # electrons RMS
        if self.dark_current_levels is None:
            self.dark_current_levels = [0.1, 1.0, 5.0, 10.0]  # electrons/pixel/sec
        if self.magnitude_test_points is None:
            self.magnitude_test_points = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        self.output_dir = Path(self.output_dir)

@dataclass
class NoiseMeasurement:
    """Single noise characterization measurement."""
    snr_input: float
    snr_measured: float
    signal_electrons: float
    noise_electrons: float
    read_noise_electrons: float
    dark_current_electrons: float
    shot_noise_electrons: float
    centroid_error_x: float  # pixels
    centroid_error_y: float  # pixels
    centroid_error_radial: float  # pixels
    true_centroid_x: float
    true_centroid_y: float
    measured_centroid_x: float
    measured_centroid_y: float
    magnitude: float
    exposure_time: float
    temperature_c: float
    noise_type: str
    algorithm: str
    measurement_id: int
    timestamp: str

@dataclass
class NoiseStatistics:
    """Noise statistics validation results."""
    noise_type: str
    mean: float
    std: float
    skewness: float
    kurtosis: float
    normality_test_p_value: float
    is_gaussian: bool
    theoretical_std: float
    measured_vs_theoretical_ratio: float

class NoiseValidator:
    """
    Comprehensive noise characterization and validation.
    
    Validates noise performance by:
    1. Injecting controlled noise sources into synthetic images
    2. Measuring centroiding performance vs SNR
    3. Validating noise statistical properties
    4. Characterizing sensitivity degradation
    5. Comparing to theoretical noise models
    """
    
    def __init__(
        self,
        pipeline: StarTrackerPipeline,
        detector_model: Any,
        config: Optional[NoiseConfig] = None
    ):
        """
        Initialize noise validator.
        
        Parameters
        ----------
        pipeline : StarTrackerPipeline
            Star tracker pipeline for processing
        detector_model : Any
            Detector model for noise simulation
        config : NoiseConfig, optional
            Validation configuration
        """
        self.pipeline = pipeline
        self.detector_model = detector_model
        self.config = config or NoiseConfig()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detector parameters
        self.detector_params = self._extract_detector_parameters()
        
        logger.info(f"NoiseValidator initialized for SNR range {self.config.snr_range}")
        
    def inject_noise_sources(
        self,
        image: np.ndarray,
        read_noise_std: float,
        dark_current: float,
        shot_noise: bool = True,
        exposure_time: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Inject realistic noise sources into clean image.
        
        Parameters
        ----------
        image : np.ndarray
            Clean input image (signal only)
        read_noise_std : float
            Read noise standard deviation (electrons RMS)
        dark_current : float
            Dark current (electrons/pixel/second)
        shot_noise : bool
            Whether to include Poisson shot noise
        exposure_time : float
            Exposure time for dark current scaling
            
        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Noisy image and noise statistics
        """
        noisy_image = image.copy().astype(float)
        noise_components = {}
        
        # 1. Add dark current (Poisson distributed)
        if dark_current > 0:
            dark_electrons = dark_current * exposure_time
            dark_noise = np.random.poisson(dark_electrons, image.shape)
            noisy_image += dark_noise
            noise_components['dark_current'] = dark_noise
            logger.debug(f"Added dark current: {dark_electrons:.2f} e⁻/pixel")
        
        # 2. Add shot noise to signal + dark current (Poisson)
        if shot_noise:
            # Shot noise is already included if image contains Poisson-distributed signal
            # For pure signal, apply Poisson noise
            signal_with_dark = noisy_image.copy()
            shot_noisy = np.random.poisson(signal_with_dark)
            shot_noise_component = shot_noisy - signal_with_dark
            noisy_image = shot_noisy
            noise_components['shot_noise'] = shot_noise_component
            logger.debug(f"Applied Poisson shot noise")
        
        # 3. Add read noise (Gaussian)
        if read_noise_std > 0:
            read_noise = np.random.normal(0, read_noise_std, image.shape)
            noisy_image += read_noise
            noise_components['read_noise'] = read_noise
            logger.debug(f"Added read noise: {read_noise_std:.2f} e⁻ RMS")
        
        # Calculate total noise statistics
        total_noise = noisy_image - image
        noise_stats = {
            'total_noise_std': float(np.std(total_noise)),
            'total_noise_mean': float(np.mean(total_noise)),
            'read_noise_contribution': read_noise_std,
            'dark_current_contribution': dark_current * exposure_time,
            'signal_dependent_noise': shot_noise,
            'exposure_time': exposure_time
        }
        
        return noisy_image, noise_stats
    
    def centroid_error_vs_snr(
        self,
        snr_range: Optional[Tuple[float, float]] = None,
        magnitude: float = 4.0
    ) -> Dict[str, Any]:
        """
        Characterize centroiding error vs SNR relationship.
        
        Parameters
        ----------
        snr_range : Tuple[float, float], optional
            SNR range to test
        magnitude : float
            Star magnitude for testing
            
        Returns
        -------
        Dict[str, Any]
            Centroiding error vs SNR analysis results
        """
        if snr_range is None:
            snr_range = self.config.snr_range
        
        logger.info(f"Characterizing centroiding error vs SNR: {snr_range}")
        
        # Generate SNR test points (logarithmic spacing)
        snr_points = np.logspace(
            np.log10(snr_range[0]),
            np.log10(snr_range[1]),
            self.config.n_snr_points
        )
        
        measurements = []
        
        for snr_target in snr_points:
            logger.debug(f"Testing SNR = {snr_target:.2f}")
            
            # Calculate required signal level for target SNR
            read_noise = self.detector_params['read_noise_electrons']
            
            # For shot noise limited case: SNR = sqrt(signal)
            # For read noise limited case: SNR = signal / read_noise
            # Combined: SNR = signal / sqrt(signal + read_noise^2)
            
            # Solve for signal: SNR^2 = signal^2 / (signal + read_noise^2)
            # SNR^2 * (signal + read_noise^2) = signal^2
            # SNR^2 * signal + SNR^2 * read_noise^2 = signal^2
            # signal^2 - SNR^2 * signal - SNR^2 * read_noise^2 = 0
            # signal = (SNR^2 + sqrt(SNR^4 + 4*SNR^2*read_noise^2)) / 2
            
            discriminant = snr_target**4 + 4 * snr_target**2 * read_noise**2
            signal_electrons = (snr_target**2 + np.sqrt(discriminant)) / 2
            
            # Run multiple trials at this SNR
            snr_measurements = []
            centroid_errors = []
            
            for trial in range(self.config.n_trials_per_snr):
                try:
                    # Generate synthetic star image
                    clean_image = self._generate_synthetic_star_image(
                        signal_electrons, magnitude
                    )
                    
                    # Add noise to achieve target SNR
                    noisy_image, noise_stats = self.inject_noise_sources(
                        clean_image,
                        read_noise_std=read_noise,
                        dark_current=1.0,  # Standard dark current
                        shot_noise=True,
                        exposure_time=1.0
                    )
                    
                    # Measure actual SNR
                    actual_snr = self._measure_image_snr(noisy_image, clean_image)
                    
                    # Perform centroiding
                    true_centroid = self._get_true_centroid_position(clean_image)
                    measured_centroid = self._measure_centroid(noisy_image)
                    
                    # Calculate centroiding error
                    error_x = measured_centroid[0] - true_centroid[0]
                    error_y = measured_centroid[1] - true_centroid[1]
                    error_radial = np.sqrt(error_x**2 + error_y**2)
                    
                    # Create measurement record
                    measurement = NoiseMeasurement(
                        snr_input=snr_target,
                        snr_measured=actual_snr,
                        signal_electrons=signal_electrons,
                        noise_electrons=noise_stats['total_noise_std'],
                        read_noise_electrons=read_noise,
                        dark_current_electrons=noise_stats['dark_current_contribution'],
                        shot_noise_electrons=np.sqrt(signal_electrons),
                        centroid_error_x=error_x,
                        centroid_error_y=error_y,
                        centroid_error_radial=error_radial,
                        true_centroid_x=true_centroid[0],
                        true_centroid_y=true_centroid[1],
                        measured_centroid_x=measured_centroid[0],
                        measured_centroid_y=measured_centroid[1],
                        magnitude=magnitude,
                        exposure_time=1.0,
                        temperature_c=-20.0,
                        noise_type="combined",
                        algorithm=self.config.centroiding_algorithm,
                        measurement_id=len(measurements),
                        timestamp=datetime.utcnow().isoformat()
                    )
                    
                    measurements.append(measurement)
                    snr_measurements.append(actual_snr)
                    centroid_errors.append(error_radial)
                    
                except Exception as e:
                    logger.error(f"Trial failed at SNR {snr_target:.2f}: {e}")
                    continue
        
        # Analyze results
        analysis_results = self._analyze_centroiding_vs_snr(measurements)
        
        # Theoretical comparison
        if self.config.theoretical_comparison:
            theoretical_analysis = self._compare_to_theoretical_centroiding_limit(measurements)
            analysis_results['theoretical_comparison'] = theoretical_analysis
        
        return analysis_results
    
    def validate_noise_statistics(
        self,
        noise_realizations: List[np.ndarray]
    ) -> Dict[str, NoiseStatistics]:
        """
        Validate noise statistical properties (Gaussian/Poisson).
        
        Parameters
        ----------
        noise_realizations : List[np.ndarray]
            Multiple noise realizations for statistical analysis
            
        Returns
        -------
        Dict[str, NoiseStatistics]
            Statistical validation results for each noise type
        """
        logger.info(f"Validating noise statistics for {len(noise_realizations)} realizations")
        
        noise_statistics = {}
        
        # Test different noise components
        test_configs = [
            {'type': 'read_noise', 'std': 13.0, 'dark': 0.0, 'shot': False},
            {'type': 'shot_noise', 'std': 0.0, 'dark': 0.0, 'shot': True},
            {'type': 'dark_current', 'std': 0.0, 'dark': 5.0, 'shot': False},
            {'type': 'combined', 'std': 13.0, 'dark': 1.0, 'shot': True}
        ]
        
        for config in test_configs:
            logger.debug(f"Testing {config['type']} noise statistics")
            
            noise_samples = []
            
            # Generate noise samples
            for realization in noise_realizations[:10]:  # Use subset for efficiency
                try:
                    # Create clean signal image
                    signal_level = 1000.0  # electrons
                    clean_image = np.full_like(realization, signal_level)
                    
                    # Add specified noise
                    noisy_image, _ = self.inject_noise_sources(
                        clean_image,
                        read_noise_std=config['std'],
                        dark_current=config['dark'],
                        shot_noise=config['shot'],
                        exposure_time=1.0
                    )
                    
                    # Extract noise component
                    noise_component = noisy_image - clean_image
                    noise_samples.extend(noise_component.flatten())
                    
                except Exception as e:
                    logger.error(f"Failed to generate {config['type']} noise: {e}")
                    continue
            
            if noise_samples:
                # Statistical analysis
                noise_array = np.array(noise_samples)
                
                # Basic statistics
                mean_noise = np.mean(noise_array)
                std_noise = np.std(noise_array)
                skewness = stats.skew(noise_array)
                kurtosis = stats.kurtosis(noise_array)
                
                # Normality test
                if len(noise_array) > 5000:
                    # Use subset for Shapiro-Wilk test (max 5000 samples)
                    test_subset = np.random.choice(noise_array, 5000, replace=False)
                else:
                    test_subset = noise_array
                
                _, normality_p = stats.shapiro(test_subset)
                is_gaussian = normality_p > 0.05
                
                # Theoretical expectation
                if config['type'] == 'read_noise':
                    theoretical_std = config['std']
                elif config['type'] == 'shot_noise':
                    theoretical_std = np.sqrt(signal_level)
                elif config['type'] == 'dark_current':
                    theoretical_std = np.sqrt(config['dark'])
                else:  # combined
                    theoretical_std = np.sqrt(signal_level + config['std']**2 + config['dark'])
                
                ratio = std_noise / theoretical_std if theoretical_std > 0 else np.inf
                
                # Create statistics object
                noise_stats = NoiseStatistics(
                    noise_type=config['type'],
                    mean=float(mean_noise),
                    std=float(std_noise),
                    skewness=float(skewness),
                    kurtosis=float(kurtosis),
                    normality_test_p_value=float(normality_p),
                    is_gaussian=is_gaussian,
                    theoretical_std=float(theoretical_std),
                    measured_vs_theoretical_ratio=float(ratio)
                )
                
                noise_statistics[config['type']] = noise_stats
                
                logger.info(f"{config['type']}: σ_measured = {std_noise:.2f}, σ_theoretical = {theoretical_std:.2f}")
        
        return noise_statistics
    
    def sensitivity_degradation(
        self,
        magnitude_range: List[float],
        noise_levels: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze performance degradation vs noise level.
        
        Parameters
        ----------
        magnitude_range : List[float]
            Star magnitudes to test
        noise_levels : List[float]
            Noise multipliers to test
            
        Returns
        -------
        Dict[str, Any]
            Sensitivity degradation analysis results
        """
        logger.info(f"Analyzing sensitivity degradation for {len(noise_levels)} noise levels")
        
        degradation_results = {}
        
        for noise_multiplier in noise_levels:
            logger.debug(f"Testing noise level: {noise_multiplier}x nominal")
            
            magnitude_performance = {}
            
            for magnitude in magnitude_range:
                try:
                    # Calculate signal level for magnitude
                    signal_electrons = self._magnitude_to_electrons(magnitude)
                    
                    # Scale noise components
                    read_noise = self.detector_params['read_noise_electrons'] * noise_multiplier
                    dark_current = self.detector_params.get('dark_current', 1.0) * noise_multiplier
                    
                    # Run centroiding trials
                    centroid_errors = []
                    snr_values = []
                    
                    for trial in range(50):  # Reduced trials for efficiency
                        # Generate test image
                        clean_image = self._generate_synthetic_star_image(signal_electrons, magnitude)
                        
                        # Add scaled noise
                        noisy_image, noise_stats = self.inject_noise_sources(
                            clean_image,
                            read_noise_std=read_noise,
                            dark_current=dark_current,
                            shot_noise=True
                        )
                        
                        # Measure performance
                        snr = self._measure_image_snr(noisy_image, clean_image)
                        
                        true_centroid = self._get_true_centroid_position(clean_image)
                        measured_centroid = self._measure_centroid(noisy_image)
                        
                        error_radial = np.sqrt(
                            (measured_centroid[0] - true_centroid[0])**2 +
                            (measured_centroid[1] - true_centroid[1])**2
                        )
                        
                        centroid_errors.append(error_radial)
                        snr_values.append(snr)
                    
                    # Calculate performance metrics
                    magnitude_performance[magnitude] = {
                        'mean_centroid_error': float(np.mean(centroid_errors)),
                        'std_centroid_error': float(np.std(centroid_errors)),
                        'mean_snr': float(np.mean(snr_values)),
                        'centroid_rms': float(np.sqrt(np.mean(np.array(centroid_errors)**2))),
                        'success_rate': float(np.sum(np.array(snr_values) > 3.0) / len(snr_values))
                    }
                    
                except Exception as e:
                    logger.error(f"Failed magnitude {magnitude} at noise level {noise_multiplier}: {e}")
                    magnitude_performance[magnitude] = {
                        'mean_centroid_error': np.inf,
                        'std_centroid_error': np.inf,
                        'mean_snr': 0.0,
                        'centroid_rms': np.inf,
                        'success_rate': 0.0
                    }
            
            degradation_results[f'noise_{noise_multiplier}x'] = magnitude_performance
        
        # Calculate degradation factors
        degradation_analysis = self._analyze_sensitivity_degradation(degradation_results)
        
        return {
            'degradation_data': degradation_results,
            'degradation_analysis': degradation_analysis,
            'test_parameters': {
                'magnitude_range': magnitude_range,
                'noise_levels': noise_levels,
                'base_read_noise': self.detector_params['read_noise_electrons']
            }
        }
    
    def run_noise_characterization(
        self,
        comprehensive_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete noise characterization campaign.
        
        Parameters
        ----------
        comprehensive_analysis : bool
            Whether to run all noise validation tests
            
        Returns
        -------
        Dict[str, Any]
            Complete noise characterization results
        """
        logger.info("Starting comprehensive noise characterization")
        campaign_start = time.time()
        
        characterization_results = {
            'campaign_summary': {
                'start_time': datetime.utcnow().isoformat(),
                'configuration': {
                    'snr_range': self.config.snr_range,
                    'noise_types': self.config.noise_types,
                    'centroiding_algorithm': self.config.centroiding_algorithm
                }
            }
        }
        
        # 1. Centroiding error vs SNR analysis
        logger.info("Running centroiding error vs SNR analysis...")
        centroiding_analysis = self.centroid_error_vs_snr()
        characterization_results['centroiding_vs_snr'] = centroiding_analysis
        
        # 2. Noise statistics validation
        if self.config.validate_noise_statistics:
            logger.info("Validating noise statistics...")
            # Generate noise realizations for testing
            test_images = [np.random.normal(1000, 50, (64, 64)) for _ in range(20)]
            noise_stats = self.validate_noise_statistics(test_images)
            characterization_results['noise_statistics'] = {
                name: {
                    'noise_type': stats.noise_type,
                    'mean': stats.mean,
                    'std': stats.std,
                    'skewness': stats.skewness,
                    'kurtosis': stats.kurtosis,
                    'normality_test_p_value': stats.normality_test_p_value,
                    'is_gaussian': stats.is_gaussian,
                    'theoretical_std': stats.theoretical_std,
                    'measured_vs_theoretical_ratio': stats.measured_vs_theoretical_ratio
                } for name, stats in noise_stats.items()
            }
        
        # 3. Sensitivity degradation analysis
        logger.info("Running sensitivity degradation analysis...")
        degradation_analysis = self.sensitivity_degradation(
            magnitude_range=self.config.magnitude_test_points,
            noise_levels=[0.5, 1.0, 1.5, 2.0, 3.0]
        )
        characterization_results['sensitivity_degradation'] = degradation_analysis
        
        # 4. Temperature sensitivity (if configured)
        if hasattr(self.config, 'temperature_range'):
            logger.info("Running temperature sensitivity analysis...")
            temperature_analysis = self._analyze_temperature_sensitivity()
            characterization_results['temperature_sensitivity'] = temperature_analysis
        
        # 5. Generate analysis plots
        plot_paths = self._generate_noise_plots(characterization_results)
        characterization_results['plot_files'] = plot_paths
        
        # 6. Save detailed results
        if self.config.save_noise_maps:
            results_file = self._save_noise_characterization_data(characterization_results)
            characterization_results['detailed_results_file'] = str(results_file)
        
        campaign_time = time.time() - campaign_start
        characterization_results['campaign_summary'].update({
            'end_time': datetime.utcnow().isoformat(),
            'duration_seconds': campaign_time,
            'characterization_modules_run': list(characterization_results.keys())
        })
        
        logger.info(f"Noise characterization completed in {campaign_time:.1f}s")
        return characterization_results
    
    # Helper methods
    def _extract_detector_parameters(self) -> Dict[str, Any]:
        """Extract detector parameters from model."""
        return {
            'model': 'CMV4000',
            'pixel_pitch_um': 5.5,
            'read_noise_electrons': 13.0,
            'dark_current': 1.0,  # electrons/pixel/sec
            'full_well_electrons': 13500,
            'quantum_efficiency': 0.6,
            'gain_e_per_adu': 1.0,
            'temperature_c': -20
        }
    
    def _generate_synthetic_star_image(
        self,
        signal_electrons: float,
        magnitude: float,
        image_size: Tuple[int, int] = (64, 64)
    ) -> np.ndarray:
        """Generate synthetic star image with specified signal level."""
        # Create Gaussian PSF
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        x = np.arange(image_size[0]) - center_x
        y = np.arange(image_size[1]) - center_y
        X, Y = np.meshgrid(x, y)
        
        # PSF parameters
        sigma = 1.5  # pixels
        psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        psf = psf / np.sum(psf)  # Normalize
        
        # Scale to desired signal level
        star_image = psf * signal_electrons
        
        return star_image
    
    def _measure_image_snr(self, noisy_image: np.ndarray, clean_image: np.ndarray) -> float:
        """Measure SNR of noisy image."""
        signal = np.sum(clean_image)
        noise_image = noisy_image - clean_image
        noise = np.std(noise_image)
        
        return signal / noise if noise > 0 else np.inf
    
    def _get_true_centroid_position(self, image: np.ndarray) -> Tuple[float, float]:
        """Get true centroid position from clean image."""
        # Moment-based centroiding
        y_indices, x_indices = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        
        total_intensity = np.sum(image)
        if total_intensity > 0:
            centroid_x = np.sum(x_indices * image) / total_intensity
            centroid_y = np.sum(y_indices * image) / total_intensity
        else:
            centroid_x = image.shape[1] / 2
            centroid_y = image.shape[0] / 2
        
        return (centroid_x, centroid_y)
    
    def _measure_centroid(self, noisy_image: np.ndarray) -> Tuple[float, float]:
        """Measure centroid from noisy image using specified algorithm."""
        if self.config.centroiding_algorithm == "moment_based":
            return self._moment_based_centroiding(noisy_image)
        else:
            # Default to moment-based
            return self._moment_based_centroiding(noisy_image)
    
    def _moment_based_centroiding(self, image: np.ndarray) -> Tuple[float, float]:
        """Moment-based centroiding with noise handling."""
        # Apply threshold to reduce noise impact
        threshold = np.mean(image) + 2 * np.std(image)
        thresholded = np.maximum(image - threshold, 0)
        
        # Calculate moments
        y_indices, x_indices = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        
        total_intensity = np.sum(thresholded)
        if total_intensity > 0:
            centroid_x = np.sum(x_indices * thresholded) / total_intensity
            centroid_y = np.sum(y_indices * thresholded) / total_intensity
        else:
            # Fall back to image center if no signal above threshold
            centroid_x = image.shape[1] / 2
            centroid_y = image.shape[0] / 2
        
        return (centroid_x, centroid_y)
    
    def _magnitude_to_electrons(self, magnitude: float) -> float:
        """Convert magnitude to electron count."""
        # Simple conversion (1 second exposure)
        flux_0_mag = 1e6  # electrons for magnitude 0
        electrons = flux_0_mag * 10**(-0.4 * magnitude)
        return electrons
    
    def _analyze_centroiding_vs_snr(
        self,
        measurements: List[NoiseMeasurement]
    ) -> Dict[str, Any]:
        """Analyze centroiding error vs SNR relationship."""
        if not measurements:
            return {'error': 'No measurements for analysis'}
        
        # Extract data arrays
        snr_values = np.array([m.snr_measured for m in measurements])
        centroid_errors = np.array([m.centroid_error_radial for m in measurements])
        
        # Filter valid measurements
        valid_mask = np.isfinite(snr_values) & np.isfinite(centroid_errors) & (snr_values > 0)
        snr_valid = snr_values[valid_mask]
        errors_valid = centroid_errors[valid_mask]
        
        if len(snr_valid) < 3:
            return {'error': 'Insufficient valid measurements for analysis'}
        
        # Fit theoretical scaling: error = A / SNR^B
        try:
            def scaling_model(snr, A, B):
                return A / (snr**B)
            
            # Use log-log fit for better numerical stability
            log_snr = np.log10(snr_valid)
            log_error = np.log10(errors_valid)
            
            # Linear fit: log(error) = log(A) - B * log(SNR)
            coeffs = np.polyfit(log_snr, log_error, 1)
            B_fit = -coeffs[0]  # Negative slope
            A_fit = 10**coeffs[1]
            
            # Calculate fit quality
            predicted_errors = scaling_model(snr_valid, A_fit, B_fit)
            residuals = errors_valid - predicted_errors
            rms_error = np.sqrt(np.mean(residuals**2))
            r_squared = 1 - np.var(residuals) / np.var(errors_valid)
            
            scaling_analysis = {
                'scaling_fit': {
                    'amplitude': float(A_fit),
                    'exponent': float(B_fit),
                    'rms_fit_error': float(rms_error),
                    'r_squared': float(r_squared)
                },
                'theoretical_expectation': {
                    'expected_exponent': 1.0,  # σ ∝ 1/SNR for shot noise limit
                    'exponent_ratio': float(B_fit / 1.0)
                }
            }
            
        except Exception as e:
            scaling_analysis = {
                'scaling_fit': {'error': f'Fitting failed: {str(e)}'},
                'theoretical_expectation': {}
            }
        
        # Statistical summary
        analysis_results = {
            'measurement_summary': {
                'total_measurements': len(measurements),
                'valid_measurements': len(snr_valid),
                'snr_range': [float(np.min(snr_valid)), float(np.max(snr_valid))],
                'error_range': [float(np.min(errors_valid)), float(np.max(errors_valid))]
            },
            'scaling_analysis': scaling_analysis,
            'binned_statistics': self._create_snr_bins(snr_valid, errors_valid)
        }
        
        return analysis_results
    
    def _compare_to_theoretical_centroiding_limit(
        self,
        measurements: List[NoiseMeasurement]
    ) -> Dict[str, Any]:
        """Compare measured centroiding to theoretical limits."""
        # Theoretical Cramér-Rao bound for centroiding
        # σ_centroid ≈ σ_pixel / (SNR * sqrt(N_pixels))
        
        theoretical_comparison = {
            'cramer_rao_bounds': [],
            'empirical_performance': [],
            'efficiency_ratios': []
        }
        
        for measurement in measurements:
            if measurement.snr_measured > 0:
                # Simplified Cramér-Rao bound
                psf_width_pixels = 2.0  # Typical PSF width
                n_effective_pixels = np.pi * psf_width_pixels**2
                pixel_noise = 1.0  # Normalized
                
                crb = pixel_noise / (measurement.snr_measured * np.sqrt(n_effective_pixels))
                empirical = measurement.centroid_error_radial
                efficiency = empirical / crb if crb > 0 else np.inf
                
                theoretical_comparison['cramer_rao_bounds'].append(crb)
                theoretical_comparison['empirical_performance'].append(empirical)
                theoretical_comparison['efficiency_ratios'].append(efficiency)
        
        # Summary statistics
        if theoretical_comparison['efficiency_ratios']:
            efficiency_ratios = np.array(theoretical_comparison['efficiency_ratios'])
            valid_ratios = efficiency_ratios[np.isfinite(efficiency_ratios)]
            
            theoretical_comparison['summary'] = {
                'mean_efficiency_ratio': float(np.mean(valid_ratios)) if len(valid_ratios) > 0 else np.inf,
                'median_efficiency_ratio': float(np.median(valid_ratios)) if len(valid_ratios) > 0 else np.inf,
                'efficiency_interpretation': 'Ratio > 1 indicates performance worse than theoretical limit'
            }
        
        return theoretical_comparison
    
    def _analyze_sensitivity_degradation(
        self,
        degradation_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Analyze sensitivity degradation patterns."""
        noise_levels = list(degradation_results.keys())
        magnitudes = list(degradation_results[noise_levels[0]].keys()) if noise_levels else []
        
        degradation_analysis = {
            'degradation_factors': {},
            'limiting_magnitude_shift': {},
            'performance_thresholds': {}
        }
        
        # Calculate degradation factors relative to lowest noise level
        if len(noise_levels) >= 2:
            baseline_key = min(noise_levels)  # Lowest noise level
            baseline_data = degradation_results[baseline_key]
            
            for noise_key in noise_levels:
                if noise_key != baseline_key:
                    degradation_factors = {}
                    for mag in magnitudes:
                        if mag in baseline_data and mag in degradation_results[noise_key]:
                            baseline_error = baseline_data[mag]['mean_centroid_error']
                            current_error = degradation_results[noise_key][mag]['mean_centroid_error']
                            
                            if baseline_error > 0 and np.isfinite(current_error):
                                degradation_factors[mag] = current_error / baseline_error
                            else:
                                degradation_factors[mag] = np.inf
                    
                    degradation_analysis['degradation_factors'][noise_key] = degradation_factors
        
        return degradation_analysis
    
    def _analyze_temperature_sensitivity(self) -> Dict[str, Any]:
        """Analyze temperature sensitivity of noise performance."""
        # Placeholder for temperature analysis
        # Would implement actual temperature-dependent noise measurements
        
        temperature_analysis = {
            'temperature_range_c': self.config.temperature_range,
            'dark_current_scaling': {
                'doubling_temperature_c': 7.0,  # Typical for CCDs
                'scaling_model': 'exponential'
            },
            'read_noise_temperature_dependence': {
                'temperature_coefficient': 0.02,  # e⁻/°C typical
                'measurement_note': 'Read noise typically shows weak temperature dependence'
            },
            'analysis_note': 'Full temperature characterization requires hardware testing'
        }
        
        return temperature_analysis
    
    def _create_snr_bins(
        self,
        snr_values: np.ndarray,
        error_values: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Create binned SNR vs error statistics."""
        # Create logarithmic bins
        log_snr_min = np.log10(np.min(snr_values))
        log_snr_max = np.log10(np.max(snr_values))
        log_bin_edges = np.linspace(log_snr_min, log_snr_max, n_bins + 1)
        bin_edges = 10**log_bin_edges
        
        binned_stats = {
            'bin_centers': [],
            'mean_errors': [],
            'std_errors': [],
            'n_samples': []
        }
        
        for i in range(n_bins):
            mask = (snr_values >= bin_edges[i]) & (snr_values < bin_edges[i+1])
            if i == n_bins - 1:  # Include upper edge in last bin
                mask = (snr_values >= bin_edges[i]) & (snr_values <= bin_edges[i+1])
            
            bin_errors = error_values[mask]
            bin_snrs = snr_values[mask]
            
            if len(bin_errors) > 0:
                binned_stats['bin_centers'].append(float(np.mean(bin_snrs)))
                binned_stats['mean_errors'].append(float(np.mean(bin_errors)))
                binned_stats['std_errors'].append(float(np.std(bin_errors)))
                binned_stats['n_samples'].append(len(bin_errors))
            else:
                binned_stats['bin_centers'].append(float(np.sqrt(bin_edges[i] * bin_edges[i+1])))
                binned_stats['mean_errors'].append(np.nan)
                binned_stats['std_errors'].append(np.nan)
                binned_stats['n_samples'].append(0)
        
        return binned_stats
    
    def _generate_noise_plots(self, characterization_results: Dict[str, Any]) -> Dict[str, Path]:
        """Generate noise characterization plots."""
        plot_paths = {}
        
        # Centroiding error vs SNR plot
        if 'centroiding_vs_snr' in characterization_results:
            plot_paths['centroiding_vs_snr'] = self._plot_centroiding_vs_snr(
                characterization_results['centroiding_vs_snr']
            )
        
        # Noise statistics plots
        if 'noise_statistics' in characterization_results:
            plot_paths['noise_statistics'] = self._plot_noise_statistics(
                characterization_results['noise_statistics']
            )
        
        # Sensitivity degradation plots
        if 'sensitivity_degradation' in characterization_results:
            plot_paths['sensitivity_degradation'] = self._plot_sensitivity_degradation(
                characterization_results['sensitivity_degradation']
            )
        
        return plot_paths
    
    def _plot_centroiding_vs_snr(self, analysis_results: Dict[str, Any]) -> Path:
        """Plot centroiding error vs SNR relationship."""
        output_path = self.config.output_dir / "centroiding_error_vs_snr.png"
        
        if 'binned_statistics' not in analysis_results:
            return output_path
        
        binned_stats = analysis_results['binned_statistics']
        snr_centers = np.array(binned_stats['bin_centers'])
        mean_errors = np.array(binned_stats['mean_errors'])
        std_errors = np.array(binned_stats['std_errors'])
        
        # Filter valid bins
        valid_mask = np.isfinite(mean_errors) & np.isfinite(std_errors)
        
        plt.figure(figsize=(10, 6))
        plt.loglog(snr_centers[valid_mask], mean_errors[valid_mask], 'bo-', 
                  markersize=6, linewidth=2, label='Measured')
        plt.errorbar(snr_centers[valid_mask], mean_errors[valid_mask], 
                    yerr=std_errors[valid_mask], fmt='none', capsize=3, alpha=0.7)
        
        # Plot theoretical scaling if fit available
        if 'scaling_analysis' in analysis_results:
            scaling = analysis_results['scaling_analysis'].get('scaling_fit', {})
            if 'amplitude' in scaling and 'exponent' in scaling:
                A, B = scaling['amplitude'], scaling['exponent']
                theoretical_errors = A / (snr_centers[valid_mask]**B)
                plt.loglog(snr_centers[valid_mask], theoretical_errors, 'r--', 
                          linewidth=2, label=f'Fitted: {A:.3f} / SNR^{B:.2f}')
        
        # Plot ideal 1/SNR scaling
        ideal_errors = mean_errors[valid_mask][0] * snr_centers[0] / snr_centers[valid_mask]
        plt.loglog(snr_centers[valid_mask], ideal_errors, 'g:', 
                  linewidth=2, label='Ideal 1/SNR scaling')
        
        plt.xlabel('Signal-to-Noise Ratio')
        plt.ylabel('Centroiding Error (pixels)')
        plt.title('Centroiding Error vs SNR')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_noise_statistics(self, noise_statistics: Dict[str, Dict[str, Any]]) -> Path:
        """Plot noise statistics validation."""
        output_path = self.config.output_dir / "noise_statistics_validation.png"
        
        noise_types = list(noise_statistics.keys())
        if not noise_types:
            return output_path
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Measured vs theoretical standard deviation
        measured_stds = [noise_statistics[nt]['std'] for nt in noise_types]
        theoretical_stds = [noise_statistics[nt]['theoretical_std'] for nt in noise_types]
        
        ax1.scatter(theoretical_stds, measured_stds, s=100, alpha=0.7)
        for i, nt in enumerate(noise_types):
            ax1.annotate(nt, (theoretical_stds[i], measured_stds[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Plot perfect agreement line
        min_std = min(min(measured_stds), min(theoretical_stds))
        max_std = max(max(measured_stds), max(theoretical_stds))
        ax1.plot([min_std, max_std], [min_std, max_std], 'r--', label='Perfect Agreement')
        ax1.set_xlabel('Theoretical Std (electrons)')
        ax1.set_ylabel('Measured Std (electrons)')
        ax1.set_title('Measured vs Theoretical Noise')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Normality test results
        normality_p_values = [noise_statistics[nt]['normality_test_p_value'] for nt in noise_types]
        colors = ['green' if p > 0.05 else 'red' for p in normality_p_values]
        
        ax2.bar(range(len(noise_types)), normality_p_values, color=colors, alpha=0.7)
        ax2.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        ax2.set_xlabel('Noise Type')
        ax2.set_ylabel('Normality Test p-value')
        ax2.set_title('Gaussian Noise Validation')
        ax2.set_xticks(range(len(noise_types)))
        ax2.set_xticklabels(noise_types, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Skewness and kurtosis
        skewness_values = [noise_statistics[nt]['skewness'] for nt in noise_types]
        kurtosis_values = [noise_statistics[nt]['kurtosis'] for nt in noise_types]
        
        x_pos = np.arange(len(noise_types))
        width = 0.35
        
        ax3.bar(x_pos - width/2, skewness_values, width, label='Skewness', alpha=0.7)
        ax3.bar(x_pos + width/2, kurtosis_values, width, label='Kurtosis', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Noise Type')
        ax3.set_ylabel('Statistical Moment')
        ax3.set_title('Noise Distribution Shape')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(noise_types, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Ratio of measured to theoretical
        ratios = [noise_statistics[nt]['measured_vs_theoretical_ratio'] for nt in noise_types]
        
        ax4.bar(range(len(noise_types)), ratios, alpha=0.7)
        ax4.axhline(y=1.0, color='red', linestyle='--', label='Perfect Agreement')
        ax4.set_xlabel('Noise Type')
        ax4.set_ylabel('Measured / Theoretical Ratio')
        ax4.set_title('Noise Model Accuracy')
        ax4.set_xticks(range(len(noise_types)))
        ax4.set_xticklabels(noise_types, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_sensitivity_degradation(self, degradation_analysis: Dict[str, Any]) -> Path:
        """Plot sensitivity degradation analysis."""
        output_path = self.config.output_dir / "sensitivity_degradation.png"
        
        if 'degradation_data' not in degradation_analysis:
            return output_path
        
        degradation_data = degradation_analysis['degradation_data']
        noise_levels = list(degradation_data.keys())
        
        if not noise_levels:
            return output_path
        
        # Get magnitude values from first noise level
        magnitudes = list(degradation_data[noise_levels[0]].keys())
        
        plt.figure(figsize=(12, 8))
        
        # Plot centroiding error vs magnitude for different noise levels
        for noise_level in noise_levels:
            centroid_errors = []
            mags = []
            
            for mag in magnitudes:
                if mag in degradation_data[noise_level]:
                    error = degradation_data[noise_level][mag]['mean_centroid_error']
                    if np.isfinite(error):
                        centroid_errors.append(error)
                        mags.append(mag)
            
            if mags:
                plt.semilogy(mags, centroid_errors, 'o-', linewidth=2, 
                           markersize=6, label=noise_level)
        
        plt.xlabel('Star Magnitude')
        plt.ylabel('Mean Centroiding Error (pixels)')
        plt.title('Centroiding Performance vs Noise Level')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _save_noise_characterization_data(self, characterization_results: Dict[str, Any]) -> Path:
        """Save detailed noise characterization data."""
        output_path = self.config.output_dir / "noise_characterization_data.csv"
        
        # Extract data for CSV export
        export_data = []
        
        # Add centroiding vs SNR data if available
        if 'centroiding_vs_snr' in characterization_results:
            binned_stats = characterization_results['centroiding_vs_snr'].get('binned_statistics', {})
            if 'bin_centers' in binned_stats:
                for i, (snr, error, std, n_samples) in enumerate(zip(
                    binned_stats['bin_centers'],
                    binned_stats['mean_errors'],
                    binned_stats['std_errors'],
                    binned_stats['n_samples']
                )):
                    export_data.append({
                        'analysis_type': 'centroiding_vs_snr',
                        'snr': snr,
                        'mean_centroid_error': error,
                        'std_centroid_error': std,
                        'n_samples': n_samples,
                        'bin_index': i
                    })
        
        if export_data:
            df = pd.DataFrame(export_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved noise characterization data: {output_path}")
        
        return output_path

# Export main class
__all__ = ['NoiseValidator', 'NoiseConfig', 'NoiseMeasurement', 'NoiseStatistics']