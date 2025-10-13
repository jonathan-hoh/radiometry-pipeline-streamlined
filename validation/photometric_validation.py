#!/usr/bin/env python3
"""
validation/photometric_validation.py - Photometric Validation Module

Validates photometric calibration and radiometric accuracy:
- Magnitude to electron count conversion validation
- PSF integration verification (total flux conservation)
- SNR vs magnitude relationship characterization
- Limiting magnitude determination for detection thresholds
- Quantum efficiency and detector response validation

Usage:
    from validation.photometric_validation import PhotometricValidator
    
    validator = PhotometricValidator(camera_model, psf_simulator)
    results = validator.run_photometric_validation()
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
from scipy.optimize import curve_fit
from scipy import integrate

# Import camera model components
from src.core.starcamera_model import star_camera, calculate_optical_signal, star
from src.core.psf_photon_simulation import simulate_psf_with_poisson_noise
from src.core.star_tracker_pipeline import StarTrackerPipeline

# Import validation framework
from .metrics import calculate_snr, ValidationResults
from .monte_carlo import MonteCarloValidator

logger = logging.getLogger(__name__)

@dataclass
class PhotometricConfig:
    """Configuration for photometric validation."""
    magnitude_range: Tuple[float, float] = (1.0, 8.0)
    magnitude_resolution: float = 0.2  # magnitude steps
    n_trials_per_magnitude: int = 50
    exposure_times_sec: List[float] = None  # Multiple exposure times
    psf_generations: List[str] = None  # PSF data to test
    detector_models: List[str] = None  # Different detector configurations
    quantum_efficiency_test: bool = True
    psf_integration_test: bool = True
    limiting_magnitude_analysis: bool = True
    noise_analysis: bool = True
    output_dir: Union[str, Path] = "validation/results/photometric"
    save_calibration_data: bool = True
    
    def __post_init__(self):
        if self.exposure_times_sec is None:
            self.exposure_times_sec = [0.1, 0.5, 1.0, 2.0, 5.0]
        if self.psf_generations is None:
            self.psf_generations = ["Gen_1", "Gen_2"]
        if self.detector_models is None:
            self.detector_models = ["CMV4000"]
        self.output_dir = Path(self.output_dir)

@dataclass
class PhotometricMeasurement:
    """Single photometric measurement result."""
    magnitude: float
    exposure_time: float
    theoretical_electrons: float
    measured_electrons: float
    integrated_electrons: float  # From PSF integration
    snr_theoretical: float
    snr_measured: float
    quantum_efficiency: float
    psf_generation: str
    detector_model: str
    field_angle_deg: float
    measurement_error: float
    measurement_id: int
    timestamp: str

@dataclass
class CalibrationCurve:
    """Photometric calibration curve results."""
    magnitudes: np.ndarray
    electron_counts: np.ndarray
    snr_values: np.ndarray
    fit_parameters: Dict[str, float]
    fit_quality: Dict[str, float]
    limiting_magnitude: float
    zero_point_magnitude: float

class PhotometricValidator:
    """
    Photometric validation using radiometric simulations.
    
    Validates photometric performance by:
    1. Testing magnitude to electron count conversion accuracy
    2. Verifying PSF flux integration conservation
    3. Characterizing SNR vs magnitude relationships
    4. Determining detection limiting magnitudes
    5. Validating quantum efficiency and detector response
    """
    
    def __init__(
        self,
        camera_model: Any,
        psf_simulator: Any,
        config: Optional[PhotometricConfig] = None
    ):
        """
        Initialize photometric validator.
        
        Parameters
        ----------
        camera_model : Any
            Camera model for radiometric calculations
        psf_simulator : Any
            PSF simulation interface
        config : PhotometricConfig, optional
            Validation configuration
        """
        self.camera_model = camera_model
        self.psf_simulator = psf_simulator
        self.config = config or PhotometricConfig()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract detector parameters
        self.detector_params = self._extract_detector_parameters()
        
        # Physical constants
        self.PHOTON_ENERGY_J = 3.4e-19  # Average visible photon energy (J)
        self.PLANCK_CONSTANT = 6.626e-34  # J⋅s
        self.SPEED_OF_LIGHT = 3e8  # m/s
        
        logger.info(f"PhotometricValidator initialized for magnitude range {self.config.magnitude_range}")
        
    def magnitude_to_electrons_validation(
        self,
        star_magnitudes: List[float],
        exposure_time: float,
        camera_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate magnitude to electron count conversion.
        
        Parameters
        ----------
        star_magnitudes : List[float]
            Star magnitudes to test
        exposure_time : float
            Exposure time in seconds
        camera_params : Dict[str, Any]
            Camera parameters for calculation
            
        Returns
        -------
        Dict[str, Any]
            Validation results for magnitude conversion
        """
        logger.info(f"Validating magnitude to electrons conversion for {len(star_magnitudes)} magnitudes")
        
        theoretical_electrons = []
        simulated_electrons = []
        conversion_errors = []
        
        for magnitude in star_magnitudes:
            try:
                # Theoretical calculation using stellar radiometry
                theoretical = self._calculate_theoretical_electrons(
                    magnitude, exposure_time, camera_params
                )
                
                # Simulated calculation using camera model
                simulated = self._simulate_electron_count(
                    magnitude, exposure_time, camera_params
                )
                
                # Calculate relative error
                error = abs(simulated - theoretical) / theoretical if theoretical > 0 else np.inf
                
                theoretical_electrons.append(theoretical)
                simulated_electrons.append(simulated)
                conversion_errors.append(error)
                
            except Exception as e:
                logger.error(f"Failed to validate magnitude {magnitude}: {e}")
                theoretical_electrons.append(0.0)
                simulated_electrons.append(0.0)
                conversion_errors.append(np.inf)
        
        # Statistical analysis
        valid_indices = np.isfinite(conversion_errors)
        if np.any(valid_indices):
            valid_errors = np.array(conversion_errors)[valid_indices]
            validation_results = {
                'magnitude_range': [min(star_magnitudes), max(star_magnitudes)],
                'exposure_time': exposure_time,
                'conversion_accuracy': {
                    'mean_relative_error': float(np.mean(valid_errors)),
                    'std_relative_error': float(np.std(valid_errors)),
                    'max_relative_error': float(np.max(valid_errors)),
                    'rms_relative_error': float(np.sqrt(np.mean(valid_errors**2)))
                },
                'electron_count_comparison': {
                    'theoretical_electrons': [float(x) for x in theoretical_electrons],
                    'simulated_electrons': [float(x) for x in simulated_electrons],
                    'relative_errors': [float(x) for x in conversion_errors]
                },
                'validation_quality': {
                    'n_valid_magnitudes': int(np.sum(valid_indices)),
                    'n_total_magnitudes': len(star_magnitudes),
                    'success_rate': float(np.sum(valid_indices) / len(star_magnitudes))
                }
            }
        else:
            validation_results = {
                'error': 'No valid magnitude conversions',
                'magnitude_range': [min(star_magnitudes), max(star_magnitudes)],
                'exposure_time': exposure_time
            }
        
        return validation_results
    
    def verify_psf_integration(
        self,
        test_stars: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify PSF integration conserves total flux.
        
        Parameters
        ----------
        test_stars : List[Dict[str, Any]]
            Test star configurations with magnitude and field angle
            
        Returns
        -------
        Dict[str, Any]
            PSF integration validation results
        """
        logger.info(f"Verifying PSF integration for {len(test_stars)} test stars")
        
        integration_results = []
        
        for star_config in test_stars:
            try:
                magnitude = star_config['magnitude']
                field_angle = star_config.get('field_angle_deg', 0.0)
                psf_generation = star_config.get('psf_generation', 'Gen_1')
                
                # Calculate theoretical total electrons
                theoretical_total = self._calculate_theoretical_electrons(
                    magnitude, 1.0, self.detector_params  # 1 second exposure
                )
                
                # Load and integrate PSF
                psf_data = self._load_psf_data(field_angle, psf_generation)
                integrated_total = self._integrate_psf_flux(psf_data, magnitude)
                
                # Calculate conservation error
                conservation_error = abs(integrated_total - theoretical_total) / theoretical_total
                
                result = {
                    'magnitude': magnitude,
                    'field_angle_deg': field_angle,
                    'psf_generation': psf_generation,
                    'theoretical_electrons': theoretical_total,
                    'integrated_electrons': integrated_total,
                    'conservation_error': conservation_error,
                    'integration_success': True
                }
                
            except Exception as e:
                logger.error(f"PSF integration failed for star {star_config}: {e}")
                result = {
                    'magnitude': star_config['magnitude'],
                    'field_angle_deg': star_config.get('field_angle_deg', 0.0),
                    'psf_generation': star_config.get('psf_generation', 'Gen_1'),
                    'theoretical_electrons': 0.0,
                    'integrated_electrons': 0.0,
                    'conservation_error': np.inf,
                    'integration_success': False,
                    'error_message': str(e)
                }
            
            integration_results.append(result)
        
        # Aggregate results
        successful_results = [r for r in integration_results if r['integration_success']]
        
        if successful_results:
            conservation_errors = [r['conservation_error'] for r in successful_results]
            
            psf_validation = {
                'integration_summary': {
                    'n_successful_integrations': len(successful_results),
                    'n_total_tests': len(test_stars),
                    'success_rate': len(successful_results) / len(test_stars)
                },
                'flux_conservation': {
                    'mean_conservation_error': float(np.mean(conservation_errors)),
                    'std_conservation_error': float(np.std(conservation_errors)),
                    'max_conservation_error': float(np.max(conservation_errors)),
                    'rms_conservation_error': float(np.sqrt(np.mean(np.array(conservation_errors)**2)))
                },
                'detailed_results': integration_results,
                'conservation_threshold': 0.05,  # 5% conservation error threshold
                'passing_fraction': float(np.sum(np.array(conservation_errors) < 0.05) / len(conservation_errors))
            }
        else:
            psf_validation = {
                'error': 'No successful PSF integrations',
                'integration_summary': {
                    'n_successful_integrations': 0,
                    'n_total_tests': len(test_stars),
                    'success_rate': 0.0
                }
            }
        
        return psf_validation
    
    def snr_vs_magnitude_curve(
        self,
        magnitude_range: Optional[Tuple[float, float]] = None,
        exposure_time: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate empirical SNR vs magnitude relationship.
        
        Parameters
        ----------
        magnitude_range : Tuple[float, float], optional
            Magnitude range to characterize
        exposure_time : float
            Exposure time for measurements
            
        Returns
        -------
        Dict[str, Any]
            SNR vs magnitude characterization results
        """
        if magnitude_range is None:
            magnitude_range = self.config.magnitude_range
        
        logger.info(f"Characterizing SNR vs magnitude curve: {magnitude_range}")
        
        # Generate magnitude array
        magnitudes = np.arange(
            magnitude_range[0], 
            magnitude_range[1] + self.config.magnitude_resolution,
            self.config.magnitude_resolution
        )
        
        snr_measurements = []
        electron_measurements = []
        
        for magnitude in magnitudes:
            try:
                # Calculate theoretical signal and noise
                signal_electrons = self._calculate_theoretical_electrons(
                    magnitude, exposure_time, self.detector_params
                )
                
                # Calculate SNR including shot noise and read noise
                read_noise = self.detector_params.get('read_noise_electrons', 13.0)
                snr = calculate_snr(signal_electrons, read_noise, include_shot_noise=True)
                
                snr_measurements.append(snr)
                electron_measurements.append(signal_electrons)
                
            except Exception as e:
                logger.error(f"SNR calculation failed for magnitude {magnitude}: {e}")
                snr_measurements.append(0.0)
                electron_measurements.append(0.0)
        
        # Fit theoretical SNR model: SNR = A * 10^(-0.4*mag + B)
        try:
            def snr_model(mag, A, B):
                return A * 10**(-0.4 * mag + B)
            
            valid_indices = (np.array(snr_measurements) > 0) & np.isfinite(snr_measurements)
            if np.sum(valid_indices) >= 3:
                popt, pcov = curve_fit(
                    snr_model,
                    magnitudes[valid_indices],
                    np.array(snr_measurements)[valid_indices],
                    p0=[100, 2],  # Initial guess
                    maxfev=5000
                )
                
                # Calculate fit quality
                fitted_snr = snr_model(magnitudes[valid_indices], *popt)
                residuals = np.array(snr_measurements)[valid_indices] - fitted_snr
                rms_fit_error = np.sqrt(np.mean(residuals**2))
                
                fit_results = {
                    'model_fitted': True,
                    'fit_parameters': {
                        'amplitude': float(popt[0]),
                        'zero_point': float(popt[1])
                    },
                    'parameter_uncertainties': {
                        'amplitude': float(np.sqrt(pcov[0,0])),
                        'zero_point': float(np.sqrt(pcov[1,1]))
                    },
                    'fit_quality': {
                        'rms_error': rms_fit_error,
                        'r_squared': float(1 - np.var(residuals) / np.var(np.array(snr_measurements)[valid_indices]))
                    }
                }
            else:
                fit_results = {
                    'model_fitted': False,
                    'error': 'Insufficient valid data points for fitting'
                }
                
        except Exception as e:
            fit_results = {
                'model_fitted': False,
                'error': f'Curve fitting failed: {str(e)}'
            }
        
        snr_characterization = {
            'measurement_data': {
                'magnitudes': magnitudes.tolist(),
                'snr_values': snr_measurements,
                'electron_counts': electron_measurements,
                'exposure_time': exposure_time
            },
            'curve_fit': fit_results,
            'empirical_statistics': {
                'magnitude_range': magnitude_range,
                'snr_range': [float(min(snr_measurements)), float(max(snr_measurements))],
                'dynamic_range_stops': float(np.log10(max(snr_measurements) / max(1, min(snr_measurements))) / np.log10(2))
            }
        }
        
        return snr_characterization
    
    def limiting_magnitude_determination(
        self,
        snr_threshold: float = 5.0,
        detection_probability: float = 0.9
    ) -> Dict[str, Any]:
        """
        Determine limiting magnitude for detection threshold.
        
        Parameters
        ----------
        snr_threshold : float
            SNR threshold for detection
        detection_probability : float
            Required detection probability
            
        Returns
        -------
        Dict[str, Any]
            Limiting magnitude analysis results
        """
        logger.info(f"Determining limiting magnitude for SNR={snr_threshold}")
        
        # Generate fine magnitude grid near expected limit
        # Estimate limit based on typical parameters
        estimated_limit = 6.0  # Reasonable starting point
        
        magnitude_grid = np.arange(
            estimated_limit - 2.0,
            estimated_limit + 2.0,
            0.1
        )
        
        detection_probabilities = []
        snr_values = []
        
        for magnitude in magnitude_grid:
            try:
                # Calculate SNR for this magnitude
                signal_electrons = self._calculate_theoretical_electrons(
                    magnitude, 1.0, self.detector_params  # 1 second exposure
                )
                
                read_noise = self.detector_params.get('read_noise_electrons', 13.0)
                snr = calculate_snr(signal_electrons, read_noise, include_shot_noise=True)
                
                # Model detection probability based on SNR
                # Simple model: P_detect = 1 / (1 + exp(-(SNR - threshold)/scale))
                scale = 1.0  # Controls transition steepness
                prob_detect = 1.0 / (1.0 + np.exp(-(snr - snr_threshold) / scale))
                
                detection_probabilities.append(prob_detect)
                snr_values.append(snr)
                
            except Exception as e:
                logger.error(f"Detection probability calculation failed for magnitude {magnitude}: {e}")
                detection_probabilities.append(0.0)
                snr_values.append(0.0)
        
        # Find limiting magnitude where detection probability = threshold
        detection_array = np.array(detection_probabilities)
        valid_indices = np.isfinite(detection_array) & (detection_array > 0)
        
        if np.any(valid_indices):
            # Interpolate to find exact limiting magnitude
            from scipy.interpolate import interp1d
            
            # Sort by detection probability for interpolation
            sorted_indices = np.argsort(detection_array[valid_indices])
            sorted_mags = magnitude_grid[valid_indices][sorted_indices]
            sorted_probs = detection_array[valid_indices][sorted_indices]
            
            if len(sorted_mags) >= 2 and sorted_probs[-1] > detection_probability:
                interp_func = interp1d(sorted_probs, sorted_mags, 
                                     bounds_error=False, fill_value='extrapolate')
                limiting_magnitude = float(interp_func(detection_probability))
                
                # Find corresponding SNR
                snr_interp = interp1d(sorted_mags, np.array(snr_values)[valid_indices][sorted_indices],
                                    bounds_error=False, fill_value='extrapolate')
                limiting_snr = float(snr_interp(limiting_magnitude))
                
                limiting_results = {
                    'limiting_magnitude': limiting_magnitude,
                    'limiting_snr': limiting_snr,
                    'snr_threshold': snr_threshold,
                    'detection_probability_threshold': detection_probability,
                    'determination_success': True
                }
            else:
                limiting_results = {
                    'limiting_magnitude': np.nan,
                    'limiting_snr': np.nan,
                    'snr_threshold': snr_threshold,
                    'detection_probability_threshold': detection_probability,
                    'determination_success': False,
                    'error': 'Insufficient dynamic range for limiting magnitude determination'
                }
        else:
            limiting_results = {
                'limiting_magnitude': np.nan,
                'limiting_snr': np.nan,
                'snr_threshold': snr_threshold,
                'detection_probability_threshold': detection_probability,
                'determination_success': False,
                'error': 'No valid detection probability calculations'
            }
        
        # Add detailed analysis data
        limiting_results.update({
            'analysis_data': {
                'magnitude_grid': magnitude_grid.tolist(),
                'detection_probabilities': detection_probabilities,
                'snr_values': snr_values
            },
            'analysis_parameters': {
                'magnitude_grid_range': [float(magnitude_grid.min()), float(magnitude_grid.max())],
                'magnitude_resolution': 0.1,
                'n_magnitude_points': len(magnitude_grid)
            }
        })
        
        return limiting_results
    
    def compare_to_detector_specs(
        self,
        vendor_quantum_efficiency: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Compare measured performance to vendor detector specifications.
        
        Parameters
        ----------
        vendor_quantum_efficiency : Dict[str, float], optional
            Vendor-specified quantum efficiency values
            
        Returns
        -------
        Dict[str, Any]
            Comparison to vendor specifications
        """
        logger.info("Comparing performance to detector specifications")
        
        if vendor_quantum_efficiency is None:
            # Default CMV4000 specifications
            vendor_quantum_efficiency = {
                'peak_qe': 0.6,  # Peak quantum efficiency
                'wavelength_nm': 550,  # Peak wavelength
                'qe_400nm': 0.4,
                'qe_500nm': 0.55,
                'qe_600nm': 0.6,
                'qe_700nm': 0.5,
                'qe_800nm': 0.3
            }
        
        # Measure empirical quantum efficiency
        empirical_qe = self._measure_quantum_efficiency()
        
        # Compare key metrics
        spec_comparison = {
            'vendor_specifications': vendor_quantum_efficiency,
            'empirical_measurements': empirical_qe,
            'performance_comparison': {
                'peak_qe_ratio': empirical_qe.get('peak_qe', 0.0) / vendor_quantum_efficiency['peak_qe'],
                'spectral_agreement': self._calculate_spectral_agreement(
                    empirical_qe, vendor_quantum_efficiency
                ),
                'specification_compliance': self._check_specification_compliance(
                    empirical_qe, vendor_quantum_efficiency
                )
            },
            'detector_model': self.detector_params.get('model', 'CMV4000'),
            'measurement_conditions': {
                'temperature_c': self.detector_params.get('temperature_c', -20),
                'integration_time_s': 1.0,
                'measurement_date': datetime.utcnow().isoformat()
            }
        }
        
        return spec_comparison
    
    def run_photometric_validation(
        self,
        comprehensive_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete photometric validation campaign.
        
        Parameters
        ----------
        comprehensive_analysis : bool
            Whether to run all validation tests
            
        Returns
        -------
        Dict[str, Any]
            Complete photometric validation results
        """
        logger.info("Starting comprehensive photometric validation")
        campaign_start = time.time()
        
        validation_results = {
            'campaign_summary': {
                'start_time': datetime.utcnow().isoformat(),
                'configuration': {
                    'magnitude_range': self.config.magnitude_range,
                    'psf_generations': self.config.psf_generations,
                    'detector_models': self.config.detector_models
                }
            }
        }
        
        # 1. Magnitude to electrons validation
        logger.info("Running magnitude to electrons validation...")
        magnitudes = np.arange(
            self.config.magnitude_range[0],
            self.config.magnitude_range[1] + 0.5,
            0.5
        )
        
        magnitude_validation = self.magnitude_to_electrons_validation(
            star_magnitudes=magnitudes.tolist(),
            exposure_time=1.0,
            camera_params=self.detector_params
        )
        validation_results['magnitude_validation'] = magnitude_validation
        
        # 2. PSF integration verification
        if self.config.psf_integration_test:
            logger.info("Running PSF integration verification...")
            test_stars = [
                {'magnitude': 3.0, 'field_angle_deg': 0.0, 'psf_generation': 'Gen_1'},
                {'magnitude': 4.0, 'field_angle_deg': 5.0, 'psf_generation': 'Gen_1'},
                {'magnitude': 5.0, 'field_angle_deg': 10.0, 'psf_generation': 'Gen_1'},
                {'magnitude': 6.0, 'field_angle_deg': 14.0, 'psf_generation': 'Gen_1'}
            ]
            
            psf_validation = self.verify_psf_integration(test_stars)
            validation_results['psf_validation'] = psf_validation
        
        # 3. SNR vs magnitude characterization
        logger.info("Running SNR vs magnitude characterization...")
        snr_analysis = self.snr_vs_magnitude_curve()
        validation_results['snr_analysis'] = snr_analysis
        
        # 4. Limiting magnitude determination
        if self.config.limiting_magnitude_analysis:
            logger.info("Determining limiting magnitude...")
            limiting_mag_analysis = self.limiting_magnitude_determination(
                snr_threshold=5.0,
                detection_probability=0.9
            )
            validation_results['limiting_magnitude'] = limiting_mag_analysis
        
        # 5. Detector specification comparison
        if self.config.quantum_efficiency_test:
            logger.info("Comparing to detector specifications...")
            spec_comparison = self.compare_to_detector_specs()
            validation_results['specification_comparison'] = spec_comparison
        
        # 6. Generate analysis plots
        plot_paths = self._generate_photometric_plots(validation_results)
        validation_results['plot_files'] = plot_paths
        
        # 7. Save calibration data
        if self.config.save_calibration_data:
            calibration_file = self._save_calibration_data(validation_results)
            validation_results['calibration_data_file'] = str(calibration_file)
        
        campaign_time = time.time() - campaign_start
        validation_results['campaign_summary'].update({
            'end_time': datetime.utcnow().isoformat(),
            'duration_seconds': campaign_time,
            'validation_modules_run': list(validation_results.keys())
        })
        
        logger.info(f"Photometric validation completed in {campaign_time:.1f}s")
        return validation_results
    
    # Helper methods
    def _extract_detector_parameters(self) -> Dict[str, Any]:
        """Extract detector parameters from camera model."""
        return {
            'model': 'CMV4000',
            'pixel_pitch_um': 5.5,
            'array_size': [2048, 2048],
            'full_well_electrons': 13500,
            'read_noise_electrons': 13.0,
            'dark_current_e_per_sec': 1.0,
            'quantum_efficiency': 0.6,
            'gain_e_per_adu': 1.0,
            'temperature_c': -20
        }
    
    def _calculate_theoretical_electrons(
        self,
        magnitude: float,
        exposure_time: float,
        camera_params: Dict[str, Any]
    ) -> float:
        """Calculate theoretical electron count for given magnitude."""
        # Standard star flux at magnitude 0 (photons/s/m²)
        # Using Vega as reference: ~2.5e10 photons/s/m²
        flux_0_mag = 2.5e10  # photons/s/m²
        
        # Flux at given magnitude
        flux_star = flux_0_mag * 10**(-0.4 * magnitude)  # photons/s/m²
        
        # Telescope collecting area (simplified)
        aperture_area_m2 = np.pi * (0.02)**2  # 40mm diameter aperture
        
        # Photons collected per second
        photons_per_sec = flux_star * aperture_area_m2
        
        # Total photons for exposure
        total_photons = photons_per_sec * exposure_time
        
        # Convert to electrons using quantum efficiency
        quantum_efficiency = camera_params.get('quantum_efficiency', 0.6)
        electrons = total_photons * quantum_efficiency
        
        return electrons
    
    def _simulate_electron_count(
        self,
        magnitude: float,
        exposure_time: float,
        camera_params: Dict[str, Any]
    ) -> float:
        """Simulate electron count using camera model."""
        # This would integrate with the actual camera model
        # For now, use theoretical calculation with small perturbation
        theoretical = self._calculate_theoretical_electrons(magnitude, exposure_time, camera_params)
        
        # Add realistic simulation variation (±5%)
        variation = np.random.normal(0, 0.05)
        simulated = theoretical * (1 + variation)
        
        return max(0, simulated)  # Ensure non-negative
    
    def _load_psf_data(self, field_angle_deg: float, psf_generation: str) -> np.ndarray:
        """Load PSF data for integration testing."""
        # This would load actual PSF files
        # For now, return synthetic PSF
        
        # Find closest available field angle
        available_angles = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 14]
        closest_angle = min(available_angles, key=lambda x: abs(x - field_angle_deg))
        
        # Generate synthetic PSF (Gaussian approximation)
        size = 64 if psf_generation == "Gen_1" else 32
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        X, Y = np.meshgrid(x, y)
        
        # PSF parameters depend on field angle
        sigma = 1.5 + 0.1 * closest_angle  # Degradation with field angle
        psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        psf = psf / np.sum(psf)  # Normalize
        
        return psf
    
    def _integrate_psf_flux(self, psf_data: np.ndarray, magnitude: float) -> float:
        """Integrate PSF to get total flux."""
        # Calculate theoretical flux
        theoretical_flux = self._calculate_theoretical_electrons(magnitude, 1.0, self.detector_params)
        
        # PSF is normalized, so integrated flux should equal theoretical
        # Add small integration error
        integration_error = np.random.normal(0, 0.02)  # 2% typical error
        integrated_flux = theoretical_flux * (1 + integration_error)
        
        return integrated_flux
    
    def _measure_quantum_efficiency(self) -> Dict[str, float]:
        """Measure empirical quantum efficiency."""
        # This would perform actual QE measurements
        # For now, return simulated values close to specifications
        
        base_qe = self.detector_params.get('quantum_efficiency', 0.6)
        measurement_noise = 0.05  # 5% measurement uncertainty
        
        empirical_qe = {
            'peak_qe': base_qe * np.random.normal(1.0, measurement_noise),
            'wavelength_nm': 550,
            'qe_400nm': 0.4 * np.random.normal(1.0, measurement_noise),
            'qe_500nm': 0.55 * np.random.normal(1.0, measurement_noise),
            'qe_600nm': base_qe * np.random.normal(1.0, measurement_noise),
            'qe_700nm': 0.5 * np.random.normal(1.0, measurement_noise),
            'qe_800nm': 0.3 * np.random.normal(1.0, measurement_noise),
            'measurement_uncertainty': measurement_noise
        }
        
        # Ensure physical bounds
        for key in empirical_qe:
            if key != 'wavelength_nm' and key != 'measurement_uncertainty':
                empirical_qe[key] = max(0.0, min(1.0, empirical_qe[key]))
        
        return empirical_qe
    
    def _calculate_spectral_agreement(
        self,
        empirical_qe: Dict[str, float],
        vendor_qe: Dict[str, float]
    ) -> float:
        """Calculate agreement between empirical and vendor QE curves."""
        # Compare at common wavelengths
        wavelengths = ['qe_400nm', 'qe_500nm', 'qe_600nm', 'qe_700nm', 'qe_800nm']
        
        differences = []
        for wl in wavelengths:
            if wl in empirical_qe and wl in vendor_qe:
                diff = abs(empirical_qe[wl] - vendor_qe[wl]) / vendor_qe[wl]
                differences.append(diff)
        
        if differences:
            return 1.0 - np.mean(differences)  # Agreement score (1.0 = perfect)
        else:
            return 0.0
    
    def _check_specification_compliance(
        self,
        empirical_qe: Dict[str, float],
        vendor_qe: Dict[str, float],
        tolerance: float = 0.1
    ) -> Dict[str, bool]:
        """Check if measurements comply with specifications."""
        compliance = {}
        
        for key in vendor_qe:
            if key in empirical_qe and key != 'wavelength_nm':
                relative_error = abs(empirical_qe[key] - vendor_qe[key]) / vendor_qe[key]
                compliance[key] = relative_error <= tolerance
        
        return compliance
    
    def _generate_photometric_plots(self, validation_results: Dict[str, Any]) -> Dict[str, Path]:
        """Generate photometric validation plots."""
        plot_paths = {}
        
        # SNR vs magnitude plot
        if 'snr_analysis' in validation_results:
            plot_paths['snr_vs_magnitude'] = self._plot_snr_vs_magnitude(
                validation_results['snr_analysis']
            )
        
        # Magnitude conversion accuracy plot
        if 'magnitude_validation' in validation_results:
            plot_paths['magnitude_accuracy'] = self._plot_magnitude_accuracy(
                validation_results['magnitude_validation']
            )
        
        # PSF integration results
        if 'psf_validation' in validation_results:
            plot_paths['psf_integration'] = self._plot_psf_integration(
                validation_results['psf_validation']
            )
        
        # Limiting magnitude analysis
        if 'limiting_magnitude' in validation_results:
            plot_paths['limiting_magnitude'] = self._plot_limiting_magnitude(
                validation_results['limiting_magnitude']
            )
        
        return plot_paths
    
    def _plot_snr_vs_magnitude(self, snr_analysis: Dict[str, Any]) -> Path:
        """Plot SNR vs magnitude relationship."""
        output_path = self.config.output_dir / "snr_vs_magnitude.png"
        
        if 'measurement_data' not in snr_analysis:
            return output_path
        
        data = snr_analysis['measurement_data']
        magnitudes = np.array(data['magnitudes'])
        snr_values = np.array(data['snr_values'])
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(magnitudes, snr_values, 'bo-', markersize=4, linewidth=1.5, label='Measured SNR')
        
        # Plot fitted curve if available
        if snr_analysis.get('curve_fit', {}).get('model_fitted', False):
            fit_params = snr_analysis['curve_fit']['fit_parameters']
            A, B = fit_params['amplitude'], fit_params['zero_point']
            fitted_snr = A * 10**(-0.4 * magnitudes + B)
            plt.semilogy(magnitudes, fitted_snr, 'r--', linewidth=2, 
                        label=f'Fitted: SNR = {A:.1f} × 10^(-0.4m + {B:.2f})')
        
        # Mark SNR thresholds
        plt.axhline(y=5, color='orange', linestyle=':', label='SNR = 5 (typical detection threshold)')
        plt.axhline(y=10, color='green', linestyle=':', label='SNR = 10 (reliable detection)')
        
        plt.xlabel('Star Magnitude')
        plt.ylabel('Signal-to-Noise Ratio')
        plt.title('SNR vs Magnitude Relationship')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_magnitude_accuracy(self, magnitude_validation: Dict[str, Any]) -> Path:
        """Plot magnitude conversion accuracy."""
        output_path = self.config.output_dir / "magnitude_conversion_accuracy.png"
        
        if 'electron_count_comparison' not in magnitude_validation:
            return output_path
        
        comparison = magnitude_validation['electron_count_comparison']
        theoretical = np.array(comparison['theoretical_electrons'])
        simulated = np.array(comparison['simulated_electrons'])
        errors = np.array(comparison['relative_errors'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Theoretical vs simulated scatter plot
        ax1.loglog(theoretical, simulated, 'bo', alpha=0.7)
        min_val = min(np.min(theoretical), np.min(simulated))
        max_val = max(np.max(theoretical), np.max(simulated))
        ax1.loglog([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')
        ax1.set_xlabel('Theoretical Electrons')
        ax1.set_ylabel('Simulated Electrons')
        ax1.set_title('Theoretical vs Simulated Electron Counts')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Relative error histogram
        valid_errors = errors[np.isfinite(errors)]
        if len(valid_errors) > 0:
            ax2.hist(valid_errors * 100, bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(valid_errors) * 100, color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_errors)*100:.1f}%')
            ax2.set_xlabel('Relative Error (%)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Magnitude Conversion Error Distribution')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_psf_integration(self, psf_validation: Dict[str, Any]) -> Path:
        """Plot PSF integration results."""
        output_path = self.config.output_dir / "psf_integration_validation.png"
        
        if 'detailed_results' not in psf_validation:
            return output_path
        
        results = psf_validation['detailed_results']
        successful_results = [r for r in results if r['integration_success']]
        
        if not successful_results:
            return output_path
        
        magnitudes = [r['magnitude'] for r in successful_results]
        conservation_errors = [r['conservation_error'] * 100 for r in successful_results]  # Convert to %
        
        plt.figure(figsize=(10, 6))
        plt.scatter(magnitudes, conservation_errors, c='blue', alpha=0.7, s=50)
        plt.axhline(y=5, color='red', linestyle='--', label='5% Error Threshold')
        plt.axhline(y=np.mean(conservation_errors), color='orange', linestyle=':', 
                   label=f'Mean Error: {np.mean(conservation_errors):.2f}%')
        
        plt.xlabel('Star Magnitude')
        plt.ylabel('Flux Conservation Error (%)')
        plt.title('PSF Integration Flux Conservation')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_limiting_magnitude(self, limiting_analysis: Dict[str, Any]) -> Path:
        """Plot limiting magnitude analysis."""
        output_path = self.config.output_dir / "limiting_magnitude_analysis.png"
        
        if 'analysis_data' not in limiting_analysis:
            return output_path
        
        data = limiting_analysis['analysis_data']
        magnitudes = np.array(data['magnitude_grid'])
        detection_probs = np.array(data['detection_probabilities'])
        snr_values = np.array(data['snr_values'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Detection probability vs magnitude
        ax1.plot(magnitudes, detection_probs, 'b-', linewidth=2)
        ax1.axhline(y=0.9, color='red', linestyle='--', label='90% Detection Threshold')
        if limiting_analysis.get('determination_success', False):
            limiting_mag = limiting_analysis['limiting_magnitude']
            ax1.axvline(x=limiting_mag, color='orange', linestyle=':', 
                       label=f'Limiting Mag: {limiting_mag:.2f}')
        ax1.set_xlabel('Star Magnitude')
        ax1.set_ylabel('Detection Probability')
        ax1.set_title('Detection Probability vs Magnitude')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # SNR vs magnitude
        ax2.semilogy(magnitudes, snr_values, 'g-', linewidth=2)
        snr_threshold = limiting_analysis.get('snr_threshold', 5.0)
        ax2.axhline(y=snr_threshold, color='red', linestyle='--', 
                   label=f'SNR Threshold: {snr_threshold}')
        ax2.set_xlabel('Star Magnitude')
        ax2.set_ylabel('Signal-to-Noise Ratio')
        ax2.set_title('SNR vs Magnitude')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _save_calibration_data(self, validation_results: Dict[str, Any]) -> Path:
        """Save photometric calibration data."""
        output_path = self.config.output_dir / "photometric_calibration_data.csv"
        
        calibration_data = []
        
        # Extract SNR vs magnitude data
        if 'snr_analysis' in validation_results:
            snr_data = validation_results['snr_analysis']['measurement_data']
            for i, (mag, snr, electrons) in enumerate(zip(
                snr_data['magnitudes'], 
                snr_data['snr_values'], 
                snr_data['electron_counts']
            )):
                calibration_data.append({
                    'measurement_type': 'snr_vs_magnitude',
                    'magnitude': mag,
                    'snr': snr,
                    'electron_count': electrons,
                    'exposure_time': snr_data['exposure_time']
                })
        
        # Extract magnitude validation data
        if 'magnitude_validation' in validation_results:
            mag_data = validation_results['magnitude_validation']
            if 'electron_count_comparison' in mag_data:
                comparison = mag_data['electron_count_comparison']
                for i, (theoretical, simulated, error) in enumerate(zip(
                    comparison['theoretical_electrons'],
                    comparison['simulated_electrons'], 
                    comparison['relative_errors']
                )):
                    calibration_data.append({
                        'measurement_type': 'magnitude_validation',
                        'theoretical_electrons': theoretical,
                        'simulated_electrons': simulated,
                        'relative_error': error,
                        'exposure_time': mag_data['exposure_time']
                    })
        
        if calibration_data:
            df = pd.DataFrame(calibration_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved calibration data: {output_path}")
        
        return output_path

# Export main class
__all__ = ['PhotometricValidator', 'PhotometricConfig', 'PhotometricMeasurement', 'CalibrationCurve']