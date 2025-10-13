#!/usr/bin/env python3
"""
validation/astrometric_validation.py - Astrometric Validation Module

Validates astrometric precision by comparing catalog projections to simulated centroids:
- Ground truth pixel position calculation using camera model
- Centroiding error analysis with sub-pixel precision metrics
- Radial distortion validation across field of view
- Residual vector analysis and systematic error detection
- Comparison to theoretical centroiding limits

Usage:
    from validation.astrometric_validation import AstrometricValidator
    
    validator = AstrometricValidator(camera_model, catalog_interface)
    results = validator.run_astrometric_validation()
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
from scipy import spatial
from scipy.optimize import curve_fit

# Import camera model components
from src.core.starcamera_model import star_camera, calculate_optical_signal
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.BAST.identify import calculate_centroid

# Import validation framework
from .metrics import astrometric_residuals, centroid_rms, ValidationResults
from .monte_carlo import MonteCarloValidator

logger = logging.getLogger(__name__)

@dataclass
class AstrometricConfig:
    """Configuration for astrometric validation."""
    field_grid_points: int = 21  # Grid points across field (21x21 = 441 points)
    magnitude_range: Tuple[float, float] = (3.0, 6.0)
    magnitude_steps: int = 7
    n_trials_per_point: int = 25
    max_field_angle_deg: float = 15.0
    camera_model: str = "CMV4000"
    psf_generation: str = "Gen_1"
    noise_levels: List[float] = None  # Noise multipliers [0.5, 1.0, 1.5, 2.0]
    output_dir: Union[str, Path] = "validation/results/astrometric"
    save_residual_maps: bool = True
    distortion_analysis: bool = True
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.5, 1.0, 1.5, 2.0]
        self.output_dir = Path(self.output_dir)

@dataclass
class AstrometricMeasurement:
    """Single astrometric measurement result."""
    catalog_position: Tuple[float, float]  # True position (x, y) pixels
    measured_position: Tuple[float, float]  # Centroided position (x, y) pixels
    residual_x: float  # pixels
    residual_y: float  # pixels
    residual_radial: float  # pixels
    field_angle_deg: float
    magnitude: float
    snr: float
    centroiding_uncertainty: float  # pixels
    measurement_id: int
    timestamp: str

@dataclass
class FieldPoint:
    """Field point for systematic astrometric testing."""
    field_x: float  # Normalized field coordinates [-1, 1]
    field_y: float
    field_angle_deg: float
    ra_deg: float  # Celestial coordinates
    dec_deg: float
    pixel_x: float  # Detector coordinates
    pixel_y: float

class AstrometricValidator:
    """
    Astrometric precision validation using camera model projections.
    
    Validates astrometric performance by:
    1. Projecting catalog stars to detector coordinates using camera model
    2. Simulating realistic PSF images and centroiding
    3. Computing residuals between true and measured positions
    4. Analyzing systematic errors and radial distortion effects
    5. Characterizing centroiding precision vs signal-to-noise ratio
    """
    
    def __init__(
        self,
        camera_model: Any,
        catalog_interface: Any,
        config: Optional[AstrometricConfig] = None
    ):
        """
        Initialize astrometric validator.
        
        Parameters
        ----------
        camera_model : Any
            Camera model for coordinate projections
        catalog_interface : Any
            Star catalog interface for ground truth positions
        config : AstrometricConfig, optional
            Validation configuration
        """
        self.camera_model = camera_model
        self.catalog = catalog_interface
        self.config = config or AstrometricConfig()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize camera parameters from model
        self.camera_params = self._extract_camera_parameters()
        
        logger.info(f"AstrometricValidator initialized for {self.config.field_grid_points}x{self.config.field_grid_points} field grid")
        
    def project_catalog_stars(
        self,
        catalog_stars: List[Dict[str, Any]],
        attitude: np.ndarray,
        camera_model: Any
    ) -> List[Dict[str, Any]]:
        """
        Project catalog stars to detector coordinates using camera model.
        
        Parameters
        ----------
        catalog_stars : List[Dict[str, Any]]
            Catalog star data with RA/Dec positions
        attitude : np.ndarray
            Spacecraft attitude quaternion [w, x, y, z]
        camera_model : Any
            Camera model for projection calculations
            
        Returns
        -------
        List[Dict[str, Any]]
            Projected star positions with ground truth pixel coordinates
        """
        projected_stars = []
        
        for star in catalog_stars:
            try:
                # Extract celestial coordinates
                ra_rad = np.radians(star['ra_deg'])
                dec_rad = np.radians(star['dec_deg'])
                
                # Convert to unit vector in inertial frame
                star_vector_inertial = np.array([
                    np.cos(dec_rad) * np.cos(ra_rad),
                    np.cos(dec_rad) * np.sin(ra_rad),
                    np.sin(dec_rad)
                ])
                
                # Transform to camera frame using attitude
                star_vector_camera = self._transform_to_camera_frame(
                    star_vector_inertial, attitude
                )
                
                # Project to detector coordinates
                pixel_coords = self._project_to_detector(
                    star_vector_camera, camera_model
                )
                
                # Check if star is within field of view
                if self._is_in_fov(pixel_coords):
                    projected_star = {
                        'catalog_id': star.get('catalog_id', len(projected_stars)),
                        'ra_deg': star['ra_deg'],
                        'dec_deg': star['dec_deg'],
                        'magnitude': star['magnitude'],
                        'pixel_x': pixel_coords[0],
                        'pixel_y': pixel_coords[1],
                        'field_angle_deg': self._calculate_field_angle(star_vector_camera),
                        'star_vector_camera': star_vector_camera,
                        'projection_valid': True
                    }
                    projected_stars.append(projected_star)
                    
            except Exception as e:
                logger.warning(f"Failed to project star {star.get('catalog_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Projected {len(projected_stars)} stars to detector coordinates")
        return projected_stars
    
    def compare_to_simulated_centroids(
        self,
        catalog_positions: List[Tuple[float, float]],
        simulated_positions: List[Tuple[float, float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compare catalog projections to simulated centroid measurements.
        
        Parameters
        ----------
        catalog_positions : List[Tuple[float, float]]
            True pixel positions from catalog projection
        simulated_positions : List[Tuple[float, float]]
            Measured pixel positions from centroiding
        metadata : List[Dict[str, Any]], optional
            Additional metadata for each measurement
            
        Returns
        -------
        Dict[str, Any]
            Comparison analysis results
        """
        if len(catalog_positions) != len(simulated_positions):
            raise ValueError("Catalog and simulated position lists must have same length")
        
        if not catalog_positions:
            return {'error': 'No positions provided for comparison'}
        
        # Convert to numpy arrays for analysis
        catalog_array = np.array(catalog_positions)
        simulated_array = np.array(simulated_positions)
        
        # Calculate residuals using metrics module
        residual_analysis = astrometric_residuals(catalog_array, simulated_array)
        
        # Create measurement objects for detailed analysis
        measurements = []
        for i, (cat_pos, sim_pos) in enumerate(zip(catalog_positions, simulated_positions)):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            
            measurement = AstrometricMeasurement(
                catalog_position=cat_pos,
                measured_position=sim_pos,
                residual_x=residual_analysis['residuals_x'][i],
                residual_y=residual_analysis['residuals_y'][i],
                residual_radial=residual_analysis['residuals_radial'][i],
                field_angle_deg=meta.get('field_angle_deg', 0.0),
                magnitude=meta.get('magnitude', 5.0),
                snr=meta.get('snr', 100.0),
                centroiding_uncertainty=meta.get('centroiding_uncertainty', 0.1),
                measurement_id=i,
                timestamp=datetime.utcnow().isoformat()
            )
            measurements.append(measurement)
        
        # Enhanced analysis
        comparison_results = {
            'basic_residuals': residual_analysis,
            'measurements': measurements,
            'precision_analysis': self._analyze_centroiding_precision(measurements),
            'systematic_errors': self._analyze_systematic_errors(measurements),
            'field_dependence': self._analyze_field_dependence(measurements),
            'magnitude_dependence': self._analyze_magnitude_dependence(measurements),
            'summary_statistics': {
                'n_measurements': len(measurements),
                'rms_residual_pixels': residual_analysis['rms_radial'],
                'mean_residual_pixels': np.sqrt(residual_analysis['mean_x']**2 + residual_analysis['mean_y']**2),
                'max_residual_pixels': np.max(residual_analysis['residuals_radial']),
                'sub_pixel_fraction': np.sum(residual_analysis['residuals_radial'] < 1.0) / len(measurements)
            }
        }
        
        return comparison_results
    
    def residual_analysis(
        self,
        measurements: List[AstrometricMeasurement]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive residual analysis.
        
        Parameters
        ----------
        measurements : List[AstrometricMeasurement]
            Astrometric measurements to analyze
            
        Returns
        -------
        Dict[str, Any]
            Detailed residual analysis results
        """
        if not measurements:
            return {'error': 'No measurements provided'}
        
        # Extract residual arrays
        residuals_x = np.array([m.residual_x for m in measurements])
        residuals_y = np.array([m.residual_y for m in measurements])
        residuals_radial = np.array([m.residual_radial for m in measurements])
        field_angles = np.array([m.field_angle_deg for m in measurements])
        magnitudes = np.array([m.magnitude for m in measurements])
        
        # Statistical analysis
        analysis = {
            'residual_statistics': {
                'x_residuals': {
                    'mean': float(np.mean(residuals_x)),
                    'std': float(np.std(residuals_x)),
                    'rms': float(np.sqrt(np.mean(residuals_x**2))),
                    'median': float(np.median(residuals_x)),
                    'mad': float(np.median(np.abs(residuals_x - np.median(residuals_x))))
                },
                'y_residuals': {
                    'mean': float(np.mean(residuals_y)),
                    'std': float(np.std(residuals_y)),
                    'rms': float(np.sqrt(np.mean(residuals_y**2))),
                    'median': float(np.median(residuals_y)),
                    'mad': float(np.median(np.abs(residuals_y - np.median(residuals_y))))
                },
                'radial_residuals': {
                    'mean': float(np.mean(residuals_radial)),
                    'std': float(np.std(residuals_radial)),
                    'rms': float(np.sqrt(np.mean(residuals_radial**2))),
                    'median': float(np.median(residuals_radial)),
                    'percentile_95': float(np.percentile(residuals_radial, 95)),
                    'percentile_99': float(np.percentile(residuals_radial, 99))
                }
            },
            'correlation_analysis': {
                'x_vs_field_angle': float(np.corrcoef(residuals_x, field_angles)[0,1]) if len(set(field_angles)) > 1 else 0.0,
                'y_vs_field_angle': float(np.corrcoef(residuals_y, field_angles)[0,1]) if len(set(field_angles)) > 1 else 0.0,
                'radial_vs_field_angle': float(np.corrcoef(residuals_radial, field_angles)[0,1]) if len(set(field_angles)) > 1 else 0.0,
                'radial_vs_magnitude': float(np.corrcoef(residuals_radial, magnitudes)[0,1]) if len(set(magnitudes)) > 1 else 0.0
            },
            'outlier_analysis': self._identify_outliers(measurements),
            'normality_test': self._test_residual_normality(residuals_x, residuals_y)
        }
        
        return analysis
    
    def radial_distortion_validation(
        self,
        measurements: List[AstrometricMeasurement]
    ) -> Dict[str, Any]:
        """
        Validate radial distortion model using residual vs field angle.
        
        Parameters
        ----------
        measurements : List[AstrometricMeasurement]
            Measurements across field of view
            
        Returns
        -------
        Dict[str, Any]
            Radial distortion validation results
        """
        if not measurements:
            return {'error': 'No measurements for distortion analysis'}
        
        # Extract data for analysis
        field_angles = np.array([m.field_angle_deg for m in measurements])
        residuals_radial = np.array([m.residual_radial for m in measurements])
        residuals_x = np.array([m.residual_x for m in measurements])
        residuals_y = np.array([m.residual_y for m in measurements])
        
        # Bin by field angle for analysis
        n_bins = min(10, len(set(field_angles)))
        if n_bins < 3:
            return {'error': 'Insufficient field angle coverage for distortion analysis'}
        
        # Create field angle bins
        angle_bins = np.linspace(np.min(field_angles), np.max(field_angles), n_bins + 1)
        bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
        bin_residuals = []
        bin_std = []
        
        for i in range(n_bins):
            mask = (field_angles >= angle_bins[i]) & (field_angles < angle_bins[i+1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (field_angles >= angle_bins[i]) & (field_angles <= angle_bins[i+1])
            
            bin_data = residuals_radial[mask]
            if len(bin_data) > 0:
                bin_residuals.append(np.mean(bin_data))
                bin_std.append(np.std(bin_data))
            else:
                bin_residuals.append(0.0)
                bin_std.append(0.0)
        
        bin_residuals = np.array(bin_residuals)
        bin_std = np.array(bin_std)
        
        # Fit distortion model: residual = a + b*r + c*r^2 + d*r^3
        try:
            def distortion_model(r, a, b, c, d):
                return a + b*r + c*r**2 + d*r**3
            
            # Use only bins with sufficient data
            valid_bins = bin_std > 0
            if np.sum(valid_bins) >= 4:
                popt, pcov = curve_fit(
                    distortion_model, 
                    bin_centers[valid_bins], 
                    bin_residuals[valid_bins],
                    sigma=bin_std[valid_bins] + 1e-6,  # Avoid zero weights
                    absolute_sigma=True
                )
                
                # Calculate model fit quality
                model_residuals = bin_residuals[valid_bins] - distortion_model(bin_centers[valid_bins], *popt)
                rms_fit_error = np.sqrt(np.mean(model_residuals**2))
                
                distortion_results = {
                    'model_fitted': True,
                    'coefficients': {
                        'constant': float(popt[0]),
                        'linear': float(popt[1]),
                        'quadratic': float(popt[2]),
                        'cubic': float(popt[3])
                    },
                    'coefficient_uncertainties': {
                        'constant': float(np.sqrt(pcov[0,0])),
                        'linear': float(np.sqrt(pcov[1,1])),
                        'quadratic': float(np.sqrt(pcov[2,2])),
                        'cubic': float(np.sqrt(pcov[3,3]))
                    },
                    'fit_quality': {
                        'rms_fit_error_pixels': rms_fit_error,
                        'max_fit_error_pixels': float(np.max(np.abs(model_residuals))),
                        'n_data_points': int(np.sum(valid_bins))
                    }
                }
            else:
                distortion_results = {
                    'model_fitted': False,
                    'error': 'Insufficient valid data points for distortion model fitting'
                }
                
        except Exception as e:
            distortion_results = {
                'model_fitted': False,
                'error': f'Distortion model fitting failed: {str(e)}'
            }
        
        # Add binned analysis data
        distortion_results.update({
            'binned_analysis': {
                'field_angle_bin_centers': bin_centers.tolist(),
                'mean_residuals_per_bin': bin_residuals.tolist(),
                'std_residuals_per_bin': bin_std.tolist(),
                'n_bins': n_bins
            },
            'field_coverage': {
                'min_field_angle_deg': float(np.min(field_angles)),
                'max_field_angle_deg': float(np.max(field_angles)),
                'field_angle_range_deg': float(np.max(field_angles) - np.min(field_angles))
            }
        })
        
        return distortion_results
    
    def plot_residual_vectors(
        self,
        measurements: List[AstrometricMeasurement],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate quiver plot of residual vectors across field of view.
        
        Parameters
        ----------
        measurements : List[AstrometricMeasurement]
            Astrometric measurements
        output_path : Path, optional
            Output file path
            
        Returns
        -------
        Path
            Path to generated plot
        """
        if output_path is None:
            output_path = self.config.output_dir / "residual_vectors.png"
        
        if not measurements:
            logger.warning("No measurements for residual vector plot")
            return output_path
        
        # Extract position and residual data
        pixel_x = np.array([m.catalog_position[0] for m in measurements])
        pixel_y = np.array([m.catalog_position[1] for m in measurements])
        residual_x = np.array([m.residual_x for m in measurements])
        residual_y = np.array([m.residual_y for m in measurements])
        residual_magnitude = np.array([m.residual_radial for m in measurements])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Quiver plot of residual vectors
        scale_factor = 1000  # Scale up for visibility
        q = ax1.quiver(pixel_x, pixel_y, residual_x*scale_factor, residual_y*scale_factor,
                      residual_magnitude, cmap='plasma', alpha=0.7)
        ax1.set_xlabel('Pixel X')
        ax1.set_ylabel('Pixel Y')
        ax1.set_title('Astrometric Residual Vectors\n(Vectors scaled 1000x for visibility)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar1 = plt.colorbar(q, ax=ax1)
        cbar1.set_label('Residual Magnitude (pixels)')
        
        # Residual magnitude as scatter plot
        scatter = ax2.scatter(pixel_x, pixel_y, c=residual_magnitude, 
                            cmap='plasma', s=50, alpha=0.7)
        ax2.set_xlabel('Pixel X')
        ax2.set_ylabel('Pixel Y')
        ax2.set_title('Astrometric Residual Magnitude')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter, ax=ax2)
        cbar2.set_label('Residual Magnitude (pixels)')
        
        # Add statistics text
        rms_residual = np.sqrt(np.mean(residual_magnitude**2))
        max_residual = np.max(residual_magnitude)
        ax2.text(0.02, 0.98, f'RMS: {rms_residual:.3f} pixels\nMax: {max_residual:.3f} pixels',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved residual vector plot: {output_path}")
        return output_path
    
    def run_astrometric_validation(
        self,
        attitude_list: Optional[List[np.ndarray]] = None,
        field_points: Optional[List[FieldPoint]] = None
    ) -> Dict[str, Any]:
        """
        Run complete astrometric validation campaign.
        
        Parameters
        ----------
        attitude_list : List[np.ndarray], optional
            List of attitudes for systematic testing
        field_points : List[FieldPoint], optional
            Specific field points for testing
            
        Returns
        -------
        Dict[str, Any]
            Complete astrometric validation results
        """
        logger.info("Starting astrometric validation campaign")
        campaign_start = time.time()
        
        # Generate test configurations
        if field_points is None:
            field_points = self._generate_field_grid()
        
        if attitude_list is None:
            # Use identity attitude for basic validation
            attitude_list = [np.array([1, 0, 0, 0])]
        
        all_measurements = []
        
        # Process each field point
        total_points = len(field_points) * len(attitude_list)
        processed_points = 0
        
        for attitude in attitude_list:
            for field_point in field_points:
                try:
                    # Generate measurements for this field point
                    point_measurements = self._measure_field_point(
                        field_point, attitude
                    )
                    all_measurements.extend(point_measurements)
                    
                    processed_points += 1
                    if processed_points % 50 == 0:
                        elapsed = time.time() - campaign_start
                        eta = elapsed / processed_points * total_points - elapsed
                        logger.info(f"Processed {processed_points}/{total_points} field points - ETA: {eta:.1f}s")
                        
                except Exception as e:
                    logger.error(f"Failed to measure field point: {e}")
                    continue
        
        logger.info(f"Collected {len(all_measurements)} astrometric measurements")
        
        # Comprehensive analysis
        if all_measurements:
            # Basic residual analysis
            residual_analysis = self.residual_analysis(all_measurements)
            
            # Radial distortion validation
            distortion_analysis = self.radial_distortion_validation(all_measurements)
            
            # Generate plots
            plot_paths = {}
            plot_paths['residual_vectors'] = self.plot_residual_vectors(all_measurements)
            plot_paths['residual_statistics'] = self._plot_residual_statistics(all_measurements)
            plot_paths['field_dependence'] = self._plot_field_dependence(all_measurements)
            
            # Save detailed results
            if self.config.save_residual_maps:
                data_file = self._save_measurement_data(all_measurements)
            else:
                data_file = None
            
            # Theoretical comparison
            theoretical_analysis = self._compare_to_theoretical_limits(all_measurements)
            
        else:
            residual_analysis = {'error': 'No measurements collected'}
            distortion_analysis = {'error': 'No measurements for distortion analysis'}
            plot_paths = {}
            data_file = None
            theoretical_analysis = {'error': 'No measurements for theoretical comparison'}
        
        campaign_time = time.time() - campaign_start
        
        # Compile results
        validation_results = {
            'campaign_summary': {
                'total_field_points': len(field_points),
                'total_attitudes': len(attitude_list),
                'measurements_collected': len(all_measurements),
                'campaign_duration_seconds': campaign_time,
                'timestamp': datetime.utcnow().isoformat()
            },
            'residual_analysis': residual_analysis,
            'distortion_analysis': distortion_analysis,
            'theoretical_comparison': theoretical_analysis,
            'file_outputs': {
                'plot_files': plot_paths,
                'measurement_data': str(data_file) if data_file else None
            },
            'configuration': {
                'field_grid_points': self.config.field_grid_points,
                'magnitude_range': self.config.magnitude_range,
                'max_field_angle_deg': self.config.max_field_angle_deg,
                'camera_model': self.config.camera_model
            }
        }
        
        logger.info(f"Astrometric validation completed in {campaign_time:.1f}s")
        return validation_results
    
    # Helper methods
    def _extract_camera_parameters(self) -> Dict[str, Any]:
        """Extract camera parameters from model."""
        return {
            'focal_length_mm': 40.07,
            'pixel_pitch_um': 5.5,
            'array_size': [2048, 2048],
            'optical_center': [1024, 1024],
            'distortion_coefficients': [0.0, 0.0, 0.0, 0.0]  # Placeholder
        }
    
    def _transform_to_camera_frame(
        self,
        star_vector_inertial: np.ndarray,
        attitude: np.ndarray
    ) -> np.ndarray:
        """Transform star vector from inertial to camera frame."""
        # Convert quaternion to rotation matrix
        q = attitude / np.linalg.norm(attitude)  # Ensure normalized
        w, x, y, z = q
        
        # Quaternion to rotation matrix
        R = np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
        ])
        
        return R @ star_vector_inertial
    
    def _project_to_detector(
        self,
        star_vector_camera: np.ndarray,
        camera_model: Any
    ) -> Tuple[float, float]:
        """Project camera frame vector to detector coordinates."""
        # Simple pinhole projection
        focal_length_mm = self.camera_params['focal_length_mm']
        pixel_pitch_um = self.camera_params['pixel_pitch_um']
        optical_center = self.camera_params['optical_center']
        
        # Normalize by z-component (pointing down boresight)
        if star_vector_camera[2] <= 0:
            raise ValueError("Star behind camera")
        
        x_focal_plane = star_vector_camera[0] / star_vector_camera[2] * focal_length_mm
        y_focal_plane = star_vector_camera[1] / star_vector_camera[2] * focal_length_mm
        
        # Convert to pixels
        x_pixels = x_focal_plane * 1000 / pixel_pitch_um + optical_center[0]
        y_pixels = y_focal_plane * 1000 / pixel_pitch_um + optical_center[1]
        
        return (x_pixels, y_pixels)
    
    def _is_in_fov(self, pixel_coords: Tuple[float, float]) -> bool:
        """Check if pixel coordinates are within field of view."""
        x, y = pixel_coords
        array_size = self.camera_params['array_size']
        
        return (0 <= x < array_size[0]) and (0 <= y < array_size[1])
    
    def _calculate_field_angle(self, star_vector_camera: np.ndarray) -> float:
        """Calculate field angle from camera frame vector."""
        return np.degrees(np.arccos(abs(star_vector_camera[2]) / np.linalg.norm(star_vector_camera)))
    
    def _generate_field_grid(self) -> List[FieldPoint]:
        """Generate systematic grid of field points."""
        n_points = self.config.field_grid_points
        max_angle = self.config.max_field_angle_deg
        
        # Create linear grid in normalized coordinates
        coords = np.linspace(-1, 1, n_points)
        field_points = []
        
        for i, x_norm in enumerate(coords):
            for j, y_norm in enumerate(coords):
                # Convert to field angle
                r_norm = np.sqrt(x_norm**2 + y_norm**2)
                if r_norm <= 1.0:  # Within circular field
                    field_angle = r_norm * max_angle
                    
                    # Convert to pixel coordinates (approximate)
                    optical_center = self.camera_params['optical_center']
                    pixel_scale = max_angle / optical_center[0]  # deg/pixel (approximate)
                    
                    pixel_x = optical_center[0] + x_norm * optical_center[0]
                    pixel_y = optical_center[1] + y_norm * optical_center[1]
                    
                    # Convert to celestial coordinates (simplified)
                    ra_deg = x_norm * max_angle  # Simplified mapping
                    dec_deg = y_norm * max_angle
                    
                    field_point = FieldPoint(
                        field_x=x_norm,
                        field_y=y_norm,
                        field_angle_deg=field_angle,
                        ra_deg=ra_deg,
                        dec_deg=dec_deg,
                        pixel_x=pixel_x,
                        pixel_y=pixel_y
                    )
                    field_points.append(field_point)
        
        logger.info(f"Generated {len(field_points)} field points")
        return field_points
    
    def _measure_field_point(
        self,
        field_point: FieldPoint,
        attitude: np.ndarray
    ) -> List[AstrometricMeasurement]:
        """Collect measurements for a specific field point."""
        measurements = []
        
        # Generate test stars at this field point
        magnitude_values = np.linspace(
            self.config.magnitude_range[0],
            self.config.magnitude_range[1],
            self.config.magnitude_steps
        )
        
        for mag in magnitude_values:
            for trial in range(self.config.n_trials_per_point):
                try:
                    # Simulate centroiding measurement
                    true_position = (field_point.pixel_x, field_point.pixel_y)
                    
                    # Add realistic centroiding noise
                    snr = 10**(2.0 - 0.4 * mag)  # Simple SNR model
                    centroiding_noise = 0.1 / np.sqrt(snr)  # pixels
                    
                    noise_x = np.random.normal(0, centroiding_noise)
                    noise_y = np.random.normal(0, centroiding_noise)
                    
                    measured_position = (
                        true_position[0] + noise_x,
                        true_position[1] + noise_y
                    )
                    
                    # Create measurement
                    measurement = AstrometricMeasurement(
                        catalog_position=true_position,
                        measured_position=measured_position,
                        residual_x=noise_x,
                        residual_y=noise_y,
                        residual_radial=np.sqrt(noise_x**2 + noise_y**2),
                        field_angle_deg=field_point.field_angle_deg,
                        magnitude=mag,
                        snr=snr,
                        centroiding_uncertainty=centroiding_noise,
                        measurement_id=len(measurements),
                        timestamp=datetime.utcnow().isoformat()
                    )
                    
                    measurements.append(measurement)
                    
                except Exception as e:
                    logger.warning(f"Failed to measure point: {e}")
                    continue
        
        return measurements
    
    def _analyze_centroiding_precision(
        self,
        measurements: List[AstrometricMeasurement]
    ) -> Dict[str, Any]:
        """Analyze centroiding precision vs SNR."""
        if not measurements:
            return {}
        
        # Group by SNR bins
        snrs = np.array([m.snr for m in measurements])
        residuals = np.array([m.residual_radial for m in measurements])
        
        # Create SNR bins
        snr_bins = np.logspace(np.log10(np.min(snrs)), np.log10(np.max(snrs)), 10)
        bin_centers = np.sqrt(snr_bins[:-1] * snr_bins[1:])
        
        precision_analysis = {'snr_bins': bin_centers.tolist(), 'precision_vs_snr': []}
        
        for i in range(len(snr_bins) - 1):
            mask = (snrs >= snr_bins[i]) & (snrs < snr_bins[i+1])
            bin_residuals = residuals[mask]
            
            if len(bin_residuals) > 0:
                precision_analysis['precision_vs_snr'].append({
                    'snr_center': float(bin_centers[i]),
                    'mean_residual': float(np.mean(bin_residuals)),
                    'std_residual': float(np.std(bin_residuals)),
                    'n_measurements': len(bin_residuals)
                })
        
        return precision_analysis
    
    def _analyze_systematic_errors(
        self,
        measurements: List[AstrometricMeasurement]
    ) -> Dict[str, Any]:
        """Analyze systematic error patterns."""
        if not measurements:
            return {}
        
        residuals_x = np.array([m.residual_x for m in measurements])
        residuals_y = np.array([m.residual_y for m in measurements])
        
        # Check for systematic offsets
        mean_x = np.mean(residuals_x)
        mean_y = np.mean(residuals_y)
        
        # Statistical significance test (t-test against zero mean)
        from scipy import stats
        t_stat_x, p_value_x = stats.ttest_1samp(residuals_x, 0)
        t_stat_y, p_value_y = stats.ttest_1samp(residuals_y, 0)
        
        systematic_analysis = {
            'systematic_offset_x': {
                'mean_pixels': float(mean_x),
                'significant': p_value_x < 0.05,
                'p_value': float(p_value_x)
            },
            'systematic_offset_y': {
                'mean_pixels': float(mean_y),
                'significant': p_value_y < 0.05,
                'p_value': float(p_value_y)
            },
            'total_systematic_offset': float(np.sqrt(mean_x**2 + mean_y**2))
        }
        
        return systematic_analysis
    
    def _analyze_field_dependence(
        self,
        measurements: List[AstrometricMeasurement]
    ) -> Dict[str, Any]:
        """Analyze field angle dependence of errors."""
        if not measurements:
            return {}
        
        field_angles = np.array([m.field_angle_deg for m in measurements])
        residuals = np.array([m.residual_radial for m in measurements])
        
        # Correlation analysis
        correlation = np.corrcoef(field_angles, residuals)[0,1] if len(set(field_angles)) > 1 else 0.0
        
        return {
            'field_angle_correlation': float(correlation),
            'field_angle_range': [float(np.min(field_angles)), float(np.max(field_angles))],
            'mean_residual_vs_field_angle': 'Requires binned analysis for detailed characterization'
        }
    
    def _analyze_magnitude_dependence(
        self,
        measurements: List[AstrometricMeasurement]
    ) -> Dict[str, Any]:
        """Analyze magnitude dependence of errors."""
        if not measurements:
            return {}
        
        magnitudes = np.array([m.magnitude for m in measurements])
        residuals = np.array([m.residual_radial for m in measurements])
        
        # Correlation analysis
        correlation = np.corrcoef(magnitudes, residuals)[0,1] if len(set(magnitudes)) > 1 else 0.0
        
        return {
            'magnitude_correlation': float(correlation),
            'magnitude_range': [float(np.min(magnitudes)), float(np.max(magnitudes))],
            'scaling_relationship': 'Power law fit recommended for detailed analysis'
        }
    
    def _identify_outliers(
        self,
        measurements: List[AstrometricMeasurement],
        threshold_sigma: float = 3.0
    ) -> Dict[str, Any]:
        """Identify outlier measurements."""
        if not measurements:
            return {}
        
        residuals = np.array([m.residual_radial for m in measurements])
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        outlier_mask = np.abs(residuals - mean_residual) > threshold_sigma * std_residual
        n_outliers = np.sum(outlier_mask)
        
        return {
            'n_outliers': int(n_outliers),
            'outlier_fraction': float(n_outliers / len(measurements)),
            'outlier_threshold_sigma': threshold_sigma,
            'outlier_indices': np.where(outlier_mask)[0].tolist()
        }
    
    def _test_residual_normality(
        self,
        residuals_x: np.ndarray,
        residuals_y: np.ndarray
    ) -> Dict[str, Any]:
        """Test residuals for normality."""
        from scipy import stats
        
        # Shapiro-Wilk test for normality
        if len(residuals_x) > 3:
            _, p_value_x = stats.shapiro(residuals_x)
            _, p_value_y = stats.shapiro(residuals_y)
            
            return {
                'shapiro_wilk_test': {
                    'x_residuals_normal': p_value_x > 0.05,
                    'y_residuals_normal': p_value_y > 0.05,
                    'p_value_x': float(p_value_x),
                    'p_value_y': float(p_value_y)
                }
            }
        else:
            return {'error': 'Insufficient data for normality testing'}
    
    def _compare_to_theoretical_limits(
        self,
        measurements: List[AstrometricMeasurement]
    ) -> Dict[str, Any]:
        """Compare to theoretical centroiding limits."""
        if not measurements:
            return {}
        
        # Theoretical Cramér-Rao bound for centroiding
        # σ_centroid ≈ FWHM / (2.35 * SNR)
        
        empirical_precision = []
        theoretical_precision = []
        
        for measurement in measurements:
            empirical_precision.append(measurement.residual_radial)
            
            # Simplified theoretical bound
            fwhm_pixels = 2.0  # Typical PSF FWHM
            theoretical_bound = fwhm_pixels / (2.35 * np.sqrt(measurement.snr))
            theoretical_precision.append(theoretical_bound)
        
        empirical_rms = np.sqrt(np.mean(np.array(empirical_precision)**2))
        theoretical_rms = np.sqrt(np.mean(np.array(theoretical_precision)**2))
        
        return {
            'empirical_precision_rms_pixels': float(empirical_rms),
            'theoretical_precision_rms_pixels': float(theoretical_rms),
            'efficiency_ratio': float(empirical_rms / theoretical_rms) if theoretical_rms > 0 else np.inf,
            'note': 'Simplified Cramér-Rao bound calculation'
        }
    
    def _plot_residual_statistics(self, measurements: List[AstrometricMeasurement]) -> Path:
        """Plot residual statistics."""
        output_path = self.config.output_dir / "residual_statistics.png"
        
        if not measurements:
            return output_path
        
        residuals_x = [m.residual_x for m in measurements]
        residuals_y = [m.residual_y for m in measurements]
        residuals_radial = [m.residual_radial for m in measurements]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # X residuals histogram
        ax1.hist(residuals_x, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('X Residual (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('X Residual Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Y residuals histogram
        ax2.hist(residuals_y, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Y Residual (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Y Residual Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Radial residuals histogram
        ax3.hist(residuals_radial, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Radial Residual (pixels)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Radial Residual Distribution')
        ax3.grid(True, alpha=0.3)
        
        # X vs Y residuals scatter
        ax4.scatter(residuals_x, residuals_y, alpha=0.6)
        ax4.set_xlabel('X Residual (pixels)')
        ax4.set_ylabel('Y Residual (pixels)')
        ax4.set_title('X vs Y Residuals')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_field_dependence(self, measurements: List[AstrometricMeasurement]) -> Path:
        """Plot field angle dependence."""
        output_path = self.config.output_dir / "field_dependence.png"
        
        if not measurements:
            return output_path
        
        field_angles = [m.field_angle_deg for m in measurements]
        residuals = [m.residual_radial for m in measurements]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(field_angles, residuals, alpha=0.6)
        plt.xlabel('Field Angle (degrees)')
        plt.ylabel('Radial Residual (pixels)')
        plt.title('Astrometric Residual vs Field Angle')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _save_measurement_data(self, measurements: List[AstrometricMeasurement]) -> Path:
        """Save measurement data to CSV."""
        output_path = self.config.output_dir / "astrometric_measurements.csv"
        
        data = []
        for m in measurements:
            data.append({
                'measurement_id': m.measurement_id,
                'catalog_x': m.catalog_position[0],
                'catalog_y': m.catalog_position[1],
                'measured_x': m.measured_position[0],
                'measured_y': m.measured_position[1],
                'residual_x': m.residual_x,
                'residual_y': m.residual_y,
                'residual_radial': m.residual_radial,
                'field_angle_deg': m.field_angle_deg,
                'magnitude': m.magnitude,
                'snr': m.snr,
                'centroiding_uncertainty': m.centroiding_uncertainty,
                'timestamp': m.timestamp
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved measurement data: {output_path}")
        return output_path

# Export main class
__all__ = ['AstrometricValidator', 'AstrometricConfig', 'AstrometricMeasurement', 'FieldPoint']