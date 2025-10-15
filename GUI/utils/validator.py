#!/usr/bin/env python3
"""
validator.py - Configuration Validation System

Provides centralized validation for star tracker simulation configurations.
Part of Phase 3 implementation.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Centralized configuration validator for star tracker simulation.
    
    Provides validation methods for all configuration categories
    and combined validation across multiple tabs.
    """
    
    @staticmethod
    def validate_sensor_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate sensor configuration.
        
        Args:
            config: Sensor configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate pixel pitch
            pixel_pitch = config.get('pixel_pitch', 0)
            if not (3.0 <= pixel_pitch <= 10.0):
                return False, f"Pixel pitch must be between 3.0 and 10.0 µm (got {pixel_pitch})"
            
            # Validate resolution
            resolution = config.get('resolution', '')
            valid_resolutions = ["512x512", "1024x1024", "2048x2048"]
            if resolution not in valid_resolutions:
                return False, f"Resolution must be one of {valid_resolutions} (got '{resolution}')"
            
            # Validate quantum efficiency
            qe = config.get('quantum_efficiency', 0)
            if not (0 <= qe <= 100):
                return False, f"Quantum efficiency must be between 0 and 100% (got {qe})"
            
            # Validate read noise
            read_noise = config.get('read_noise', 0)
            if not (0.0 <= read_noise <= 50.0):
                return False, f"Read noise must be between 0 and 50 e⁻ (got {read_noise})"
            
            # Validate dark current
            dark_current = config.get('dark_current', 0)
            if not (0.0 <= dark_current <= 200.0):
                return False, f"Dark current must be between 0 and 200 e⁻/s (got {dark_current})"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error validating sensor config: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_optics_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate optics configuration.
        
        Args:
            config: Optics configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate focal length
            focal_length = config.get('focal_length', 0)
            if not (10.0 <= focal_length <= 100.0):
                return False, f"Focal length must be between 10 and 100 mm (got {focal_length})"
            
            # Validate aperture
            aperture = config.get('aperture', '')
            valid_apertures = ["f/1.2", "f/1.4", "f/2.0", "f/2.8", "f/4.0", "f/5.6"]
            if aperture not in valid_apertures:
                return False, f"Aperture must be one of {valid_apertures} (got '{aperture}')"
            
            # Validate distortion
            distortion = config.get('distortion', '')
            valid_distortions = ["None", "Minimal", "Moderate"]
            if distortion not in valid_distortions:
                return False, f"Distortion must be one of {valid_distortions} (got '{distortion}')"
            
            # Check focal length and aperture compatibility
            f_number = float(aperture.replace("f/", ""))
            aperture_diameter = focal_length / f_number
            if aperture_diameter < 5.0:
                return False, f"Aperture diameter ({aperture_diameter:.1f}mm) is very small, may affect performance"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error validating optics config: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_scenario_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate scenario configuration.
        
        Args:
            config: Scenario configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate catalog
            catalog = config.get('catalog', '')
            valid_catalogs = ["Hipparcos", "Gaia DR3", "Custom"]
            if catalog not in valid_catalogs:
                return False, f"Catalog must be one of {valid_catalogs} (got '{catalog}')"
            
            # Validate magnitude limit
            mag_limit = config.get('magnitude_limit', 0)
            if not (0.0 <= mag_limit <= 10.0):
                return False, f"Magnitude limit must be between 0 and 10 (got {mag_limit})"
            
            # Validate trials
            trials = config.get('trials', 0)
            if not (100 <= trials <= 5000):
                return False, f"Number of trials must be between 100 and 5000 (got {trials})"
            
            # Validate PSF file
            psf_file = config.get('psf_file', '')
            if not psf_file or "not found" in psf_file:
                return False, "Valid PSF file must be selected"
            
            # Check if PSF file exists
            psf_path = Path("data/PSF_sims/Gen_1") / psf_file
            if not psf_path.exists():
                return False, f"PSF file not found: {psf_path}"
            
            # Validate environment
            environment = config.get('environment', '')
            valid_environments = ["Deep Space", "LEO", "GEO"]
            if environment not in valid_environments:
                return False, f"Environment must be one of {valid_environments} (got '{environment}')"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error validating scenario config: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_complete_config(sensor_config: Dict[str, Any], 
                               optics_config: Dict[str, Any], 
                               scenario_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate complete configuration across all tabs.
        
        Args:
            sensor_config: Sensor configuration dictionary
            optics_config: Optics configuration dictionary 
            scenario_config: Scenario configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate each section individually
            sensor_valid, sensor_error = ConfigValidator.validate_sensor_config(sensor_config)
            if not sensor_valid:
                return False, f"Sensor error: {sensor_error}"
            
            optics_valid, optics_error = ConfigValidator.validate_optics_config(optics_config)
            if not optics_valid:
                return False, f"Optics error: {optics_error}"
            
            scenario_valid, scenario_error = ConfigValidator.validate_scenario_config(scenario_config)
            if not scenario_valid:
                return False, f"Scenario error: {scenario_error}"
            
            # Cross-validation checks
            cross_valid, cross_error = ConfigValidator._validate_cross_dependencies(
                sensor_config, optics_config, scenario_config
            )
            if not cross_valid:
                return False, f"Configuration compatibility error: {cross_error}"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error in complete config validation: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def _validate_cross_dependencies(sensor_config: Dict[str, Any], 
                                   optics_config: Dict[str, Any], 
                                   scenario_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate cross-dependencies between configuration sections.
        
        Args:
            sensor_config: Sensor configuration dictionary
            optics_config: Optics configuration dictionary 
            scenario_config: Scenario configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check angular resolution vs magnitude limit compatibility
            pixel_pitch = sensor_config.get('pixel_pitch', 5.6)  # µm
            focal_length = optics_config.get('focal_length', 35.0)  # mm
            mag_limit = scenario_config.get('magnitude_limit', 6.5)
            
            # Calculate angular resolution
            pixel_size_mm = pixel_pitch * 1e-3  # Convert µm to mm
            angular_res_rad = pixel_size_mm / focal_length
            angular_res_arcsec = angular_res_rad * 206265  # Convert radians to arcseconds
            
            # Check if resolution is adequate for magnitude limit
            if mag_limit > 7.0 and angular_res_arcsec > 15.0:
                return False, (f"Angular resolution ({angular_res_arcsec:.1f} arcsec) may be too coarse "
                             f"for faint stars (mag {mag_limit}). Consider longer focal length or smaller pixels.")
            
            # Check if field of view is reasonable
            resolution = sensor_config.get('resolution', '2048x2048')
            sensor_pixels = int(resolution.split('x')[0]) if 'x' in resolution else 2048
            sensor_size_mm = sensor_pixels * pixel_size_mm
            fov_rad = 2 * (sensor_size_mm / (2 * focal_length))
            fov_deg = fov_rad * 57.2958  # Convert radians to degrees
            
            if fov_deg < 5.0:
                return False, (f"Field of view ({fov_deg:.1f}°) is very narrow. "
                             f"Consider shorter focal length for better sky coverage.")
            
            if fov_deg > 45.0:
                return False, (f"Field of view ({fov_deg:.1f}°) is very wide. "
                             f"Consider longer focal length for better accuracy.")
            
            # Check trials vs environment complexity
            trials = scenario_config.get('trials', 1000)
            environment = scenario_config.get('environment', 'Deep Space')
            
            if environment in ['LEO', 'GEO'] and trials < 500:
                return False, (f"Complex environment ({environment}) requires at least 500 trials "
                             f"for reliable statistics (current: {trials})")
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error in cross-dependency validation: {e}")
            return False, f"Cross-validation error: {str(e)}"
    
    @staticmethod
    def get_validation_warnings(sensor_config: Dict[str, Any], 
                              optics_config: Dict[str, Any], 
                              scenario_config: Dict[str, Any]) -> list:
        """
        Get list of validation warnings (non-blocking issues).
        
        Args:
            sensor_config: Sensor configuration dictionary
            optics_config: Optics configuration dictionary 
            scenario_config: Scenario configuration dictionary
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        try:
            # Check for performance warnings
            qe = sensor_config.get('quantum_efficiency', 60)
            if qe < 40:
                warnings.append(f"Low quantum efficiency ({qe}%) may reduce sensitivity")
            
            read_noise = sensor_config.get('read_noise', 13)
            if read_noise > 20:
                warnings.append(f"High read noise ({read_noise} e⁻) may affect faint star detection")
            
            # Check for high computational load
            trials = scenario_config.get('trials', 1000)
            resolution = sensor_config.get('resolution', '2048x2048')
            if trials > 2000 and '2048' in resolution:
                warnings.append("High trial count with large resolution may take significant time")
            
            # Check for unusual configurations
            mag_limit = scenario_config.get('magnitude_limit', 6.5)
            if mag_limit < 3.0:
                warnings.append("Very bright magnitude limit may result in few detected stars")
            
        except Exception as e:
            logger.warning(f"Error generating validation warnings: {e}")
            
        return warnings