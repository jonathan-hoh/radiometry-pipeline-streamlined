import numpy as np
from scipy.ndimage import shift
from ..core.star_tracker_pipeline import StarTrackerPipeline

class MultiStarRadiometry:
    """Creates detector images with multiple PSFs"""

    def __init__(self, pipeline: StarTrackerPipeline):
        self.pipeline = pipeline

    def render_scene(self, scene_data, psf_data):
        """Render multiple stars onto detector using existing PSF"""

        # Create full CMV4000 detector size
        detector_image = np.zeros((2048, 2048))

        for star in scene_data['stars']:
            # Use existing FPA-projected simulation method
            star_sim = self.pipeline.simulate_star(magnitude=star['magnitude'])
            
            # Run FPA projection simulation for single star 
            fpa_results = self.pipeline.run_monte_carlo_simulation_fpa_projected(
                psf_data, 
                photon_count=star_sim['photon_count'],
                num_trials=1,
                target_pixel_pitch_um=5.5,
                create_full_fpa=False
            )
            
            # Get the simulated FPA PSF image
            star_psf_image = fpa_results['projection_results']['simulations'][0]

            # Place at star's detector position using sub-pixel precision
            detector_image, actual_psf_center = self._place_psf_at_position_subpixel(
                detector_image,
                star_psf_image,
                star['detector_position']
            )
            
            # Store the actual PSF center for accurate ground truth
            star['actual_psf_center'] = actual_psf_center

        scene_data['detector_image'] = detector_image
        return scene_data

    def _place_psf_at_position(self, detector_image, psf_image, position):
        """Place PSF at specified detector position with proper bounds handling"""
        import logging
        logger = logging.getLogger(__name__)
        
        psf_height, psf_width = psf_image.shape
        det_height, det_width = detector_image.shape

        # Calculate placement bounds
        center_x, center_y = position
        psf_start_x = int(center_x - psf_width // 2)
        psf_start_y = int(center_y - psf_height // 2)
        
        # Calculate overlap regions
        det_start_x = max(0, psf_start_x)
        det_start_y = max(0, psf_start_y)
        det_end_x = min(det_width, psf_start_x + psf_width)
        det_end_y = min(det_height, psf_start_y + psf_height)
        
        # Calculate corresponding PSF regions
        psf_offset_x = det_start_x - psf_start_x
        psf_offset_y = det_start_y - psf_start_y
        psf_end_x = psf_offset_x + (det_end_x - det_start_x)
        psf_end_y = psf_offset_y + (det_end_y - det_start_y)
        
        # Check if there's any overlap
        if det_start_x >= det_end_x or det_start_y >= det_end_y:
            logger.warning(f"PSF at position ({center_x:.1f}, {center_y:.1f}) is completely outside detector bounds")
            return detector_image
        
        # Place the overlapping portion
        try:
            detector_image[det_start_y:det_end_y, det_start_x:det_end_x] += \
                psf_image[psf_offset_y:psf_end_y, psf_offset_x:psf_end_x]
            
            logger.info(f"Placed PSF at ({center_x:.1f}, {center_y:.1f}), detector region: ({det_start_x}, {det_start_y}) to ({det_end_x}, {det_end_y})")
        except Exception as e:
            logger.error(f"Failed to place PSF at ({center_x:.1f}, {center_y:.1f}): {e}")

        return detector_image

    def _place_psf_at_position_subpixel(self, detector_image, psf_image, position):
        """
        Place PSF at specified detector position with sub-pixel precision.
        
        Args:
            detector_image: Target detector image array
            psf_image: PSF image to place
            position: (x, y) floating-point position for PSF center
            
        Returns:
            tuple: (updated_detector_image, actual_psf_center)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        center_x, center_y = position
        psf_height, psf_width = psf_image.shape
        det_height, det_width = detector_image.shape
        
        # Calculate exact PSF center position (no integer truncation)
        psf_center_x_exact = psf_width / 2.0 - 0.5  # Convert to 0-indexed center
        psf_center_y_exact = psf_height / 2.0 - 0.5
        
        # Calculate the sub-pixel offset needed to place PSF center at desired position
        target_center_x = center_x
        target_center_y = center_y
        
        # Calculate the shift needed for sub-pixel positioning
        # We want the PSF center to be at (target_center_x, target_center_y)
        # The PSF center is currently at (psf_center_x_exact, psf_center_y_exact)
        # So we need to shift the PSF, then place it
        
        # First, determine where to place the PSF array (integer grid position)
        psf_origin_x = int(np.floor(target_center_x - psf_center_x_exact))
        psf_origin_y = int(np.floor(target_center_y - psf_center_y_exact))
        
        # Calculate the sub-pixel shift needed within that placement
        fractional_shift_x = (target_center_x - psf_center_x_exact) - psf_origin_x
        fractional_shift_y = (target_center_y - psf_center_y_exact) - psf_origin_y
        
        # Apply sub-pixel shift to PSF using scipy's shift function (uses spline interpolation)
        if abs(fractional_shift_x) > 1e-6 or abs(fractional_shift_y) > 1e-6:
            # Only apply shift if it's significant (avoid numerical noise)
            shifted_psf = shift(psf_image, (fractional_shift_y, fractional_shift_x), 
                              mode='constant', cval=0.0, order=1)  # Linear interpolation
        else:
            shifted_psf = psf_image.copy()
        
        # Calculate actual PSF center after placement and shifting
        actual_psf_center_x = psf_origin_x + psf_center_x_exact + fractional_shift_x
        actual_psf_center_y = psf_origin_y + psf_center_y_exact + fractional_shift_y
        
        # Calculate overlap regions for placement
        det_start_x = max(0, psf_origin_x)
        det_start_y = max(0, psf_origin_y)
        det_end_x = min(det_width, psf_origin_x + psf_width)
        det_end_y = min(det_height, psf_origin_y + psf_height)
        
        # Calculate corresponding PSF regions
        psf_offset_x = det_start_x - psf_origin_x
        psf_offset_y = det_start_y - psf_origin_y
        psf_end_x = psf_offset_x + (det_end_x - det_start_x)
        psf_end_y = psf_offset_y + (det_end_y - det_start_y)
        
        # Check if there's any overlap
        if det_start_x >= det_end_x or det_start_y >= det_end_y:
            logger.warning(f"PSF at position ({center_x:.3f}, {center_y:.3f}) is completely outside detector bounds")
            return detector_image, (center_x, center_y)  # Return intended position if placement failed
        
        # Place the overlapping portion of the shifted PSF
        try:
            detector_image[det_start_y:det_end_y, det_start_x:det_end_x] += \
                shifted_psf[psf_offset_y:psf_end_y, psf_offset_x:psf_end_x]
            
            logger.info(f"Placed PSF at ({center_x:.3f}, {center_y:.3f}) with sub-pixel precision")
            logger.debug(f"  Fractional shift: ({fractional_shift_x:.3f}, {fractional_shift_y:.3f})")
            logger.debug(f"  Actual PSF center: ({actual_psf_center_x:.3f}, {actual_psf_center_y:.3f})")
            logger.debug(f"  Detector region: ({det_start_x}, {det_start_y}) to ({det_end_x}, {det_end_y})")
            
        except Exception as e:
            logger.error(f"Failed to place sub-pixel PSF at ({center_x:.3f}, {center_y:.3f}): {e}")
            return detector_image, (center_x, center_y)  # Return intended position if placement failed
        
        return detector_image, (actual_psf_center_x, actual_psf_center_y)