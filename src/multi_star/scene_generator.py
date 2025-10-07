import numpy as np
import logging
from typing import Optional, Tuple, Union
from ..core.star_tracker_pipeline import StarTrackerPipeline
from ..BAST.catalog import Catalog
from .attitude_transform import (
    CameraParameters, 
    catalog_to_image_plane,
    radec_to_inertial_vector,
    quaternion_to_rotation_matrix,
    euler_to_rotation_matrix,
    transform_to_camera_frame,
    project_to_focal_plane,
    focal_plane_to_pixels,
    filter_detector_bounds
)

logger = logging.getLogger(__name__)

class MultiStarSceneGenerator:
    """Generates detector scenes from synthetic catalogs with attitude transformation"""

    def __init__(self, pipeline: StarTrackerPipeline):
        self.pipeline = pipeline
        # Extract camera parameters from pipeline for attitude transforms
        self.camera_params = CameraParameters(
            focal_length_mm=getattr(pipeline.camera, 'f_length', 8.0),
            pixel_pitch_um=5.5,  # CMV4000 specification
            detector_width=2048,  # CMV4000 specification  
            detector_height=2048  # CMV4000 specification
        )
        logger.info(f"Initialized scene generator with camera params: "
                   f"f={self.camera_params.focal_length_mm}mm, "
                   f"pitch={self.camera_params.pixel_pitch_um}μm")

    def generate_scene(self, synthetic_catalog: Catalog,
                      true_attitude_quaternion: Optional[np.ndarray] = None,
                      true_attitude_euler: Optional[Tuple[float, float, float]] = None,
                      detector_center_ra=None, detector_center_dec=None):
        """Create multi-star detector scene from catalog with attitude transformation
        
        Args:
            synthetic_catalog: BAST Catalog with star positions
            true_attitude_quaternion: Camera attitude as unit quaternion [q0, q1, q2, q3]
            true_attitude_euler: Camera attitude as (roll, pitch, yaw) in radians
            detector_center_ra: Override RA for detector center (legacy compatibility)
            detector_center_dec: Override Dec for detector center (legacy compatibility)
            
        Notes:
            - If attitude parameters provided, uses new attitude transformation
            - If only detector_center provided, uses legacy gnomonic projection 
            - If neither provided, uses identity attitude (trivial case)
        """

        # Determine if using new attitude transformation or legacy mode
        using_attitude_transform = (true_attitude_quaternion is not None or 
                                  true_attitude_euler is not None)
        
        # Determine rotation matrix from attitude representation
        rotation_matrix = None
        if true_attitude_quaternion is not None:
            rotation_matrix = quaternion_to_rotation_matrix(true_attitude_quaternion)
            logger.info(f"Using quaternion attitude for scene generation")
        elif true_attitude_euler is not None:
            rotation_matrix = euler_to_rotation_matrix(*true_attitude_euler)
            logger.info(f"Using Euler attitude for scene generation")
        else:
            logger.info("Using identity attitude (trivial case) for scene generation")

        # Use camera attitude if detector center is not specified
        if detector_center_ra is None:
            detector_center_ra = self.pipeline.camera.attitude_ra
        if detector_center_dec is None:
            detector_center_dec = self.pipeline.camera.attitude_dec

        scene_data = {
            'stars': [],
            'detector_image': None,
            'ground_truth': {
                'catalog': synthetic_catalog,
                'expected_triangles': self._extract_expected_triangles(synthetic_catalog)
            },
            'attitude': {
                'ra': detector_center_ra,
                'dec': detector_center_dec,
                'transformation_method': 'integrated_gnomonic',
                'rotation_matrix': rotation_matrix
            }
        }

        # Convert RA/Dec to detector positions using the corrected, integrated method
        for idx, star_row in synthetic_catalog.iterrows():
            detector_pos = self._sky_to_detector(
                star_row['RA'], star_row['DE'],
                detector_center_ra, detector_center_dec,
                rotation_matrix=rotation_matrix
            )
            scene_data['stars'].append({
                'catalog_idx': idx,
                'detector_position': detector_pos,
                'magnitude': star_row['Magnitude'],
                'star_id': star_row['Star ID'],
                'ra': star_row['RA'],
                'dec': star_row['DE']
            })

        return scene_data

    def _sky_to_detector(self, ra, dec, center_ra, center_dec, rotation_matrix: Optional[np.ndarray] = None):
        """Convert RA/Dec to detector pixel coordinates using gnomonic projection with attitude"""
        # Use CMV4000 specifications: 2048x2048, 5.5μm pixel pitch
        detector_size = 2048
        pixel_pitch_um = 5.5
        
        # Use the camera's focal length for realistic projection
        focal_length_mm = self.pipeline.camera.f_length  # Focal length in mm
        focal_length_um = focal_length_mm * 1000  # Convert to microns
        
        # Convert celestial coordinates to 3D unit vectors on celestial sphere
        star_vector = radec_to_inertial_vector(ra, dec)
        
        # Always calculate the boresight vector from the catalog center
        center_vector = radec_to_inertial_vector(center_ra, center_dec)

        if rotation_matrix is not None:
            # Apply attitude rotation to both the star and the boresight vectors
            star_vector = rotation_matrix @ star_vector
            center_vector = rotation_matrix @ center_vector
            
        # Define the tangent plane basis vectors relative to the (potentially rotated) boresight
        # This uses the "Sloan Digital Sky Survey" (SDSS) convention
        north_pole = np.array([0, 0, 1])
        east_vector = np.cross(center_vector, north_pole)
        east_vector /= np.linalg.norm(east_vector)
        north_vector = np.cross(center_vector, east_vector)
        north_vector /= np.linalg.norm(north_vector)

        # Gnomonic projection: project star onto the tangent plane
        dot_product = np.dot(star_vector, center_vector)
        
        if dot_product <= 1e-9: # Star is at or behind the camera's focal plane
            return (detector_size + 100, detector_size + 100)

        # Project onto the tangent plane by finding the intersection
        projected_vector = star_vector / dot_product
        
        # Calculate coordinates on the tangent plane
        x_tangent = np.dot(projected_vector - center_vector, east_vector)
        y_tangent = np.dot(projected_vector - center_vector, north_vector)
        
        # Convert tangent plane coordinates to pixels
        pixels_per_radian = focal_length_um / pixel_pitch_um
        
        x_offset = x_tangent * pixels_per_radian
        y_offset = y_tangent * pixels_per_radian
        
        # Place relative to detector center
        x_detector = detector_size / 2.0 + x_offset
        y_detector = detector_size / 2.0 - y_offset  # Flip Y for image coordinates
        
        # Bounds checking
        margin = 50
        if not (margin <= x_detector < detector_size - margin and
                margin <= y_detector < detector_size - margin):
            return (detector_size + 100, detector_size + 100)
        
        return (x_detector, y_detector)

    def _extract_expected_triangles(self, catalog):
        # This is a placeholder for now.
        # In a real scenario, you'd extract the pre-computed triangles from the catalog.
        if "Triplets" in catalog.columns:
            return catalog["Triplets"].dropna().tolist()
        return []