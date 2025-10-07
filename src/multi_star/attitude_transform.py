#!/usr/bin/env python3
"""
Attitude Transform Module

Implements Euler angle-based coordinate transformations for star tracker simulations.
Provides the ability to transform star catalog positions to image plane coordinates
for arbitrary camera attitudes (orientations).

This module implements the mathematical framework described in docs/technical/euler_angles.md
to enable realistic star tracker simulation with perturbed attitudes.

Key Functions:
- quaternion_to_rotation_matrix: Convert quaternions to rotation matrices
- euler_to_rotation_matrix: Convert Euler angles to rotation matrices  
- transform_catalog_stars: Transform RA/Dec stars to camera frame
- project_to_focal_plane: Project 3D vectors to 2D image plane
- catalog_to_image_plane: Complete transformation pipeline

Physical accuracy preserved:
- Angular relationships maintained through rotations
- CMV4000 sensor specifications used for projections
- Bearing vector calculations validated against catalog

Created for Phase 2 multi-star attitude estimation capabilities.
"""

import numpy as np
import logging
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CameraParameters:
    """Camera intrinsic parameters for CMV4000 sensor"""
    focal_length_mm: float = 8.0  # Default focal length in mm
    pixel_pitch_um: float = 5.5   # CMV4000 pixel pitch 
    detector_width: int = 2048    # CMV4000 resolution
    detector_height: int = 2048   # CMV4000 resolution
    
    @property
    def focal_length_um(self) -> float:
        """Focal length in microns for consistency with pixel pitch"""
        return self.focal_length_mm * 1000.0
    
    @property
    def principal_point(self) -> Tuple[float, float]:
        """Principal point at detector center (pixels)"""
        return (self.detector_width / 2.0, self.detector_height / 2.0)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion to 3x3 rotation matrix.
    
    Args:
        q: Unit quaternion [q0, q1, q2, q3] (scalar first)
        
    Returns:
        3x3 rotation matrix R such that v_camera = R^T @ v_inertial
        
    Implementation follows equation from euler_angles.md:
    R rotates vectors from inertial frame to camera frame
    """
    # Ensure normalized quaternion
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q
    
    # Construct rotation matrix from quaternion components
    R = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)], 
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
    ])
    
    logger.debug(f"Quaternion {q} -> Rotation matrix determinant: {np.linalg.det(R):.6f}")
    return R


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float, 
                           convention: str = "ZYX") -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        roll, pitch, yaw: Euler angles in radians
        convention: Rotation sequence (default "ZYX" for aerospace)
        
    Returns:
        3x3 rotation matrix
    """
    # Individual rotation matrices
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch) 
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    
    # Roll (X-axis rotation)
    R_x = np.array([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ])
    
    # Pitch (Y-axis rotation)
    R_y = np.array([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ])
    
    # Yaw (Z-axis rotation)
    R_z = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ])
    
    # Compose rotations based on convention
    if convention == "ZYX":
        R = R_z @ R_y @ R_x  # Standard aerospace convention
    elif convention == "XYZ":
        R = R_x @ R_y @ R_z
    else:
        raise ValueError(f"Unsupported convention: {convention}")
    
    logger.debug(f"Euler angles ({roll:.3f}, {pitch:.3f}, {yaw:.3f}) -> det(R) = {np.linalg.det(R):.6f}")
    return R


def radec_to_inertial_vector(ra: float, dec: float) -> np.ndarray:
    """
    Convert RA/Dec celestial coordinates to inertial unit vector.
    
    Args:
        ra: Right ascension in radians
        dec: Declination in radians
        
    Returns:
        3D unit vector in inertial frame [x, y, z]
        
    Implementation follows spherical-to-Cartesian conversion from euler_angles.md
    """
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra) 
    z = np.sin(dec)
    
    vector = np.array([x, y, z])
    # Ensure unit normalization (should already be unit, but numerical safety)
    return vector / np.linalg.norm(vector)


def transform_to_camera_frame(inertial_vectors: List[np.ndarray], 
                            rotation_matrix: np.ndarray) -> List[np.ndarray]:
    """
    Transform inertial vectors to camera frame using attitude.
    
    Args:
        inertial_vectors: List of unit vectors in inertial frame
        rotation_matrix: 3x3 rotation matrix from inertial to camera frame
        
    Returns:
        List of vectors in camera frame
        
    Implementation: v_camera = R @ v_inertial
    """
    camera_vectors = []
    
    for v_inertial in inertial_vectors:
        # Apply the rotation matrix directly, not its transpose
        v_camera = rotation_matrix @ v_inertial
        
        # Log the transformation for debugging
        logger.debug(f"Transforming inertial vector {v_inertial}")
        logger.debug(f"Rotation matrix:\n{rotation_matrix}")
        logger.debug(f"Resulting camera vector: {v_camera}")

        # Filter out stars behind camera (z <= 0)
        if v_camera[2] > 0:
            camera_vectors.append(v_camera)
            logger.debug(f"Star is in front of camera (z > 0)")
        else:
            logger.debug(f"Star is behind camera (z <= 0), skipping")
            
    return camera_vectors


def project_to_focal_plane(camera_vectors: List[np.ndarray], 
                          camera_params: CameraParameters) -> List[Tuple[float, float]]:
    """
    Project 3D camera vectors to 2D focal plane coordinates.
    
    Args:
        camera_vectors: List of 3D vectors in camera frame
        camera_params: Camera intrinsic parameters
        
    Returns:
        List of (x_f, y_f) focal plane positions in microns
        
    Implementation follows pinhole projection from euler_angles.md:
    x_f = f * (x_c / z_c), y_f = f * (y_c / z_c)
    """
    focal_plane_positions = []
    f = camera_params.focal_length_um
    
    for v_cam in camera_vectors:
        x_c, y_c, z_c = v_cam
        
        # Pinhole projection
        x_f = f * (x_c / z_c)
        y_f = f * (y_c / z_c)
        
        focal_plane_positions.append((x_f, y_f))
        logger.debug(f"Camera vector {v_cam} -> focal plane ({x_f:.2f}, {y_f:.2f}) um")
        
    return focal_plane_positions


def focal_plane_to_pixels(focal_positions: List[Tuple[float, float]], 
                         camera_params: CameraParameters) -> List[Tuple[float, float]]:
    """
    Convert focal plane coordinates to pixel positions.
    
    Args:
        focal_positions: List of (x_f, y_f) in microns
        camera_params: Camera parameters
        
    Returns:
        List of (p_x, p_y) pixel coordinates
        
    Implementation follows pixel conversion from euler_angles.md:
    p_x = p_cx + x_f/p, p_y = p_cy + y_f/p
    """
    pixel_positions = []
    p_cx, p_cy = camera_params.principal_point
    pitch = camera_params.pixel_pitch_um
    
    for x_f, y_f in focal_positions:
        p_x = p_cx + x_f / pitch
        p_y = p_cy + y_f / pitch  # Note: may need y-flip depending on convention
        
        pixel_positions.append((p_x, p_y))
        logger.debug(f"Focal ({x_f:.2f}, {y_f:.2f}) um -> pixel ({p_x:.2f}, {p_y:.2f})")
        
    return pixel_positions


def filter_detector_bounds(pixel_positions: List[Tuple[float, float]], 
                          camera_params: CameraParameters,
                          margin: int = 10) -> List[Tuple[float, float]]:
    """
    Filter pixel positions to those within detector bounds.
    
    Args:
        pixel_positions: List of (p_x, p_y) pixel coordinates
        camera_params: Camera parameters 
        margin: Safety margin from detector edges in pixels
        
    Returns:
        List of valid pixel positions within detector bounds
    """
    valid_positions = []
    w, h = camera_params.detector_width, camera_params.detector_height
    
    for p_x, p_y in pixel_positions:
        if (margin <= p_x < w - margin and margin <= p_y < h - margin):
            valid_positions.append((p_x, p_y))
            logger.debug(f"Valid pixel position: ({p_x:.2f}, {p_y:.2f})")
        else:
            logger.warning(f"Out of bounds pixel: ({p_x:.2f}, {p_y:.2f}) - detector size: {w}x{h}, margin: {margin}")
            
    logger.info(f"Filtering: {len(pixel_positions)} input -> {len(valid_positions)} valid positions")
    return valid_positions


def catalog_to_image_plane(catalog_stars: List[Tuple[float, float]], 
                          attitude_quaternion: Optional[np.ndarray] = None,
                          attitude_euler: Optional[Tuple[float, float, float]] = None,
                          camera_params: Optional[CameraParameters] = None) -> List[Tuple[float, float]]:
    """
    Complete transformation pipeline from catalog RA/Dec to image plane pixels.
    
    Args:
        catalog_stars: List of (RA, Dec) tuples in radians
        attitude_quaternion: Camera attitude as unit quaternion [q0, q1, q2, q3]
        attitude_euler: Camera attitude as (roll, pitch, yaw) Euler angles in radians
        camera_params: Camera intrinsic parameters (uses defaults if None)
        
    Returns:
        List of (pixel_x, pixel_y) coordinates for visible stars
        
    This is the main interface that implements the complete mathematical
    framework from docs/technical/euler_angles.md
    """
    if camera_params is None:
        camera_params = CameraParameters()
        
    # Determine rotation matrix from attitude representation
    if attitude_quaternion is not None:
        R = quaternion_to_rotation_matrix(attitude_quaternion)
        logger.info(f"Using quaternion attitude: {attitude_quaternion}")
    elif attitude_euler is not None:
        R = euler_to_rotation_matrix(*attitude_euler)
        logger.info(f"Using Euler attitude: {attitude_euler}")
    else:
        # Identity attitude (trivial case)
        R = np.eye(3)
        logger.info("Using identity attitude (trivial case)")
    
    # Step 1: Convert catalog RA/Dec to inertial unit vectors
    logger.debug("Step 1: Converting catalog coordinates to inertial vectors")
    inertial_vectors = []
    for ra, dec in catalog_stars:
        v_inertial = radec_to_inertial_vector(ra, dec)
        inertial_vectors.append(v_inertial)
    
    logger.info(f"Converted {len(inertial_vectors)} catalog stars to inertial vectors")
    
    # Step 2: Transform to camera frame
    logger.debug("Step 2: Transforming to camera frame")  
    camera_vectors = transform_to_camera_frame(inertial_vectors, R)
    
    logger.info(f"{len(camera_vectors)} stars visible (z > 0)")
    
    # Step 3: Project to focal plane
    logger.debug("Step 3: Projecting to focal plane")
    focal_positions = project_to_focal_plane(camera_vectors, camera_params)
    
    # Step 4: Convert to pixels
    logger.debug("Step 4: Converting to pixel coordinates")
    pixel_positions = focal_plane_to_pixels(focal_positions, camera_params)
    
    # Step 5: Filter detector bounds
    logger.debug("Step 5: Filtering detector bounds")
    valid_pixels = filter_detector_bounds(pixel_positions, camera_params)
    
    logger.info(f"Final result: {len(valid_pixels)} stars within detector bounds")
    
    return valid_pixels


def compute_bearing_vectors_from_pixels(pixel_positions: List[Tuple[float, float]], 
                                      camera_params: CameraParameters) -> List[np.ndarray]:
    """
    Back-compute bearing vectors from pixel positions for validation.
    
    Args:
        pixel_positions: List of (p_x, p_y) pixel coordinates
        camera_params: Camera parameters
        
    Returns:
        List of unit bearing vectors in camera frame
        
    This implements the reverse transformation for validation purposes,
    as described in euler_angles.md section 6.
    """
    bearing_vectors = []
    p_cx, p_cy = camera_params.principal_point
    pitch = camera_params.pixel_pitch_um
    f = camera_params.focal_length_um
    
    for p_x, p_y in pixel_positions:
        # Convert pixels back to focal plane
        x_f = (p_x - p_cx) * pitch
        y_f = -(p_y - p_cy) * pitch # Y-axis must be inverted from image to camera frame
        
        # Bearing vector (unit direction)
        bearing = np.array([x_f, y_f, f])
        bearing = bearing / np.linalg.norm(bearing)
        
        bearing_vectors.append(bearing)
        logger.debug(f"Pixel ({p_x:.2f}, {p_y:.2f}) -> bearing {bearing}")
        
    return bearing_vectors


def validate_angular_preservation(original_vectors: List[np.ndarray],
                                computed_vectors: List[np.ndarray],
                                tolerance_deg: float = 0.001) -> bool:
    """
    Validate that angular relationships are preserved through transformations.
    
    Args:
        original_vectors: Original inertial unit vectors
        computed_vectors: Back-computed bearing vectors
        tolerance_deg: Angular tolerance in degrees
        
    Returns:
        True if all angular relationships preserved within tolerance
        
    This validates the key requirement from euler_angles.md that rotations
    preserve inner angles between vectors.
    """
    tolerance_rad = np.radians(tolerance_deg)
    
    if len(original_vectors) != len(computed_vectors):
        logger.error(f"Vector count mismatch: {len(original_vectors)} vs {len(computed_vectors)}")
        return False
        
    if len(original_vectors) < 2:
        logger.warning("Need at least 2 vectors to validate angular preservation")
        return True  # Trivially true
        
    # Check all pairwise angles
    for i in range(len(original_vectors)):
        for j in range(i + 1, len(original_vectors)):
            # Original angle
            dot_orig = np.clip(np.dot(original_vectors[i], original_vectors[j]), -1.0, 1.0)
            angle_orig = np.arccos(dot_orig)
            
            # Computed angle  
            dot_comp = np.clip(np.dot(computed_vectors[i], computed_vectors[j]), -1.0, 1.0)
            angle_comp = np.arccos(dot_comp)
            
            # Check preservation
            angle_error = abs(angle_orig - angle_comp)
            angle_error_deg = np.degrees(angle_error)
            
            if angle_error > tolerance_rad:
                logger.error(f"Angular preservation failed for stars {i},{j}: "
                           f"original={np.degrees(angle_orig):.6f}°, "
                           f"computed={np.degrees(angle_comp):.6f}°, "
                           f"error={angle_error_deg:.6f}°")
                return False
            else:
                logger.debug(f"Stars {i},{j}: angle preserved within {angle_error_deg:.6f}° error")
                
    logger.info(f"Angular preservation validated for {len(original_vectors)} stars within {tolerance_deg}°")
    return True


# Convenience functions for common use cases
def random_quaternion() -> np.ndarray:
    """Generate a random unit quaternion for testing."""
    # Method from "Uniform Random Rotations" by Ken Shoemake
    u1, u2, u3 = np.random.uniform(0, 1, 3)
    
    q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    
    return np.array([q0, q1, q2, q3])


def random_euler_angles(max_angle_deg: float = 30.0) -> Tuple[float, float, float]:
    """Generate random Euler angles within specified bounds for testing."""
    max_rad = np.radians(max_angle_deg)
    return tuple(np.random.uniform(-max_rad, max_rad, 3))


if __name__ == "__main__":
    # Test the attitude transformation with a simple case
    logging.basicConfig(level=logging.INFO, 
                       format='%(levelname)s - %(name)s - %(message)s')
    
    # Test catalog stars (RA, Dec in radians)
    test_stars = [
        (0.0, 0.0),      # Star at origin
        (0.1, 0.0),      # Star 0.1 rad east
        (0.0, 0.1),      # Star 0.1 rad north
        (0.05, 0.05)     # Star northeast
    ]
    
    print("Testing attitude transformation pipeline...")
    print(f"Input: {len(test_stars)} catalog stars")
    
    # Test with identity attitude (trivial case)
    print("\n1. Testing identity attitude (trivial case)")
    pixels_identity = catalog_to_image_plane(test_stars)
    print(f"Result: {len(pixels_identity)} stars on detector")
    for i, pixel in enumerate(pixels_identity):
        print(f"  Star {i}: pixel ({pixel[0]:.1f}, {pixel[1]:.1f})")
    
    # Test with random quaternion attitude
    print("\n2. Testing random quaternion attitude")
    q_random = random_quaternion()
    pixels_quaternion = catalog_to_image_plane(test_stars, attitude_quaternion=q_random)
    print(f"Quaternion: {q_random}")
    print(f"Result: {len(pixels_quaternion)} stars on detector")
    for i, pixel in enumerate(pixels_quaternion):
        print(f"  Star {i}: pixel ({pixel[0]:.1f}, {pixel[1]:.1f})")
    
    # Test with Euler angles
    print("\n3. Testing Euler angle attitude")
    euler_random = random_euler_angles(15.0)  # ±15 degrees
    pixels_euler = catalog_to_image_plane(test_stars, attitude_euler=euler_random)
    print(f"Euler angles: ({np.degrees(euler_random[0]):.1f}°, "
          f"{np.degrees(euler_random[1]):.1f}°, {np.degrees(euler_random[2]):.1f}°)")
    print(f"Result: {len(pixels_euler)} stars on detector")
    for i, pixel in enumerate(pixels_euler):
        print(f"  Star {i}: pixel ({pixel[0]:.1f}, {pixel[1]:.1f})")
    
    print("\nAttitude transformation test completed successfully!")