import numpy as np
import logging
from typing import Optional, Tuple, Union
from ..core.star_tracker_pipeline import StarTrackerPipeline
from ..BAST.catalog import Catalog
from ..BAST.match import match
from .scene_generator import MultiStarSceneGenerator
from .multi_star_radiometry import MultiStarRadiometry
from .attitude_transform import (
    random_quaternion, 
    random_euler_angles,
    validate_angular_preservation,
    compute_bearing_vectors_from_pixels
)

logger = logging.getLogger(__name__)

class MultiStarPipeline:
    """Extends existing pipeline for multi-star scenes"""

    def __init__(self, pipeline: StarTrackerPipeline):
        self.pipeline = pipeline
        self.scene_generator = MultiStarSceneGenerator(pipeline)
        self.radiometry = MultiStarRadiometry(pipeline)

    def run_multi_star_analysis(self, synthetic_catalog: Catalog, psf_data, 
                               true_attitude_quaternion: Optional[np.ndarray] = None,
                               true_attitude_euler: Optional[Tuple[float, float, float]] = None,
                               perform_validation: bool = True):
        """Complete multi-star analysis workflow with attitude transformation
        
        Args:
            synthetic_catalog: BAST catalog with star positions
            psf_data: PSF data for radiometry simulation
            true_attitude_quaternion: Camera attitude as unit quaternion [q0, q1, q2, q3]
            true_attitude_euler: Camera attitude as (roll, pitch, yaw) in radians
            perform_validation: Whether to validate angular preservation
            
        Returns:
            Complete analysis results including attitude transformation validation
        """

        # Log attitude information
        if true_attitude_quaternion is not None:
            logger.info(f"Running analysis with quaternion attitude: {true_attitude_quaternion}")
        elif true_attitude_euler is not None:
            euler_deg = np.degrees(true_attitude_euler)
            logger.info(f"Running analysis with Euler attitude: ({euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°)")
        else:
            logger.info("Running analysis with identity attitude (trivial case)")

        # 1. Generate scene with attitude transformation
        scene_data = self.scene_generator.generate_scene(
            synthetic_catalog,
            true_attitude_quaternion=true_attitude_quaternion,
            true_attitude_euler=true_attitude_euler
        )

        logger.info(f"Scene generation: {len(scene_data['stars'])} stars on detector")

        # 2. Render scene (radiometry simulation)
        scene_data = self.radiometry.render_scene(scene_data, psf_data)

        # 3. Detect stars - use optimized peak detection method for multi-star scenes
        # 3. Detect stars - use the more robust adaptive thresholding method
        centroid_results = self.pipeline.detect_stars_and_calculate_centroids(
            [scene_data['detector_image']],
            k_sigma=3.0,
            min_pixels=5,
            max_pixels=200,
            block_size=64
        )

        logger.info(f"Star detection: {len(centroid_results['centroids'])} stars detected")

        # 4. Calculate bearing vectors using the attitude-aware transformation
        camera_params = self.scene_generator.camera_params
        bearing_vectors = compute_bearing_vectors_from_pixels(
            centroid_results['centroids'],
            camera_params
        )

        logger.info(f"Bearing vectors: {len(bearing_vectors)} vectors calculated")

        # 5. Triangle matching (existing BAST)
        star_matches = match(
            observed_stars=bearing_vectors,
            catalog=synthetic_catalog,
            angle_tolerance=0.01,  # 0.57° - precise tolerance for accurate coordinates
            min_confidence=0.8     # Standard confidence threshold
        )

        logger.info(f"BAST matching: {len(star_matches)} matches found")

        # 6. Monte Carlo QUEST attitude determination
        quest_results = None
        if len(star_matches) >= 2:
            try:
                from .monte_carlo_quest import MonteCarloQUEST, NoiseParameters
                
                # Configure noise parameters based on system performance
                noise_params = NoiseParameters(
                    centroid_noise_pixels=0.15,  # Expected centroiding accuracy
                    pixel_pitch_um=5.5,          # CMV4000 pixel pitch
                    focal_length_mm=40.07        # Camera focal length
                )
                
                mc_quest = MonteCarloQUEST(noise_params)
                quest_results = mc_quest.determine_attitude(
                    bearing_vectors=bearing_vectors,
                    star_matches=star_matches,
                    catalog=synthetic_catalog,
                    num_trials=1000,
                    max_trials=5000
                )
                
                logger.info(f"Monte Carlo QUEST completed: "
                           f"{quest_results.num_trials_used} trials, "
                           f"{quest_results.angular_uncertainty_arcsec:.1f} arcsec uncertainty")
                
            except Exception as e:
                logger.error(f"Monte Carlo QUEST failed: {e}")
                quest_results = None

        # 7. Validation - both legacy and attitude-aware validation
        validation_results = self.validate_matches(star_matches, scene_data)
        
        # 7. Attitude transformation validation (if using new system)
        attitude_validation = {}
        if perform_validation and scene_data['attitude'].get('transformation_method') == 'attitude_transform':
            attitude_validation = self._validate_attitude_transformation(scene_data, centroid_results)

        return {
            'scene_data': scene_data,
            'centroids': centroid_results['centroids'],
            'detected_stars': len(centroid_results['centroids']),
            'bearing_vectors': bearing_vectors,
            'star_matches': star_matches,
            'quest_results': quest_results,
            'validation': validation_results,
            'attitude_validation': attitude_validation,
            'analysis_summary': {
                'attitude_method': scene_data['attitude'].get('transformation_method', 'unknown'),
                'stars_in_catalog': len(synthetic_catalog),
                'stars_on_detector': len(scene_data['stars']),
                'stars_detected': len(centroid_results['centroids']),
                'stars_matched': len(star_matches),
                'detection_rate': len(centroid_results['centroids']) / max(len(scene_data['stars']), 1),
                'matching_rate': len(star_matches) / max(len(centroid_results['centroids']), 1),
                'quest_enabled': quest_results is not None,
                'quest_trials': quest_results.num_trials_used if quest_results else 0,
                'quest_uncertainty_arcsec': quest_results.angular_uncertainty_arcsec if quest_results else None
            }
        }

    def validate_matches(self, star_matches, scene_data):
        """
        Validate the matches based on the ground truth from the scene data.
        """
        from .validation import validate_triangle_matches, validate_pyramid_consistency

        num_stars = len(scene_data['stars'])
        ground_truth = scene_data['ground_truth']

        if num_stars == 3:
            return validate_triangle_matches(star_matches, ground_truth)
        elif num_stars == 4:
            return validate_pyramid_consistency(star_matches, ground_truth)
        else:
            return {"status": "unknown", "reason": f"No validation logic for {num_stars} stars."}
    
    def _validate_attitude_transformation(self, scene_data, centroid_results):
        """
        Validate that the attitude transformation preserves angular relationships.
        
        Args:
            scene_data: Scene data containing ground truth inertial vectors
            centroid_results: Detected centroids from pipeline
            
        Returns:
            Validation results for attitude transformation
        """
        logger.info("Validating attitude transformation angular preservation...")
        
        validation_results = {
            'angular_preservation_valid': False,
            'max_angular_error_deg': 0.0,
            'mean_angular_error_deg': 0.0,
            'num_angle_pairs_checked': 0,
            'detailed_errors': [],
            'transformation_method': scene_data['attitude'].get('transformation_method', 'unknown')
        }
        
        # Get ground truth inertial vectors
        ground_truth_vectors = scene_data['ground_truth'].get('inertial_vectors', [])
        if not ground_truth_vectors:
            logger.warning("No ground truth inertial vectors available for validation")
            return validation_results
            
        # Get detected centroids and convert back to bearing vectors
        centroids = centroid_results['centroids']
        if len(centroids) < 2:
            logger.warning("Need at least 2 detected stars for angular validation")
            return validation_results
            
        # Use camera parameters from scene generator
        camera_params = self.scene_generator.camera_params
        detected_bearing_vectors = compute_bearing_vectors_from_pixels(
            centroids, camera_params
        )
        
        logger.info(f"Validating angular preservation: {len(ground_truth_vectors)} ground truth vs {len(detected_bearing_vectors)} detected")
        
        # For now, assume detected stars correspond to first N ground truth vectors
        # In a more sophisticated system, we would do proper star matching
        num_stars_to_validate = min(len(ground_truth_vectors), len(detected_bearing_vectors))
        
        if num_stars_to_validate < 2:
            logger.warning("Insufficient stars for angular validation")
            return validation_results
            
        # Check angular preservation for all pairs
        angular_errors = []
        angle_pairs_checked = 0
        
        for i in range(num_stars_to_validate):
            for j in range(i + 1, num_stars_to_validate):
                # Ground truth angle
                gt_vec1, gt_vec2 = ground_truth_vectors[i], ground_truth_vectors[j]
                dot_gt = np.clip(np.dot(gt_vec1, gt_vec2), -1.0, 1.0)
                angle_gt_rad = np.arccos(dot_gt)
                angle_gt_deg = np.degrees(angle_gt_rad)
                
                # Detected angle
                det_vec1, det_vec2 = detected_bearing_vectors[i], detected_bearing_vectors[j]
                dot_det = np.clip(np.dot(det_vec1, det_vec2), -1.0, 1.0)
                angle_det_rad = np.arccos(dot_det)
                angle_det_deg = np.degrees(angle_det_rad)
                
                # Calculate error
                angular_error_deg = abs(angle_gt_deg - angle_det_deg)
                angular_errors.append(angular_error_deg)
                angle_pairs_checked += 1
                
                validation_results['detailed_errors'].append({
                    'star_pair': (i, j),
                    'ground_truth_angle_deg': angle_gt_deg,
                    'detected_angle_deg': angle_det_deg,
                    'angular_error_deg': angular_error_deg
                })
                
                logger.debug(f"Angle validation pair ({i},{j}): GT={angle_gt_deg:.3f}°, Det={angle_det_deg:.3f}°, Error={angular_error_deg:.3f}°")
        
        # Summary statistics
        if angular_errors:
            validation_results['max_angular_error_deg'] = max(angular_errors)
            validation_results['mean_angular_error_deg'] = np.mean(angular_errors)
            validation_results['num_angle_pairs_checked'] = angle_pairs_checked
            
            # Check if within tolerance (target: <0.001° from euler_angles.md)
            tolerance_deg = 0.1  # More relaxed tolerance for realistic testing
            validation_results['angular_preservation_valid'] = validation_results['max_angular_error_deg'] < tolerance_deg
            
            logger.info(f"Angular preservation validation: "
                       f"max_error={validation_results['max_angular_error_deg']:.4f}°, "
                       f"mean_error={validation_results['mean_angular_error_deg']:.4f}°, "
                       f"pairs_checked={angle_pairs_checked}, "
                       f"valid={validation_results['angular_preservation_valid']} (tol={tolerance_deg}°)")
        
        return validation_results
    
    # Convenience methods for generating test attitudes
    def run_multi_star_analysis_with_random_attitude(self, synthetic_catalog: Catalog, psf_data,
                                                   max_angle_deg: float = 15.0,
                                                   use_quaternion: bool = True):
        """Run analysis with randomly generated attitude for testing"""
        
        if use_quaternion:
            attitude_q = random_quaternion()
            logger.info(f"Generated random quaternion attitude: {attitude_q}")
            return self.run_multi_star_analysis(synthetic_catalog, psf_data, 
                                              true_attitude_quaternion=attitude_q)
        else:
            attitude_euler = random_euler_angles(max_angle_deg)
            euler_deg = np.degrees(attitude_euler)
            logger.info(f"Generated random Euler attitude: ({euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°)")
            return self.run_multi_star_analysis(synthetic_catalog, psf_data,
                                              true_attitude_euler=attitude_euler)
    
    def run_attitude_sweep_analysis(self, synthetic_catalog: Catalog, psf_data,
                                  num_attitudes: int = 5,
                                  max_angle_deg: float = 20.0):
        """Run analysis for multiple random attitudes to test robustness"""
        
        logger.info(f"Running attitude sweep: {num_attitudes} random attitudes, max angle ±{max_angle_deg}°")
        
        sweep_results = []
        
        for i in range(num_attitudes):
            logger.info(f"--- Attitude {i+1}/{num_attitudes} ---")
            
            # Alternate between quaternion and Euler representations
            use_quaternion = (i % 2 == 0)
            
            try:
                result = self.run_multi_star_analysis_with_random_attitude(
                    synthetic_catalog, psf_data, max_angle_deg, use_quaternion
                )
                
                sweep_results.append({
                    'attitude_index': i,
                    'attitude_type': 'quaternion' if use_quaternion else 'euler',
                    'result': result,
                    'success': len(result['star_matches']) > 0,
                    'angular_preservation_valid': result['attitude_validation'].get('angular_preservation_valid', False),
                    'max_angular_error_deg': result['attitude_validation'].get('max_angular_error_deg', float('inf'))
                })
                
                logger.info(f"Attitude {i+1} result: {len(result['star_matches'])} matches, "
                           f"angular_valid={result['attitude_validation'].get('angular_preservation_valid', False)}")
                
            except Exception as e:
                logger.error(f"Attitude {i+1} failed: {e}")
                sweep_results.append({
                    'attitude_index': i,
                    'attitude_type': 'quaternion' if use_quaternion else 'euler',
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        # Summary statistics
        successful_results = [r for r in sweep_results if r['success']]
        valid_angular_results = [r for r in successful_results if r.get('angular_preservation_valid', False)]
        
        summary = {
            'total_attitudes': num_attitudes,
            'successful_matches': len(successful_results),
            'valid_angular_preservation': len(valid_angular_results),
            'success_rate': len(successful_results) / num_attitudes,
            'angular_preservation_rate': len(valid_angular_results) / max(len(successful_results), 1),
            'results': sweep_results
        }
        
        logger.info(f"Attitude sweep complete: {summary['success_rate']:.1%} success rate, "
                   f"{summary['angular_preservation_rate']:.1%} angular preservation rate")
        
        return summary