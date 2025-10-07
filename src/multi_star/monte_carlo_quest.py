#!/usr/bin/env python3
"""
monte_carlo_quest.py - Monte Carlo QUEST attitude determination

Implements Monte Carlo approach to QUEST algorithm for robust attitude determination
with realistic error propagation and confidence bounds.

Usage:
    from src.multi_star.monte_carlo_quest import MonteCarloQUEST
    
    mc_quest = MonteCarloQUEST()
    results = mc_quest.determine_attitude(
        bearing_vectors, star_matches, catalog, 
        num_trials=1000
    )
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

from ..BAST.resolve import build_davenport_matrix, quest_algorithm, quaternion_to_matrix
from ..BAST.match import StarMatch

logger = logging.getLogger(__name__)

@dataclass
class NoiseParameters:
    """Parameters for Monte Carlo noise simulation"""
    centroid_noise_pixels: float = 0.1  # RMS centroiding error in pixels
    pixel_pitch_um: float = 5.5  # CMV4000 pixel pitch
    focal_length_mm: float = 40.07  # Camera focal length
    
    @property
    def angular_noise_rad(self) -> float:
        """Convert pixel noise to angular noise in radians"""
        # Convert pixels to microns, then to angular error
        noise_um = self.centroid_noise_pixels * self.pixel_pitch_um
        noise_mm = noise_um / 1000.0
        return noise_mm / self.focal_length_mm

@dataclass
class ConvergenceMetrics:
    """Convergence analysis for Monte Carlo sampling"""
    converged: bool
    convergence_trial: int
    quaternion_std: float
    angular_std_arcsec: float

@dataclass
class MonteCarloResults:
    """Complete Monte Carlo QUEST results"""
    optimal_quaternion: np.ndarray
    rotation_matrix: np.ndarray
    quaternion_mean: np.ndarray
    quaternion_std: np.ndarray
    quaternion_covariance: np.ndarray
    eigenvalue_mean: float
    eigenvalue_std: float
    confidence_3sigma: float
    angular_uncertainty_arcsec: float
    num_trials_used: int
    computation_time: float
    convergence_metrics: ConvergenceMetrics
    trial_quaternions: np.ndarray  # For detailed analysis
    trial_eigenvalues: np.ndarray

class MonteCarloQUEST:
    """Monte Carlo QUEST attitude determination"""
    
    def __init__(self, noise_params: Optional[NoiseParameters] = None):
        """
        Initialize Monte Carlo QUEST solver
        
        Args:
            noise_params: Noise parameters for Monte Carlo simulation
        """
        self.noise_params = noise_params or NoiseParameters()
        logger.info(f"Initialized Monte Carlo QUEST with angular noise: "
                   f"{np.degrees(self.noise_params.angular_noise_rad) * 3600:.1f} arcsec")
    
    def add_bearing_vector_noise(self, vectors: List[np.ndarray], 
                                trial_idx: int) -> List[np.ndarray]:
        """
        Add realistic noise to bearing vectors for Monte Carlo trial
        
        Args:
            vectors: List of unit bearing vectors
            trial_idx: Trial number for reproducible randomization
            
        Returns:
            List of noisy bearing vectors (still normalized)
        """
        # Set seed for reproducible trials
        np.random.seed(trial_idx)
        
        noisy_vectors = []
        angular_noise = self.noise_params.angular_noise_rad
        
        for vector in vectors:
            # Add small random perturbations in spherical coordinates
            # This preserves the unit sphere constraint better than Cartesian noise
            
            # Convert to spherical
            r = np.linalg.norm(vector)  # Should be 1.0 for unit vectors
            theta = np.arccos(vector[2] / r)  # Polar angle
            phi = np.arctan2(vector[1], vector[0])  # Azimuthal angle
            
            # Add noise to angles
            theta_noisy = theta + np.random.normal(0, angular_noise)
            phi_noisy = phi + np.random.normal(0, angular_noise)
            
            # Convert back to Cartesian (ensuring unit length)
            noisy_vector = np.array([
                np.sin(theta_noisy) * np.cos(phi_noisy),
                np.sin(theta_noisy) * np.sin(phi_noisy),
                np.cos(theta_noisy)
            ])
            
            # Normalize to ensure unit length
            noisy_vector = noisy_vector / np.linalg.norm(noisy_vector)
            noisy_vectors.append(noisy_vector)
        
        return noisy_vectors
    
    def extract_matched_vectors(self, bearing_vectors: List[np.ndarray],
                              star_matches: List[StarMatch], 
                              catalog: Any) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract matched observed and catalog vectors from BAST results
        
        Args:
            bearing_vectors: List of observed bearing vectors
            star_matches: List of StarMatch objects from BAST
            catalog: Star catalog object
            
        Returns:
            Tuple of (matched_observed_vectors, matched_catalog_vectors)
        """
        matched_observed = []
        matched_catalog = []
        
        for match in star_matches:
            # Get observed vector
            obs_vector = bearing_vectors[match.observed_idx]
            matched_observed.append(obs_vector)
            
            # Get catalog vector - extract from catalog DataFrame
            catalog_row = catalog.iloc[match.catalog_idx]
            ra, dec = catalog_row['RA'], catalog_row['DE']
            
            # Convert RA/Dec to unit vector
            catalog_vector = np.array([
                np.cos(dec) * np.cos(ra),
                np.cos(dec) * np.sin(ra),
                np.sin(dec)
            ])
            matched_catalog.append(catalog_vector)
        
        logger.info(f"Extracted {len(matched_observed)} matched vector pairs")
        return matched_observed, matched_catalog
    
    def check_convergence(self, quaternions: np.ndarray, 
                         window_size: int = 100,
                         tolerance: float = 1e-4) -> ConvergenceMetrics:
        """
        Check if Monte Carlo sampling has converged
        
        Args:
            quaternions: Array of trial quaternions (trials x 4)
            window_size: Window size for convergence analysis
            tolerance: Convergence tolerance for quaternion std
            
        Returns:
            ConvergenceMetrics with convergence status
        """
        num_trials = len(quaternions)
        
        if num_trials < window_size * 2:
            return ConvergenceMetrics(False, num_trials, 1.0, 3600.0)
        
        # Calculate rolling standard deviation
        for trial in range(window_size, num_trials, window_size//4):
            window_quaternions = quaternions[trial-window_size:trial]
            
            # Calculate quaternion standard deviation
            q_mean = np.mean(window_quaternions, axis=0)
            q_std = np.std(window_quaternions, axis=0)
            max_std = np.max(q_std)
            
            # Convert to angular uncertainty (approximate)
            angular_std_arcsec = np.degrees(2 * max_std) * 3600
            
            if max_std < tolerance:
                return ConvergenceMetrics(True, trial, max_std, angular_std_arcsec)
        
        # Not converged
        final_std = np.max(np.std(quaternions, axis=0))
        angular_std = np.degrees(2 * final_std) * 3600
        return ConvergenceMetrics(False, num_trials, final_std, angular_std)
    
    def compute_batch_quest(self, observed_vectors_list: List[List[np.ndarray]],
                           catalog_vectors: List[np.ndarray],
                           batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute QUEST algorithm for multiple trials efficiently
        
        Args:
            observed_vectors_list: List of observed vector lists (one per trial)
            catalog_vectors: Reference catalog vectors (same for all trials)
            batch_size: Process in batches to manage memory
            
        Returns:
            Tuple of (quaternions array, eigenvalues array)
        """
        num_trials = len(observed_vectors_list)
        quaternions = np.zeros((num_trials, 4))
        eigenvalues = np.zeros(num_trials)
        
        # Process in batches
        for batch_start in range(0, num_trials, batch_size):
            batch_end = min(batch_start + batch_size, num_trials)
            batch_trials = batch_end - batch_start
            
            # Build K matrices for this batch
            K_batch = np.zeros((batch_trials, 4, 4))
            
            for i, trial_idx in enumerate(range(batch_start, batch_end)):
                observed_vectors = observed_vectors_list[trial_idx]
                K_batch[i] = build_davenport_matrix(observed_vectors, catalog_vectors)
            
            # Batch eigenvalue computation
            try:
                eigenvals, eigenvecs = np.linalg.eig(K_batch)
                
                # Process each trial in the batch
                for i in range(batch_trials):
                    trial_eigenvals = eigenvals[i]
                    trial_eigenvecs = eigenvecs[i]
                    
                    # Find maximum eigenvalue
                    max_idx = np.argmax(trial_eigenvals.real)
                    max_eigenval = trial_eigenvals[max_idx].real
                    max_eigenvec = trial_eigenvecs[:, max_idx].real
                    
                    # Normalize quaternion
                    quaternion = max_eigenvec / np.linalg.norm(max_eigenvec)
                    
                    # Ensure positive scalar component
                    if quaternion[0] < 0:
                        quaternion = -quaternion
                    
                    quaternions[batch_start + i] = quaternion
                    eigenvalues[batch_start + i] = max_eigenval
                    
            except np.linalg.LinAlgError as e:
                logger.error(f"Eigenvalue computation failed for batch {batch_start}-{batch_end}: {e}")
                # Fill with identity quaternion for failed trials
                for i in range(batch_trials):
                    quaternions[batch_start + i] = np.array([1.0, 0.0, 0.0, 0.0])
                    eigenvalues[batch_start + i] = 1.0
        
        return quaternions, eigenvalues
    
    def determine_attitude(self, bearing_vectors: List[np.ndarray],
                          star_matches: List[StarMatch],
                          catalog: Any,
                          num_trials: int = 1000,
                          max_trials: int = 10000,
                          enable_convergence_check: bool = True) -> MonteCarloResults:
        """
        Determine spacecraft attitude using Monte Carlo QUEST
        
        Args:
            bearing_vectors: List of observed bearing vectors
            star_matches: List of StarMatch objects from BAST
            catalog: Star catalog object
            num_trials: Target number of Monte Carlo trials
            max_trials: Maximum number of trials if not converged
            enable_convergence_check: Enable early termination on convergence
            
        Returns:
            MonteCarloResults with attitude and uncertainty estimates
            
        Raises:
            ValueError: If insufficient star matches provided
            RuntimeError: If attitude determination fails
        """
        start_time = time.time()
        
        # Validate inputs
        if len(star_matches) < 2:
            raise ValueError(f"Insufficient star matches for attitude determination: {len(star_matches)} < 2")
        
        logger.info(f"Starting Monte Carlo QUEST with {len(star_matches)} star matches")
        logger.info(f"Target trials: {num_trials}, Max trials: {max_trials}")
        
        # Extract matched vectors
        matched_observed, matched_catalog = self.extract_matched_vectors(
            bearing_vectors, star_matches, catalog
        )
        
        # Generate noisy observations for all trials
        logger.info("Generating Monte Carlo trials...")
        observed_vectors_list = []
        
        for trial in range(num_trials):
            noisy_vectors = self.add_bearing_vector_noise(matched_observed, trial)
            observed_vectors_list.append(noisy_vectors)
        
        # Compute QUEST for all trials
        logger.info("Computing batch QUEST solutions...")
        quaternions, eigenvalues = self.compute_batch_quest(
            observed_vectors_list, matched_catalog
        )
        
        # Check convergence if enabled
        convergence_metrics = ConvergenceMetrics(False, num_trials, 1.0, 3600.0)
        trials_used = num_trials
        
        if enable_convergence_check and num_trials < max_trials:
            convergence_metrics = self.check_convergence(quaternions)
            if convergence_metrics.converged:
                trials_used = convergence_metrics.convergence_trial
                quaternions = quaternions[:trials_used]
                eigenvalues = eigenvalues[:trials_used]
                logger.info(f"Converged after {trials_used} trials")
            elif num_trials < max_trials:
                logger.info(f"Not converged after {num_trials} trials, continuing...")
                # Could implement additional trials here if needed
        
        # Statistical analysis
        quaternion_mean = np.mean(quaternions, axis=0)
        quaternion_std = np.std(quaternions, axis=0)
        quaternion_cov = np.cov(quaternions.T)
        
        eigenvalue_mean = np.mean(eigenvalues)
        eigenvalue_std = np.std(eigenvalues)
        
        # Find optimal quaternion (highest eigenvalue trial)
        best_trial_idx = np.argmax(eigenvalues)
        optimal_quaternion = quaternions[best_trial_idx]
        rotation_matrix = quaternion_to_matrix(optimal_quaternion)
        
        # Calculate uncertainty metrics
        max_quaternion_std = np.max(quaternion_std)
        angular_uncertainty_arcsec = np.degrees(2 * max_quaternion_std) * 3600
        confidence_3sigma = 1.0 - 2 * (1 - 0.9973)  # Approximate 3-sigma confidence
        
        computation_time = time.time() - start_time
        
        logger.info(f"Monte Carlo QUEST completed:")
        logger.info(f"  Trials used: {trials_used}")
        logger.info(f"  Computation time: {computation_time:.2f}s")
        logger.info(f"  Optimal eigenvalue: {eigenvalues[best_trial_idx]:.4f}")
        logger.info(f"  Angular uncertainty: {angular_uncertainty_arcsec:.1f} arcsec")
        logger.info(f"  Quaternion std: {max_quaternion_std:.6f}")
        
        return MonteCarloResults(
            optimal_quaternion=optimal_quaternion,
            rotation_matrix=rotation_matrix,
            quaternion_mean=quaternion_mean,
            quaternion_std=quaternion_std,
            quaternion_covariance=quaternion_cov,
            eigenvalue_mean=eigenvalue_mean,
            eigenvalue_std=eigenvalue_std,
            confidence_3sigma=confidence_3sigma,
            angular_uncertainty_arcsec=angular_uncertainty_arcsec,
            num_trials_used=trials_used,
            computation_time=computation_time,
            convergence_metrics=convergence_metrics,
            trial_quaternions=quaternions,
            trial_eigenvalues=eigenvalues
        )

def validate_monte_carlo_quest(results: MonteCarloResults, 
                              ground_truth_quaternion: np.ndarray) -> Dict[str, float]:
    """
    Validate Monte Carlo QUEST results against ground truth
    
    Args:
        results: Monte Carlo QUEST results
        ground_truth_quaternion: Known true attitude quaternion
        
    Returns:
        Dictionary with validation metrics
    """
    # Quaternion error (angular difference)
    q_opt = results.optimal_quaternion
    q_true = ground_truth_quaternion
    
    # Ensure same sign convention
    if np.dot(q_opt, q_true) < 0:
        q_true = -q_true
    
    # Angular error between quaternions
    dot_product = np.abs(np.dot(q_opt, q_true))
    angular_error_rad = 2 * np.arccos(np.clip(dot_product, 0, 1))
    angular_error_arcsec = np.degrees(angular_error_rad) * 3600
    
    # Check if true attitude falls within uncertainty bounds
    uncertainty_ratio = angular_error_arcsec / results.angular_uncertainty_arcsec
    within_bounds = uncertainty_ratio <= 3.0  # 3-sigma test
    
    return {
        'angular_error_arcsec': angular_error_arcsec,
        'angular_error_deg': np.degrees(angular_error_rad),
        'uncertainty_arcsec': results.angular_uncertainty_arcsec,
        'uncertainty_ratio': uncertainty_ratio,
        'within_3sigma_bounds': within_bounds,
        'optimal_eigenvalue': np.max(results.trial_eigenvalues),
        'solution_quality': 'excellent' if angular_error_arcsec < 10 else 'good' if angular_error_arcsec < 60 else 'poor'
    }

if __name__ == "__main__":
    # Simple test case
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    test_bearing_vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]
    
    # Mock catalog (would normally be loaded)
    class MockCatalog:
        def iloc(self, idx):
            # Return mock RA/Dec for test
            mock_coords = [
                {'RA': 0.0, 'DE': 0.0},      # +X direction
                {'RA': np.pi/2, 'DE': 0.0},  # +Y direction  
                {'RA': 0.0, 'DE': np.pi/2}   # +Z direction
            ]
            return type('MockRow', (), mock_coords[idx])()
    
    # Mock star matches
    test_matches = [
        StarMatch(0, 0, 2, 0.95),
        StarMatch(1, 1, 2, 0.90),
        StarMatch(2, 2, 2, 0.85)
    ]
    
    # Run Monte Carlo QUEST
    mc_quest = MonteCarloQUEST()
    results = mc_quest.determine_attitude(
        test_bearing_vectors, test_matches, MockCatalog(), num_trials=100
    )
    
    print("Monte Carlo QUEST Test Results:")
    print(f"Optimal quaternion: {results.optimal_quaternion}")
    print(f"Angular uncertainty: {results.angular_uncertainty_arcsec:.1f} arcsec")
    print(f"Computation time: {results.computation_time:.3f}s")