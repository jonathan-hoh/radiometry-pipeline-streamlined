#!/usr/bin/env python3
"""
Bijective Centroid-to-Catalog Matching

Implements proper one-to-one matching algorithms to fix the duplicate matching
issue identified in the attitude error comparison analysis.

Author: Claude Code Assistant
Date: 2025-09-02
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def compute_distance_matrix(centroids: List[Tuple[float, float]], 
                          scene_stars: List[Dict], 
                          max_distance: float = 50.0) -> np.ndarray:
    """
    Compute distance matrix between centroids and scene stars.
    
    Args:
        centroids: List of (x, y) centroid coordinates
        scene_stars: List of scene star dictionaries with position information
        max_distance: Maximum distance for valid matches (pixels)
        
    Returns:
        Distance matrix where [i,j] is distance from centroid i to star j
        Distances > max_distance are set to infinity
    """
    n_centroids = len(centroids)
    n_stars = len(scene_stars)
    
    distance_matrix = np.full((n_centroids, n_stars), np.inf)
    
    for i, centroid in enumerate(centroids):
        centroid_pos = np.array(centroid)
        
        for j, star in enumerate(scene_stars):
            # Get star position from scene data
            if 'actual_psf_center' in star:
                star_pos = np.array(star['actual_psf_center'])
            else:
                star_pos = np.array(star['detector_position'])
            
            distance = np.linalg.norm(centroid_pos - star_pos)
            
            if distance <= max_distance:
                distance_matrix[i, j] = distance
    
    return distance_matrix

def hungarian_matching(distance_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Perform optimal bijective matching using Hungarian algorithm.
    
    Args:
        distance_matrix: Matrix of distances between centroids and stars
        
    Returns:
        List of (centroid_idx, star_idx, distance) tuples for optimal matches
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        logger.error("scipy not available for Hungarian algorithm. Falling back to greedy matching.")
        return greedy_bijective_matching(distance_matrix)
    
    n_centroids, n_stars = distance_matrix.shape
    
    # Check if matrix has any finite values
    finite_mask = np.isfinite(distance_matrix)
    if not np.any(finite_mask):
        logger.warning("No finite distances in matrix. No matches possible.")
        return []
    
    # For Hungarian algorithm, we need to handle infinite values
    # Replace infinite values with a large finite number
    max_finite_distance = np.max(distance_matrix[finite_mask]) if np.any(finite_mask) else 1000.0
    large_penalty = max_finite_distance * 10  # Large but finite penalty
    
    # Create working matrix with finite values only
    working_matrix = distance_matrix.copy()
    working_matrix[~finite_mask] = large_penalty
    
    # Pad matrix to make it square for Hungarian algorithm
    max_dim = max(n_centroids, n_stars)
    padded_matrix = np.full((max_dim, max_dim), large_penalty)
    padded_matrix[:n_centroids, :n_stars] = working_matrix
    
    try:
        # Run Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(padded_matrix)
        
        # Extract valid matches (not padded entries and originally finite distances)
        matches = []
        for row, col in zip(row_indices, col_indices):
            if (row < n_centroids and col < n_stars and 
                finite_mask[row, col]):  # Use original finite mask
                matches.append((row, col, distance_matrix[row, col]))
        
        logger.info(f"Hungarian algorithm found {len(matches)} optimal matches")
        return matches
        
    except ValueError as e:
        logger.warning(f"Hungarian algorithm failed: {e}. Falling back to greedy matching.")
        return greedy_bijective_matching(distance_matrix)

def greedy_bijective_matching(distance_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Perform greedy bijective matching as fallback when scipy is not available.
    
    Args:
        distance_matrix: Matrix of distances between centroids and stars
        
    Returns:
        List of (centroid_idx, star_idx, distance) tuples for greedy matches
    """
    n_centroids, n_stars = distance_matrix.shape
    
    # Create list of all valid matches with their distances
    potential_matches = []
    for i in range(n_centroids):
        for j in range(n_stars):
            if np.isfinite(distance_matrix[i, j]):
                potential_matches.append((i, j, distance_matrix[i, j]))
    
    # Sort by distance (greedy: take closest matches first)
    potential_matches.sort(key=lambda x: x[2])
    
    # Track which centroids and stars have been matched
    matched_centroids = set()
    matched_stars = set()
    final_matches = []
    
    for centroid_idx, star_idx, distance in potential_matches:
        if centroid_idx not in matched_centroids and star_idx not in matched_stars:
            final_matches.append((centroid_idx, star_idx, distance))
            matched_centroids.add(centroid_idx)
            matched_stars.add(star_idx)
    
    logger.info(f"Greedy algorithm found {len(final_matches)} bijective matches")
    return final_matches

def bijective_centroid_to_catalog_matching(centroids: List[Tuple[float, float]], 
                                         scene_stars: List[Dict],
                                         max_distance: float = 50.0,
                                         use_hungarian: bool = True) -> Dict:
    """
    Perform bijective (one-to-one) matching between centroids and catalog stars.
    
    Args:
        centroids: List of detected centroid positions
        scene_stars: List of scene star data with positions and catalog info
        max_distance: Maximum distance for valid matches (pixels)
        use_hungarian: If True, use optimal Hungarian algorithm; else use greedy
        
    Returns:
        Dictionary containing matching results and statistics
    """
    if not centroids or not scene_stars:
        logger.warning("Empty centroids or scene stars provided")
        return {
            'matches': [],
            'unmatched_centroids': list(range(len(centroids))),
            'unmatched_stars': list(range(len(scene_stars))),
            'matching_stats': {
                'total_centroids': len(centroids),
                'total_stars': len(scene_stars),
                'matches_found': 0,
                'match_rate': 0.0,
                'mean_distance': np.nan,
                'max_distance': np.nan
            }
        }
    
    logger.info(f"Performing bijective matching: {len(centroids)} centroids vs {len(scene_stars)} stars")
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(centroids, scene_stars, max_distance)
    
    # Perform matching
    if use_hungarian:
        raw_matches = hungarian_matching(distance_matrix)
    else:
        raw_matches = greedy_bijective_matching(distance_matrix)
    
    # Convert to match result format
    matches = []
    matched_centroid_indices = set()
    matched_star_indices = set()
    
    for centroid_idx, star_idx, distance in raw_matches:
        star = scene_stars[star_idx]
        matches.append({
            'centroid_idx': centroid_idx,
            'star_idx': star_idx,
            'centroid_pos': centroids[centroid_idx],
            'star_id': star['star_id'],
            'catalog_idx': star['catalog_idx'],
            'distance': distance
        })
        matched_centroid_indices.add(centroid_idx)
        matched_star_indices.add(star_idx)
    
    # Identify unmatched centroids and stars
    unmatched_centroids = [i for i in range(len(centroids)) 
                          if i not in matched_centroid_indices]
    unmatched_stars = [i for i in range(len(scene_stars)) 
                      if i not in matched_star_indices]
    
    # Calculate statistics
    distances = [match['distance'] for match in matches]
    match_rate = len(matches) / max(len(centroids), len(scene_stars))
    
    matching_stats = {
        'total_centroids': len(centroids),
        'total_stars': len(scene_stars),
        'matches_found': len(matches),
        'match_rate': match_rate,
        'mean_distance': np.mean(distances) if distances else np.nan,
        'max_distance': np.max(distances) if distances else np.nan,
        'unmatched_centroids_count': len(unmatched_centroids),
        'unmatched_stars_count': len(unmatched_stars)
    }
    
    logger.info(f"Bijective matching complete: {len(matches)} matches, "
                f"{match_rate:.1%} match rate, "
                f"mean distance {matching_stats['mean_distance']:.2f}px")
    
    return {
        'matches': matches,
        'unmatched_centroids': unmatched_centroids,
        'unmatched_stars': unmatched_stars,
        'matching_stats': matching_stats
    }

def validate_bijective_matching(matching_result: Dict) -> bool:
    """
    Validate that the matching result is truly bijective (one-to-one).
    
    Args:
        matching_result: Result from bijective_centroid_to_catalog_matching()
        
    Returns:
        True if matching is valid and bijective, False otherwise
    """
    matches = matching_result['matches']
    
    if not matches:
        return True  # Empty matching is trivially bijective
    
    # Check for duplicate centroid indices
    centroid_indices = [match['centroid_idx'] for match in matches]
    if len(set(centroid_indices)) != len(centroid_indices):
        logger.error("Validation failed: Duplicate centroid indices found")
        return False
    
    # Check for duplicate star indices
    star_indices = [match['star_idx'] for match in matches]
    if len(set(star_indices)) != len(star_indices):
        logger.error("Validation failed: Duplicate star indices found")
        return False
    
    # Check for duplicate star IDs (additional safety check)
    star_ids = [match['star_id'] for match in matches]
    if len(set(star_ids)) != len(star_ids):
        logger.error("Validation failed: Duplicate star IDs found")
        return False
    
    logger.debug(f"Bijective matching validation passed: {len(matches)} unique matches")
    return True