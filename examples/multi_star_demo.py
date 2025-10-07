#!/usr/bin/env python3
"""
Multi-Star Scene Demonstration

A self-contained example showing the complete multi-star pipeline including
scene generation, attitude transformation, detection, and BAST matching.

Usage:
    PYTHONPATH=. python examples/multi_star_demo.py
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.multi_star.multi_star_pipeline import MultiStarPipeline
from src.BAST.catalog import from_csv

def multi_star_demo():
    """Demonstrate multi-star analysis pipeline"""
    
    print("=" * 60)
    print("MULTI-STAR SCENE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize pipelines
    print("\n1. Initializing pipeline...")
    pipeline = StarTrackerPipeline()
    multi_pipeline = MultiStarPipeline(pipeline)
    
    # Load PSF data
    print("2. Loading PSF data...")
    psf_data_dict = pipeline.load_psf_data("data/PSF_sims/Gen_1", "*0_deg*")
    psf_data = list(psf_data_dict.values())[0]
    
    # Load star catalog
    print("3. Loading star catalog...")
    catalog = from_csv("data/catalogs/baseline_5_stars_spread.csv", magnitude=8.0, fov=30.0)
    print(f"   Loaded catalog with {len(catalog)} stars")
    
    # Test different attitude scenarios
    test_cases = [
        {
            'name': 'Identity (Trivial)',
            'attitude_euler': None,
            'description': 'Camera aligned with catalog frame'
        },
        {
            'name': 'Small Rotation',
            'attitude_euler': np.radians([2.0, -1.5, 1.0]),
            'description': 'Small attitude perturbation'
        }
    ]
    
    for case in test_cases:
        print(f"\n" + "="*40)
        print(f"TESTING: {case['name']}")
        print(f"Description: {case['description']}")
        if case['attitude_euler'] is not None:
            euler_deg = np.degrees(case['attitude_euler'])
            print(f"Euler angles: ({euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°)")
        print("="*40)
        
        try:
            # Run multi-star analysis
            results = multi_pipeline.run_multi_star_analysis(
                catalog,
                psf_data,
                true_attitude_euler=case['attitude_euler'],
                perform_validation=True
            )
            
            # Display results
            summary = results['analysis_summary']
            print(f"Stars in catalog: {summary['stars_in_catalog']}")
            print(f"Stars on detector: {summary['stars_on_detector']}")
            print(f"Stars detected: {summary['stars_detected']}")
            print(f"Stars matched: {summary['stars_matched']}")
            print(f"Detection rate: {summary['detection_rate']:.1%}")
            print(f"Matching rate: {summary['matching_rate']:.1%}")
            
            if results['quest_results']:
                quest = results['quest_results']
                print(f"QUEST trials: {quest.num_trials_used}")
                print(f"Attitude uncertainty: {quest.angular_uncertainty_arcsec:.1f} arcsec")
            
        except Exception as e:
            print(f"Error in analysis: {e}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThis demonstration showed:")
    print("• Multi-star scene generation")
    print("• Arbitrary attitude transformations")
    print("• Multiple star detection and centroiding")
    print("• BAST triangle matching")
    print("• Monte Carlo QUEST attitude determination")
    print("• Complete end-to-end pipeline validation")

if __name__ == "__main__":
    multi_star_demo()