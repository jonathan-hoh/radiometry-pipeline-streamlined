#!/usr/bin/env python3
"""
Single Star Analysis Demonstration

A self-contained example showing the complete single-star analysis pipeline
from PSF loading through bearing vector calculation.

Usage:
    PYTHONPATH=. python examples/single_star_demo.py
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, project_root)

from src.core.star_tracker_pipeline import StarTrackerPipeline

def single_star_demo():
    """Demonstrate single star analysis pipeline"""
    
    print("=" * 60)
    print("SINGLE STAR ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n1. Initializing StarTrackerPipeline...")
    pipeline = StarTrackerPipeline()
    
    # Load PSF data
    print("2. Loading PSF data...")
    psf_data_dict = pipeline.load_psf_data("data/PSF_sims/Gen_1", "*0_deg*")
    psf_data = list(psf_data_dict.values())[0]
    
    print(f"   Loaded PSF: {psf_data['metadata']['filename']}")
    print(f"   Grid size: {psf_data['intensity_data'].shape}")
    print(f"   Physical size: {psf_data['metadata']['physical_size_um']} µm")
    
    # Run single star analysis
    print("\n3. Running single star analysis...")
    
    # Test with different magnitudes
    magnitudes = [3.0, 4.0, 5.0]
    
    for magnitude in magnitudes:
        print(f"\n   Testing magnitude {magnitude}:")
        
        try:
            results = pipeline.run_single_star_analysis(
                psf_data, 
                magnitude=magnitude,
                num_trials=10,
                visualize=False
            )
            
            print(f"     Centroiding accuracy: {results.get('mean_centroid_error_px', 'N/A'):.3f} px")
            print(f"     Bearing vector error: {results.get('mean_bearing_vector_error_arcsec', 'N/A'):.1f} arcsec")
            print(f"     Detection success rate: {results.get('detection_success_rate', 'N/A'):.1%}")
            
        except Exception as e:
            print(f"     Error: {e}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThis demonstration showed:")
    print("• PSF data loading and processing")
    print("• Radiometric simulation with different star magnitudes")
    print("• Star detection and centroiding accuracy")
    print("• Bearing vector calculation")
    print("• Performance quantification")

if __name__ == "__main__":
    single_star_demo()