#!/usr/bin/env python3
"""
Star Scene Visualization Script

Creates visualizations showing:
1. Celestial sphere view of generated stars
2. CMV4000 image plane projection of star positions

This demonstrates the complete star tracker pipeline from catalog generation
to detector imagery using existing pipeline functions.

Usage:
    PYTHONPATH=. python star_scene_visualization.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.multi_star.multi_star_pipeline import MultiStarPipeline
from src.BAST.catalog import from_csv

def plot_celestial_sphere(stars_ra_dec, ax):
    """Plot stars on the celestial sphere in 3D with connecting lines to center"""
    # Convert RA/Dec to 3D Cartesian coordinates
    ra_rad = np.radians(stars_ra_dec[:, 0])
    dec_rad = np.radians(stars_ra_dec[:, 1])

    # Celestial sphere coordinates (right-handed)
    x_stars = np.cos(dec_rad) * np.cos(ra_rad)
    y_stars = np.cos(dec_rad) * np.sin(ra_rad)
    z_stars = np.sin(dec_rad)

    # Calculate center of star field for zoomed view
    center_x = np.mean(x_stars)
    center_y = np.mean(y_stars)
    center_z = np.mean(z_stars)

    # Create a small sphere around the star field center
    # Calculate the maximum distance from center to any star
    distances = np.sqrt((x_stars - center_x)**2 + (y_stars - center_y)**2 + (z_stars - center_z)**2)
    max_distance = np.max(distances)
    sphere_radius = max_distance * 1.5  # Make sphere slightly larger than star field

    # Create sphere surface around the star field center
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = center_x + sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = center_y + sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = center_z + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot sphere surface with transparency
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')

    # Plot stars
    ax.scatter(x_stars, y_stars, z_stars, c='red', s=100, alpha=0.9, edgecolors='black', zorder=5)

    # Plot center point
    ax.scatter(center_x, center_y, center_z, c='blue', s=200, marker='*', alpha=1.0, edgecolors='navy', zorder=10)
    ax.text(center_x, center_y, center_z + 0.05,
            'Field Center\n(Boresight)', fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

    # Add connecting lines from center to each star
    for i in range(len(x_stars)):
        ax.plot([center_x, x_stars[i]], [center_y, y_stars[i]], [center_z, z_stars[i]],
                'gray', linewidth=1.5, alpha=0.7, linestyle='--')

    # Add simple star labels
    for i in range(len(x_stars)):
        # Position labels slightly outward from the sphere
        label_offset = 1.1
        label_x = center_x + (x_stars[i] - center_x) * label_offset
        label_y = center_y + (y_stars[i] - center_y) * label_offset
        label_z = center_z + (z_stars[i] - center_z) * label_offset

        ax.text(label_x, label_y, label_z,
                f'Star {i+1}', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Set axis limits to focus on the star field
    axis_margin = max_distance * 2.0
    ax.set_xlim(center_x - axis_margin, center_x + axis_margin)
    ax.set_ylim(center_y - axis_margin, center_y + axis_margin)
    ax.set_zlim(center_z - axis_margin, center_z + axis_margin)

    # Set labels and title
    ax.set_xlabel('X (Celestial)', fontsize=10)
    ax.set_ylabel('Y (Celestial)', fontsize=10)
    ax.set_zlabel('Z (Celestial)', fontsize=10)
    ax.set_title('Celestial Sphere View\n(Zoomed Field)', fontsize=12, fontweight='bold')

    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])

    # Set view angle for better visibility
    ax.view_init(elev=25, azim=45)

def plot_image_plane(detector_positions, stars_ra_dec, ax):
    """Plot star positions on the detector image plane"""
    # Detector dimensions (CMV4000: 2048x2048 pixels, 5.5µm pitch)
    detector_size = 2048
    pixel_pitch = 5.5  # µm

    # Convert detector positions to physical coordinates (mm)
    x_pixels = np.array([pos[0] for pos in detector_positions])
    y_pixels = np.array([pos[1] for pos in detector_positions])

    # Convert to mm for physical interpretation
    x_mm = (x_pixels - detector_size/2) * pixel_pitch / 1000
    y_mm = (y_pixels - detector_size/2) * pixel_pitch / 1000

    # Plot stars on detector
    ax.scatter(x_mm, y_mm, c='blue', s=100, alpha=0.8, edgecolors='black', zorder=3)

    # Add simple star labels
    for i, (x, y) in enumerate(zip(x_mm, y_mm)):
        ax.annotate(f'Star {i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add detector boundary
    detector_width_mm = detector_size * pixel_pitch / 1000
    detector_height_mm = detector_size * pixel_pitch / 1000

    # Draw detector outline
    rect = plt.Rectangle((-detector_width_mm/2, -detector_height_mm/2),
                        detector_width_mm, detector_height_mm,
                        fill=False, color='black', linewidth=2, linestyle='--')
    ax.add_patch(rect)

    # Add field of view circles (approximate)
    fov_radii = [detector_width_mm/2 * 0.7, detector_width_mm/2 * 0.9]
    for radius in fov_radii:
        fov_circle = Circle((0, 0), radius, fill=False, color='gray',
                           linestyle=':', alpha=0.5, linewidth=1)
        ax.add_patch(fov_circle)

    # Set labels and title
    ax.set_xlabel('X Position (mm)', fontsize=10)
    ax.set_ylabel('Y Position (mm)', fontsize=10)
    ax.set_title('Image Plane Projection\n(CMV4000 Detector)', fontsize=12, fontweight='bold')

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Add grid
    ax.grid(True, alpha=0.3)

def main():
    """Main function to generate star scene visualizations"""
    print("=" * 60)
    print("STAR TRACKER SCENE VISUALIZATION")
    print("=" * 60)

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = StarTrackerPipeline()
    multi_pipeline = MultiStarPipeline(pipeline)

    # Load star catalog
    print("2. Loading star catalog...")
    catalog_path = "data/catalogs/sparse_4_stars_wide_separation.csv"
    catalog = from_csv(catalog_path, magnitude=8.0, fov=30.0)
    print(f"   Loaded catalog with {len(catalog)} stars")
    print("   Using sparse_4_stars_wide_separation.csv for cleaner visualization")

    # Load PSF data (needed for complete pipeline, but not for basic visualization)
    print("3. Loading PSF data...")
    psf_dict = pipeline.load_psf_data("data/PSF_sims/Gen_1", "*0_deg*")
    psf_data = list(psf_dict.values())[0]

    # Generate scene with attitude transformation (optional random attitude)
    print("4. Generating star scene...")
    np.random.seed(42)  # For reproducible results

    # Use small random attitude for demonstration
    attitude_euler = np.radians([5.0, -3.0, 2.0])  # Small attitude perturbation

    results = multi_pipeline.run_multi_star_analysis(
        catalog,
        psf_data,
        true_attitude_euler=attitude_euler,
        perform_validation=True  # Enable validation for complete pipeline
    )

    scene_data = results['scene_data']
    print(f"   Generated scene with {len(scene_data['stars'])} stars on detector")

    # Extract star data for plotting
    stars_data = scene_data['stars']
    ra_dec = np.array([[star['ra'], star['dec']] for star in stars_data])
    detector_positions = np.array([star['detector_position'] for star in stars_data])

    print("5. Creating combined visualization...")

    # Create combined figure with both plots side by side
    fig = plt.figure(figsize=(16, 6))

    # Left subplot: Celestial sphere
    ax1 = fig.add_subplot(121, projection='3d')
    print("   - Celestial sphere view...")
    plot_celestial_sphere(ra_dec, ax1)

    # Right subplot: Image plane
    ax2 = fig.add_subplot(122)
    print("   - Image plane projection...")
    plot_image_plane(detector_positions, ra_dec, ax2)

    # Set overall title
    fig.suptitle('Star Tracker Scene Visualization: Celestial Sphere to Image Plane', fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout()

    # Save combined figure
    print("6. Saving combined figure...")
    fig.savefig('star_tracker_visualization.png', dpi=300, bbox_inches='tight')
    fig.savefig('star_tracker_visualization.pdf', bbox_inches='tight')

    print("   Combined figure saved: star_tracker_visualization.png/pdf")

    # Display attitude information
    attitude_info = scene_data['attitude']
    print("\n7. Scene Information:")
    print(f"   Camera Center: RA={np.degrees(attitude_info['ra']):.1f}°, Dec={np.degrees(attitude_info['dec']):.1f}°")
    print(f"   Attitude Euler: {np.degrees(attitude_euler)} degrees")
    print(f"   Stars Generated: {len(stars_data)}")
    print(f"   Transformation Method: {attitude_info['transformation_method']}")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)

    print("\nGenerated files:")
    print("- celestial_sphere_view.png/pdf: 3D celestial sphere with star positions")
    print("- image_plane_projection.png/pdf: 2D detector view with projected positions")

    # Show plots if running interactively
    try:
        plt.show()
    except:
        print("Note: Use plt.show() in interactive environment to display plots")

if __name__ == "__main__":
    main()