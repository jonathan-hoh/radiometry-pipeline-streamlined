#!/usr/bin/env python3
"""
Clean Moment-Based Centroiding Visualization

Creates a simple before-and-after diagram showing star detection on the left
and weighted moment calculation on the right.

Usage: python tools/clean_centroiding_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def generate_clean_star_scene(size=400, star_intensity=1000, noise_level=5.0):
    """Generate a clean star scene for demonstration"""
    # Simple uniform background with slight gradient
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    background = 25 + 5 * x
    
    image = background.copy()
    
    # Add main star
    star_x, star_y = 200, 180  # Slightly off-center
    sigma = 3.0
    
    # Generate Gaussian PSF
    yy, xx = np.ogrid[:size, :size]
    star_psf = star_intensity * np.exp(-((xx - star_x)**2 + (yy - star_y)**2) / (2 * sigma**2))
    image += star_psf
    
    # Add a couple of background stars
    bg_stars = [(120, 300, 300), (320, 80, 250), (50, 100, 200)]
    for sx, sy, intensity in bg_stars:
        bg_psf = intensity * np.exp(-((xx - sx)**2 + (yy - sy)**2) / (2 * (sigma*0.8)**2))
        image += bg_psf
    
    # Add noise
    image = np.random.poisson(np.maximum(image, 0))
    image = image + np.random.normal(0, noise_level, image.shape)
    
    return image.astype(np.float32), (star_x, star_y)

def simple_star_detection(image, threshold_factor=4.0):
    """Simple but effective star detection"""
    # Global statistics for simple thresholding
    mean_val = np.mean(image)
    std_val = np.std(image)
    threshold = mean_val + threshold_factor * std_val
    
    # Binary mask
    binary_mask = (image > threshold).astype(np.uint8)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Find brightest region
    brightest_region = None
    max_intensity = 0
    
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if 8 <= area <= 200:  # Size filter
            region_mask = (labels == label_id)
            total_intensity = np.sum(image[region_mask])
            
            if total_intensity > max_intensity:
                max_intensity = total_intensity
                brightest_region = {
                    'label': label_id,
                    'mask': region_mask,
                    'total_intensity': total_intensity
                }
    
    return binary_mask, labels, brightest_region

def calculate_centroid_with_details(image, region_mask):
    """Calculate centroid and return pixel details for visualization"""
    # Get region pixels
    y_coords, x_coords = np.where(region_mask)
    intensities = image[region_mask]
    
    # Calculate weighted centroid
    total_intensity = np.sum(intensities)
    if total_intensity > 0:
        centroid_x = np.sum(x_coords * intensities) / total_intensity
        centroid_y = np.sum(y_coords * intensities) / total_intensity
    else:
        centroid_x, centroid_y = 0, 0
    
    return centroid_x, centroid_y, x_coords, y_coords, intensities

def create_clean_visualization():
    """Create clean before-and-after centroiding visualization"""
    # Generate test scene
    np.random.seed(42)
    image, true_star_pos = generate_clean_star_scene()
    
    # Detect stars
    binary_mask, labels, selected_region = simple_star_detection(image)
    
    if selected_region is None:
        print("No suitable star region found!")
        return None
    
    # Calculate centroid
    centroid_x, centroid_y, star_x_coords, star_y_coords, star_intensities = calculate_centroid_with_details(
        image, selected_region['mask']
    )
    
    # Create figure with clean layout
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 7))
    
    # LEFT PANEL: Original image with detections
    im_left = ax_left.imshow(image, cmap='viridis', origin='lower')
    ax_left.set_title('Star Detection Results', fontsize=16, fontweight='bold', pad=20)
    ax_left.set_xlabel('X Pixel', fontsize=12)
    ax_left.set_ylabel('Y Pixel', fontsize=12)
    
    # Show all detected regions as white contours
    all_regions = (labels > 0)
    ax_left.contour(all_regions, colors='white', alpha=0.6, linewidths=1.5)
    
    # Highlight selected region with red contour
    selected_contour = np.zeros_like(image)
    selected_contour[selected_region['mask']] = 1
    ax_left.contour(selected_contour, colors='red', linewidths=2.5)
    
    # Mark centroid with red cross and yellow circle
    ax_left.plot(centroid_x, centroid_y, 'r+', markersize=15, markeredgewidth=3)
    ax_left.plot(centroid_x, centroid_y, 'yo', markersize=10, fillstyle='none', markeredgewidth=2)
    
    # Add text annotation
    ax_left.text(centroid_x + 15, centroid_y + 15, 
                f'Detected\nCentroid\n({centroid_x:.1f}, {centroid_y:.1f})', 
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    plt.colorbar(im_left, ax=ax_left, label='Intensity (DN)', shrink=0.8)
    
    # RIGHT PANEL: Zoomed moment calculation
    # Extract region around star
    margin = 25
    x_min = max(0, int(centroid_x) - margin)
    x_max = min(image.shape[1], int(centroid_x) + margin)
    y_min = max(0, int(centroid_y) - margin)
    y_max = min(image.shape[0], int(centroid_y) + margin)
    
    zoomed_image = image[y_min:y_max, x_min:x_max]
    
    # Show zoomed region
    ax_right.imshow(zoomed_image, cmap='gray', origin='lower', alpha=0.4)
    ax_right.set_title('Weighted Moment Calculation', fontsize=16, fontweight='bold', pad=20)
    ax_right.set_xlabel('X Pixel', fontsize=12)
    ax_right.set_ylabel('Y Pixel', fontsize=12)
    
    # Adjust coordinates to zoomed region
    zoomed_centroid_x = centroid_x - x_min
    zoomed_centroid_y = centroid_y - y_min
    
    # Show star pixels with intensity-based sizing and coloring
    star_pixels_in_zoom = []
    for i in range(len(star_x_coords)):
        px, py = star_x_coords[i], star_y_coords[i]
        if x_min <= px < x_max and y_min <= py < y_max:
            rel_x, rel_y = px - x_min, py - y_min
            intensity = star_intensities[i]
            star_pixels_in_zoom.append((rel_x, rel_y, intensity))
    
    if star_pixels_in_zoom:
        # Normalize intensities for visualization
        max_intensity = max([p[2] for p in star_pixels_in_zoom])
        
        # Draw pixels as circles
        for rel_x, rel_y, intensity in star_pixels_in_zoom:
            intensity_norm = intensity / max_intensity
            circle_size = 50 + 200 * intensity_norm  # Size based on intensity
            
            # Color from blue (low) to red (high)
            color = plt.cm.plasma(intensity_norm)
            ax_right.scatter(rel_x, rel_y, s=circle_size, c=[color], 
                           edgecolors='black', linewidth=0.8, alpha=0.8)
            
            # Draw arrow pointing to centroid (for brighter pixels only)
            if intensity_norm > 0.3:  # Only show arrows for significant pixels
                dx = zoomed_centroid_x - rel_x
                dy = zoomed_centroid_y - rel_y
                arrow_length = np.sqrt(dx**2 + dy**2)
                if arrow_length > 1:  # Avoid tiny arrows
                    ax_right.arrow(rel_x, rel_y, dx*0.7, dy*0.7, 
                                 head_width=0.8, head_length=1.0, 
                                 fc='yellow', ec='orange', alpha=0.7, linewidth=1)
    
    # Mark final centroid
    ax_right.plot(zoomed_centroid_x, zoomed_centroid_y, '*', 
                 markersize=20, color='red', markeredgecolor='white', markeredgewidth=2)
    ax_right.text(zoomed_centroid_x + 2, zoomed_centroid_y + 2, 'Centroid', 
                 fontsize=12, fontweight='bold', color='red',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], s=150, c='blue', alpha=0.8, edgecolors='black', label='Low Intensity Pixels'),
        plt.scatter([], [], s=250, c='red', alpha=0.8, edgecolors='black', label='High Intensity Pixels'),
        plt.scatter([], [], s=100, marker='*', color='red', edgecolors='white', label='Final Centroid')
    ]
    ax_right.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # Add formula annotation
    formula_text = ('Centroid Calculation:\n' +
                   r'$x_c = \frac{\sum x_i \cdot I_i}{\sum I_i}$' + '\n' +
                   r'$y_c = \frac{\sum y_i \cdot I_i}{\sum I_i}$')
    ax_right.text(0.02, 0.02, formula_text, transform=ax_right.transAxes, 
                 fontsize=11, verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.9', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Add main title
    fig.suptitle('Before and After: Star Detection and Moment-Based Centroiding', 
                fontsize=18, fontweight='bold', y=0.95)
    
    return fig, (centroid_x, centroid_y), true_star_pos

def print_results(detected_centroid, true_position):
    """Print clean analysis results"""
    print("\n" + "="*50)
    print("CENTROIDING RESULTS")
    print("="*50)
    
    cx, cy = detected_centroid
    tx, ty = true_position
    
    error_x = cx - tx
    error_y = cy - ty
    error_magnitude = np.sqrt(error_x**2 + error_y**2)
    
    print(f"Detected Centroid: ({cx:.2f}, {cy:.2f}) pixels")
    print(f"True Position:     ({tx:.2f}, {ty:.2f}) pixels")
    print(f"Error:             {error_magnitude:.3f} pixels ({error_magnitude*5.5:.2f} μm)")
    print(f"Sub-pixel accuracy: {error_magnitude < 1.0}")

def main():
    """Generate clean centroiding visualization"""
    print("Generating clean moment-based centroiding visualization...")
    
    result = create_clean_visualization()
    
    if result is not None:
        fig, centroid, true_pos = result
        
        # Print results
        print_results(centroid, true_pos)
        
        # Save figure
        output_path = "clean_centroiding_visualization.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nFigure saved as: {output_path}")
        
        plt.show()
    else:
        print("Failed to generate visualization")

if __name__ == "__main__":
    main()