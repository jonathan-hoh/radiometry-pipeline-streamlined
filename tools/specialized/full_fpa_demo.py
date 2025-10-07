#!/usr/bin/env python3
"""
full_fpa_demo.py - Demonstrate FPA projection with full CMV4000 detector visualization
Shows both the 11×11 "zoomed" PSF and the 2048×2048 full detector view
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.core.psf_plot import parse_psf_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def demonstrate_full_fpa_projection(psf_file, magnitude=3.0, num_trials=5, output_dir="full_fpa_demo"):
    """
    Demonstrate FPA projection with both zoomed and full detector views
    
    Args:
        psf_file: Path to PSF file
        magnitude: Star magnitude to simulate
        num_trials: Number of Monte Carlo trials
        output_dir: Output directory for results
    """
    # Create pipeline
    pipeline = StarTrackerPipeline(debug=False)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load PSF data
    logger.info(f"Loading PSF file: {psf_file}")
    try:
        metadata, intensity_data = parse_psf_file(psf_file)
        psf_data = {
            'metadata': metadata,
            'intensity_data': intensity_data,
            'file_path': psf_file
        }
    except Exception as e:
        logger.error(f"Error loading PSF file: {e}")
        return None
    
    field_angle = metadata.get('field_angle', 0.0)
    logger.info(f"Field angle: {field_angle}°")
    
    # === STEP 1: Original PSF Analysis ===
    logger.info("="*60)
    logger.info("STEP 1: Original PSF Analysis")
    logger.info("="*60)
    
    original_results = pipeline.run_monte_carlo_simulation(
        psf_data, magnitude=magnitude, num_trials=num_trials
    )
    
    # === STEP 2: FPA Projection (11×11 only) ===
    logger.info("="*60)
    logger.info("STEP 2: FPA Projection (11×11 zoomed view)")
    logger.info("="*60)
    
    fpa_results_zoomed = pipeline.run_monte_carlo_simulation_fpa_projected(
        psf_data, magnitude=magnitude, num_trials=num_trials, 
        create_full_fpa=False
    )
    
    # === STEP 3: FPA Projection with Full Detector ===
    logger.info("="*60)
    logger.info("STEP 3: FPA Projection with Full CMV4000 Detector")
    logger.info("="*60)
    
    fpa_results_full = pipeline.run_monte_carlo_simulation_fpa_projected(
        psf_data, magnitude=magnitude, num_trials=num_trials, 
        create_full_fpa=True, fpa_size=(2048, 2048)
    )
    
    # === STEP 4: Create Comprehensive Visualization ===
    logger.info("="*60)
    logger.info("STEP 4: Creating Visualization")
    logger.info("="*60)
    
    create_full_fpa_visualization(
        original_results, fpa_results_zoomed, fpa_results_full, 
        output_dir, field_angle, magnitude
    )
    
    # === STEP 5: Summary ===
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    print_performance_summary(original_results, fpa_results_zoomed, fpa_results_full)
    
    logger.info(f"Results saved to: {output_dir}")
    return {
        'original': original_results,
        'fpa_zoomed': fpa_results_zoomed,
        'fpa_full': fpa_results_full
    }

def create_full_fpa_visualization(original_results, fpa_zoomed, fpa_full, output_dir, field_angle, magnitude):
    """Create comprehensive visualization showing all analysis modes"""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # === Top Row: PSF Comparisons ===
    
    # Original PSF
    ax1 = plt.subplot(3, 4, 1)
    original_psf = original_results['projection_results']['normalized_psf']
    im1 = ax1.imshow(original_psf, cmap='hot', origin='lower')
    ax1.set_title('Original PSF\n(128×128, 0.5µm/px)', fontsize=12)
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # FPA Projected PSF (11×11)
    ax2 = plt.subplot(3, 4, 2)
    fpa_psf = fpa_zoomed['fpa_psf_data']['intensity_data']
    im2 = ax2.imshow(fpa_psf, cmap='hot', origin='lower')
    ax2.set_title('FPA Projected PSF\n(11×11, 5.5µm/px)', fontsize=12)
    ax2.set_xlabel('FPA Pixels')
    ax2.set_ylabel('FPA Pixels')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Full FPA with PSF (zoomed out view)
    ax3 = plt.subplot(3, 4, 3)
    full_fpa = fpa_full['fpa_psf_data']['full_fpa_intensity']
    if full_fpa is not None:
        # Show full detector with PSF
        im3 = ax3.imshow(full_fpa, cmap='hot', origin='lower', vmin=0, vmax=np.max(full_fpa)*0.1)
        ax3.set_title('Full CMV4000 Detector\n(2048×2048, 5.5µm/px)', fontsize=12)
        ax3.set_xlabel('FPA Pixels')
        ax3.set_ylabel('FPA Pixels')
        
        # Add rectangle showing PSF location
        fpa_position = fpa_full['fpa_psf_data']['scaling_info']['fpa_position']
        if fpa_position is not None:
            row_start, col_start = fpa_position
            fpa_height, fpa_width = fpa_full['fpa_psf_data']['scaling_info']['fpa_shape']
            rect = patches.Rectangle((col_start, row_start), fpa_width, fpa_height, 
                                   linewidth=2, edgecolor='cyan', facecolor='none')
            ax3.add_patch(rect)
            ax3.text(col_start + fpa_width/2, row_start + fpa_height + 50, 'PSF Location', 
                    ha='center', va='bottom', color='cyan', fontweight='bold')
        
        plt.colorbar(im3, ax=ax3, shrink=0.8)
    else:
        ax3.text(0.5, 0.5, 'Full FPA\nNot Created', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Full CMV4000 Detector\n(Not Available)', fontsize=12)
    
    # Full FPA zoomed to PSF region
    ax4 = plt.subplot(3, 4, 4)
    if full_fpa is not None and fpa_position is not None:
        row_start, col_start = fpa_position
        fpa_height, fpa_width = fpa_full['fpa_psf_data']['scaling_info']['fpa_shape']
        
        # Extract region around PSF with some padding
        padding = 20
        r_start = max(0, row_start - padding)
        r_end = min(full_fpa.shape[0], row_start + fpa_height + padding)
        c_start = max(0, col_start - padding)
        c_end = min(full_fpa.shape[1], col_start + fpa_width + padding)
        
        zoomed_region = full_fpa[r_start:r_end, c_start:c_end]
        im4 = ax4.imshow(zoomed_region, cmap='hot', origin='lower')
        ax4.set_title('Full FPA (Zoomed to PSF)\nwith Context', fontsize=12)
        ax4.set_xlabel('FPA Pixels')
        ax4.set_ylabel('FPA Pixels')
        
        # Add rectangle showing exact PSF boundaries
        rect_x = col_start - c_start
        rect_y = row_start - r_start
        rect = patches.Rectangle((rect_x, rect_y), fpa_width, fpa_height, 
                               linewidth=2, edgecolor='cyan', facecolor='none')
        ax4.add_patch(rect)
        
        plt.colorbar(im4, ax=ax4, shrink=0.8)
    else:
        ax4.text(0.5, 0.5, 'Zoomed View\nNot Available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Full FPA (Zoomed)\n(Not Available)', fontsize=12)
    
    # === Middle Row: Simulated Star Images ===
    
    # Original PSF simulation
    ax5 = plt.subplot(3, 4, 5)
    if original_results['projection_results']['simulations']:
        orig_sim = original_results['projection_results']['simulations'][0]
        im5 = ax5.imshow(orig_sim, cmap='viridis', origin='lower')
        ax5.set_title('Original PSF\nSimulated Star', fontsize=12)
        plt.colorbar(im5, ax=ax5, shrink=0.8)
        
        # Add detected centroid if available
        if original_results['centroid_results']['centroids']:
            cx, cy = original_results['centroid_results']['centroids'][0]
            ax5.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2, label='Detected')
        if original_results['centroid_results']['true_center']:
            tx, ty = original_results['centroid_results']['true_center']
            ax5.plot(tx, ty, 'g+', markersize=10, markeredgewidth=2, label='True')
        ax5.legend()
    
    # FPA simulation (11×11)
    ax6 = plt.subplot(3, 4, 6)
    if fpa_zoomed['projection_results']['simulations']:
        fpa_sim = fpa_zoomed['projection_results']['simulations'][0]
        im6 = ax6.imshow(fpa_sim, cmap='viridis', origin='lower')
        ax6.set_title('FPA Projected\nSimulated Star', fontsize=12)
        plt.colorbar(im6, ax=ax6, shrink=0.8)
        
        # Add detected centroid if available
        if fpa_zoomed['centroid_results']['centroids']:
            cx, cy = fpa_zoomed['centroid_results']['centroids'][0]
            ax6.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2, label='Detected')
        if fpa_zoomed['centroid_results']['true_center']:
            tx, ty = fpa_zoomed['centroid_results']['true_center']
            ax6.plot(tx, ty, 'g+', markersize=10, markeredgewidth=2, label='True')
        ax6.legend()
    
    # Full FPA simulation (if available) - show full detector
    ax7 = plt.subplot(3, 4, 7)
    if full_fpa is not None and fpa_full['projection_results']['simulations']:
        # For visualization, we'd need to place the simulation on the full detector
        # For now, just show the 11×11 simulation
        fpa_sim_full = fpa_full['projection_results']['simulations'][0]
        im7 = ax7.imshow(fpa_sim_full, cmap='viridis', origin='lower')
        ax7.set_title('FPA Simulation\n(11×11 region)', fontsize=12)
        plt.colorbar(im7, ax=ax7, shrink=0.8)
        
        if fpa_full['centroid_results']['centroids']:
            cx, cy = fpa_full['centroid_results']['centroids'][0]
            ax7.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2, label='Detected')
        if fpa_full['centroid_results']['true_center']:
            tx, ty = fpa_full['centroid_results']['true_center']
            ax7.plot(tx, ty, 'g+', markersize=10, markeredgewidth=2, label='True')
        ax7.legend()
    
    # Physical scale comparison
    ax8 = plt.subplot(3, 4, 8)
    create_scale_comparison_plot(ax8, original_results, fpa_zoomed)
    
    # === Bottom Row: Performance Metrics ===
    
    # Centroiding performance
    ax9 = plt.subplot(3, 4, 9)
    create_performance_bar_chart(ax9, original_results, fpa_zoomed, fpa_full, 'centroid')
    
    # Bearing vector performance  
    ax10 = plt.subplot(3, 4, 10)
    create_performance_bar_chart(ax10, original_results, fpa_zoomed, fpa_full, 'vector')
    
    # Success rates
    ax11 = plt.subplot(3, 4, 11)
    create_performance_bar_chart(ax11, original_results, fpa_zoomed, fpa_full, 'success')
    
    # Summary text
    ax12 = plt.subplot(3, 4, 12)
    create_summary_text(ax12, original_results, fpa_zoomed, fpa_full, field_angle, magnitude)
    
    plt.suptitle(f'Full FPA Projection Demonstration\nField Angle: {field_angle}°, Magnitude: {magnitude}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"full_fpa_demo_{field_angle:.1f}deg_mag_{magnitude:.1f}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved full FPA demonstration plot: {plot_path}")
    plt.show()

def create_scale_comparison_plot(ax, original_results, fpa_results):
    """Create a plot showing physical scale comparison"""
    # Physical dimensions
    orig_pixel_size = 0.5  # µm
    fpa_pixel_size = 5.5   # µm
    orig_size = 128 * orig_pixel_size  # 64 µm
    fpa_size = 11 * fpa_pixel_size     # 60.5 µm
    
    # Create scale bars
    ax.barh([0], [orig_size], height=0.3, color='blue', alpha=0.7, label=f'Original PSF\n{orig_size:.1f} µm')
    ax.barh([1], [fpa_size], height=0.3, color='red', alpha=0.7, label=f'FPA Projected\n{fpa_size:.1f} µm')
    
    ax.set_xlim(0, 70)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('Physical Size (µm)')
    ax.set_ylabel('')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Original', 'FPA'])
    ax.set_title('Physical Scale\nComparison', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_performance_bar_chart(ax, original, fpa_zoomed, fpa_full, metric_type):
    """Create bar chart for performance comparison"""
    
    if metric_type == 'centroid':
        orig_val = original['mean_centroid_error_px'] * 0.5  # Convert to µm
        fpa_z_val = fpa_zoomed.get('mean_centroid_error_um', float('nan'))
        fpa_f_val = fpa_full.get('mean_centroid_error_um', float('nan'))
        title = 'Centroiding Error'
        ylabel = 'Error (µm)'
        colors = ['blue', 'orange', 'green']
    elif metric_type == 'vector':
        orig_val = original['mean_vector_error_arcsec']
        fpa_z_val = fpa_zoomed['mean_vector_error_arcsec']
        fpa_f_val = fpa_full['mean_vector_error_arcsec']
        title = 'Bearing Vector Error'
        ylabel = 'Error (arcsec)'
        colors = ['blue', 'orange', 'green']
    else:  # success
        orig_val = original['success_rate']
        fpa_z_val = fpa_zoomed['success_rate']
        fpa_f_val = fpa_full['success_rate']
        title = 'Success Rate'
        ylabel = 'Success Rate'
        colors = ['blue', 'orange', 'green']
    
    values = [orig_val, fpa_z_val, fpa_f_val]
    labels = ['Original\nPSF', 'FPA\nZoomed', 'FPA\nFull']
    
    # Filter out NaN values
    valid_data = [(i, v, l, c) for i, (v, l, c) in enumerate(zip(values, labels, colors)) if not np.isnan(v)]
    
    if valid_data:
        indices, vals, labs, cols = zip(*valid_data)
        bars = ax.bar(range(len(vals)), vals, color=cols, alpha=0.7)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labs)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

def create_summary_text(ax, original, fpa_zoomed, fpa_full, field_angle, magnitude):
    """Create summary text box"""
    ax.axis('off')
    
    # Calculate improvements
    orig_vector = original['mean_vector_error_arcsec']
    fpa_vector = fpa_zoomed['mean_vector_error_arcsec']
    
    if not np.isnan(orig_vector) and not np.isnan(fpa_vector) and orig_vector > 0:
        vector_improvement = ((orig_vector - fpa_vector) / orig_vector) * 100
    else:
        vector_improvement = 0
    
    # Create summary text
    summary_text = f"""ANALYSIS SUMMARY
    
Field Angle: {field_angle:.1f}°
Star Magnitude: {magnitude:.1f}
    
PERFORMANCE COMPARISON:
Original PSF → FPA Projection

Bearing Vector Error:
{orig_vector:.1f} → {fpa_vector:.1f} arcsec
Improvement: {vector_improvement:+.1f}%

Success Rates:
Original: {original['success_rate']:.1%}
FPA: {fpa_zoomed['success_rate']:.1%}

FPA Benefits:
• Realistic detector modeling
• Natural noise filtering  
• Proper scale representation
• Hardware-accurate predictions"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def print_performance_summary(original, fpa_zoomed, fpa_full):
    """Print performance summary to console"""
    
    print(f"Performance Comparison:")
    print(f"{'Metric':<25} {'Original PSF':<15} {'FPA Zoomed':<15} {'FPA Full':<15}")
    print("-" * 70)
    
    # Centroiding (in µm)
    orig_centroid_um = original['mean_centroid_error_px'] * 0.5
    fpa_z_centroid_um = fpa_zoomed.get('mean_centroid_error_um', float('nan'))
    fpa_f_centroid_um = fpa_full.get('mean_centroid_error_um', float('nan'))
    
    print(f"{'Centroiding (µm)':<25} {orig_centroid_um:<15.3f} {fpa_z_centroid_um:<15.3f} {fpa_f_centroid_um:<15.3f}")
    
    # Bearing vectors
    orig_vector = original['mean_vector_error_arcsec']
    fpa_z_vector = fpa_zoomed['mean_vector_error_arcsec']
    fpa_f_vector = fpa_full['mean_vector_error_arcsec']
    
    print(f"{'Vectors (arcsec)':<25} {orig_vector:<15.2f} {fpa_z_vector:<15.2f} {fpa_f_vector:<15.2f}")
    
    # Success rates
    orig_success = original['success_rate']
    fpa_z_success = fpa_zoomed['success_rate']
    fpa_f_success = fpa_full['success_rate']
    
    print(f"{'Success Rate':<25} {orig_success:<15.2%} {fpa_z_success:<15.2%} {fpa_f_success:<15.2%}")
    
    # Calculate improvement
    if not np.isnan(orig_vector) and not np.isnan(fpa_z_vector) and orig_vector > 0:
        improvement = ((orig_vector - fpa_z_vector) / orig_vector) * 100
        print(f"\nKey Insight: FPA projection provides {improvement:.1f}% better bearing vector accuracy!")
    
    # Physical scale info
    if fpa_full['fpa_psf_data']['scaling_info']['fpa_position'] is not None:
        pos = fpa_full['fpa_psf_data']['scaling_info']['fpa_position']
        print(f"\nFull FPA Details:")
        print(f"  PSF position on 2048×2048 detector: {pos}")
        print(f"  PSF size: 11×11 pixels (60.5×60.5 µm)")
        print(f"  Detector size: 2048×2048 pixels (11.3×11.3 mm)")
        print(f"  PSF occupies {(11*11)/(2048*2048)*100:.4f}% of detector area")

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate Full FPA Projection with CMV4000 Visualization")
    parser.add_argument("psf_file", help="Path to PSF file")
    parser.add_argument("--magnitude", type=float, default=3.0, help="Star magnitude (default: 3.0)")
    parser.add_argument("--trials", type=int, default=5, help="Number of Monte Carlo trials (default: 5)")
    parser.add_argument("--output", default="full_fpa_demo", help="Output directory (default: full_fpa_demo)")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("FULL FPA PROJECTION DEMONSTRATION")
    logger.info("Original PSF → 11×11 FPA → 2048×2048 CMV4000 Visualization")
    logger.info("="*80)
    
    # Run demonstration
    results = demonstrate_full_fpa_projection(
        args.psf_file, 
        magnitude=args.magnitude,
        num_trials=args.trials,
        output_dir=args.output
    )
    
    if results:
        logger.info("Demonstration completed successfully!")
    else:
        logger.error("Demonstration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 