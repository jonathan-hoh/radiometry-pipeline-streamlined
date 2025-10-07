#!/usr/bin/env python3
"""
Radiometry and Poisson Statistics Demonstration

This script showcases the sophisticated radiometry modeling capabilities of the 
star tracker simulation, including proper Poisson noise statistics and photon-level 
accuracy. Perfect for demonstrating physical realism beyond simple geometric models.

Generates: Comprehensive figure showing radiometric chain from star magnitude to 
detector response with realistic noise modeling.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import logging
from scipy.stats import poisson

# Import pipeline components
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.core.starcamera_model import star, scene, calculate_optical_signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_radiometry_demonstration():
    """Create comprehensive radiometry and Poisson statistics demonstration."""
    
    # Set up output directory
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize pipeline to get camera and scene models
    pipeline = StarTrackerPipeline()
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Star magnitude to photon count relationship
    ax1 = plt.subplot(3, 4, 1)
    
    magnitudes = np.linspace(1, 8, 50)
    photon_counts = []
    
    for mag in magnitudes:
        star_obj = star(magnitude=mag, passband=pipeline.camera.passband)
        photons = calculate_optical_signal(star_obj, pipeline.camera, pipeline.scene)
        photon_counts.append(photons)
    
    photon_counts = np.array(photon_counts)
    
    ax1.semilogy(magnitudes, photon_counts, 'b-', linewidth=3, label='Radiometric Model')
    
    # Highlight key magnitudes
    key_mags = [3.0, 4.0, 5.0, 6.0]
    key_colors = ['red', 'orange', 'green', 'purple']
    for i, (mag, color) in enumerate(zip(key_mags, key_colors)):
        star_obj = star(magnitude=mag, passband=pipeline.camera.passband)
        photons = calculate_optical_signal(star_obj, pipeline.camera, pipeline.scene)
        ax1.plot(mag, photons, 'o', color=color, markersize=10, 
                label=f'Mag {mag}: {photons:.0f} photons')
    
    ax1.set_xlabel('Star Magnitude', fontsize=12)
    ax1.set_ylabel('Photon Count (per integration)', fontsize=12)
    ax1.set_title('Radiometric Chain: Magnitude → Photons\n(Physical Light Collection)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Poisson statistics demonstration
    ax2 = plt.subplot(3, 4, 2)
    
    # Show Poisson distributions for different photon levels
    photon_levels = [100, 500, 2000, 8000]
    colors = ['red', 'orange', 'green', 'blue']
    
    max_range = 0
    for photon_level, color in zip(photon_levels, colors):
        # Generate Poisson distribution
        x_range = np.arange(max(0, photon_level - 5*np.sqrt(photon_level)), 
                           photon_level + 5*np.sqrt(photon_level))
        poisson_pmf = poisson.pmf(x_range, photon_level)
        
        ax2.plot(x_range, poisson_pmf, color=color, linewidth=2, 
                label=f'{photon_level} photons (σ={np.sqrt(photon_level):.1f})')
        max_range = max(max_range, x_range[-1])
    
    ax2.set_xlabel('Detected Photon Count', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Poisson Noise Statistics\n(Fundamental Quantum Noise)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Signal-to-noise ratio analysis
    ax3 = plt.subplot(3, 4, 3)
    
    # SNR = signal / sqrt(signal + noise^2) for Poisson + read noise
    read_noise = 13.0  # electrons RMS (CMV4000)
    
    signal_levels = np.logspace(1, 5, 100)  # 10 to 100,000 photons
    snr_shot_limited = np.sqrt(signal_levels)  # Shot noise limited
    snr_read_limited = signal_levels / read_noise  # Read noise limited
    snr_combined = signal_levels / np.sqrt(signal_levels + read_noise**2)  # Combined
    
    ax3.loglog(signal_levels, snr_shot_limited, 'b--', linewidth=2, 
              label='Shot noise limited (√N)')
    ax3.loglog(signal_levels, snr_read_limited, 'r--', linewidth=2, 
              label='Read noise limited (N/σ_read)')
    ax3.loglog(signal_levels, snr_combined, 'k-', linewidth=3, 
              label='Combined (realistic)')
    
    # Mark operating points for different magnitudes
    for i, (mag, color) in enumerate(zip(key_mags, key_colors)):
        star_obj = star(magnitude=mag, passband=pipeline.camera.passband)
        photons = calculate_optical_signal(star_obj, pipeline.camera, pipeline.scene)
        snr = photons / np.sqrt(photons + read_noise**2)
        ax3.plot(photons, snr, 'o', color=color, markersize=8,
                label=f'Mag {mag} (SNR={snr:.1f})')
    
    ax3.set_xlabel('Signal Level (photons)', fontsize=12)
    ax3.set_ylabel('Signal-to-Noise Ratio', fontsize=12)
    ax3.set_title('SNR vs Signal Level\n(Detection Performance)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Detector response simulation
    ax4 = plt.subplot(3, 4, 4)
    
    # Simulate detector response for different star magnitudes
    n_trials = 1000
    
    measured_counts = {}
    for mag in key_mags:
        star_obj = star(magnitude=mag, passband=pipeline.camera.passband)
        true_photons = calculate_optical_signal(star_obj, pipeline.camera, pipeline.scene)
        
        # Simulate quantum efficiency
        qe = pipeline.camera.fpa.qe  # ~0.6 for CMV4000
        detected_photons = np.random.binomial(int(true_photons), qe, n_trials)
        
        # Add read noise
        read_noise_samples = np.random.normal(0, read_noise, n_trials)
        final_counts = detected_photons + read_noise_samples
        final_counts = np.maximum(final_counts, 0)  # Clip negative values
        
        measured_counts[mag] = final_counts
    
    # Create box plot
    box_data = [measured_counts[mag] for mag in key_mags]
    bp = ax4.boxplot(box_data, labels=[f'Mag {mag}' for mag in key_mags], patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], key_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('Star Magnitude', fontsize=12)
    ax4.set_ylabel('Detected Counts (electrons)', fontsize=12)
    ax4.set_title('Realistic Detector Response\n(QE + Poisson + Read Noise)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Centroiding accuracy vs SNR
    ax5 = plt.subplot(3, 4, 5)
    
    snr_range = np.logspace(0.5, 3, 50)  # SNR from ~3 to 1000
    
    # Theoretical centroiding accuracy (Cramer-Rao bound approximation)
    # σ_centroid ≈ σ_psf / SNR for well-sampled PSFs
    psf_width_pixels = 2.0  # Typical PSF width
    centroiding_accuracy_theoretical = psf_width_pixels / snr_range
    
    # Practical centroiding accuracy (includes systematic errors)
    centroiding_accuracy_practical = np.sqrt((psf_width_pixels / snr_range)**2 + 0.05**2)
    
    ax5.loglog(snr_range, centroiding_accuracy_theoretical, 'b--', linewidth=2,
              label='Theoretical (Cramér-Rao)')
    ax5.loglog(snr_range, centroiding_accuracy_practical, 'r-', linewidth=3,
              label='Practical (with systematics)')
    
    # Mark operating points
    for i, (mag, color) in enumerate(zip(key_mags, key_colors)):
        star_obj = star(magnitude=mag, passband=pipeline.camera.passband)
        photons = calculate_optical_signal(star_obj, pipeline.camera, pipeline.scene)
        snr = photons / np.sqrt(photons + read_noise**2)
        accuracy = np.sqrt((psf_width_pixels / snr)**2 + 0.05**2)
        ax5.plot(snr, accuracy, 'o', color=color, markersize=8)
    
    ax5.set_xlabel('Signal-to-Noise Ratio', fontsize=12)
    ax5.set_ylabel('Centroiding Accuracy (pixels)', fontsize=12)
    ax5.set_title('Centroiding Performance vs SNR\n(Noise Propagation to Accuracy)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. CMV4000 sensor model validation
    ax6 = plt.subplot(3, 4, 6)
    
    # Display key sensor parameters
    sensor_params = {
        'Resolution': f"{pipeline.camera.fpa.x_pixels}×{pipeline.camera.fpa.y_pixels} pixels",
        'Pixel Pitch': f"{pipeline.camera.fpa.pitch} µm",
        'Quantum Efficiency': f"{pipeline.camera.fpa.qe*100:.0f}%",
        'Full Well': f"{pipeline.camera.fpa.full_well:,} e⁻",
        'Read Noise': f"{pipeline.camera.fpa.read_noise:.1f} e⁻ RMS",
        'Dark Current': f"{pipeline.camera.fpa.dark_current_ref} e⁻/s @ {pipeline.camera.fpa.dark_current_ref_temp}°C"
    }
    
    y_positions = np.arange(len(sensor_params))
    ax6.barh(y_positions, [1]*len(sensor_params), color='lightblue', alpha=0.7)
    
    for i, (param, value) in enumerate(sensor_params.items()):
        ax6.text(0.05, i, f"{param}:", fontweight='bold', va='center', fontsize=11)
        ax6.text(0.55, i, value, va='center', fontsize=11)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(-0.5, len(sensor_params)-0.5)
    ax6.set_yticks([])
    ax6.set_xticks([])
    ax6.set_title('CMV4000 Sensor Model\n(Hardware-Accurate Parameters)', fontsize=14, fontweight='bold')
    
    # Remove spines
    for spine in ax6.spines.values():
        spine.set_visible(False)
    
    # 7. Noise sources breakdown
    ax7 = plt.subplot(3, 4, 7)
    
    # Calculate noise contributions for a typical case
    typical_signal = 3000  # photons
    shot_noise = np.sqrt(typical_signal)
    read_noise_val = 13.0
    dark_noise = np.sqrt(pipeline.camera.fpa.dark_current_ref * pipeline.scene.int_time/1000)
    quantization_noise = 1/np.sqrt(12)  # For well-exposed signals
    
    noise_sources = ['Shot Noise\n(√N)', 'Read Noise\n(13e⁻)', 'Dark Current\n(temp dependent)', 
                    'Quantization\n(ADC)']
    noise_values = [shot_noise, read_noise_val, dark_noise, quantization_noise]
    colors_noise = ['red', 'blue', 'green', 'orange']
    
    bars = ax7.bar(noise_sources, noise_values, color=colors_noise, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, noise_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}e⁻', ha='center', va='bottom', fontweight='bold')
    
    # Show total noise
    total_noise = np.sqrt(sum([n**2 for n in noise_values]))
    ax7.axhline(y=total_noise, color='black', linestyle='--', linewidth=2,
               label=f'Total: {total_noise:.1f}e⁻')
    
    ax7.set_ylabel('Noise Level (electrons RMS)', fontsize=12)
    ax7.set_title(f'Noise Source Analysis\n(3000 photon signal)', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.legend()
    
    # 8. Quantum efficiency spectral response
    ax8 = plt.subplot(3, 4, 8)
    
    # Typical CMV4000 QE curve (approximated)
    wavelengths = np.linspace(400, 1000, 100)  # nm
    
    # Simplified QE model
    qe_peak = 0.65
    qe_curve = qe_peak * np.exp(-((wavelengths - 550)**2) / (2 * 150**2))
    qe_curve *= (wavelengths > 400) * (wavelengths < 1000)  # Cutoff filters
    
    ax8.plot(wavelengths, qe_curve * 100, 'b-', linewidth=3, label='CMV4000 QE')
    
    # Show passband
    passband_min, passband_max = pipeline.camera.passband[0]*1000, pipeline.camera.passband[1]*1000
    ax8.axvspan(passband_min, passband_max, alpha=0.3, color='yellow', 
               label=f'System Passband\n({passband_min:.0f}-{passband_max:.0f}nm)')
    
    ax8.set_xlabel('Wavelength (nm)', fontsize=12)
    ax8.set_ylabel('Quantum Efficiency (%)', fontsize=12)
    ax8.set_title('Spectral Response\n(Wavelength-Dependent Detection)', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    ax8.set_ylim(0, 70)
    
    # 9. Integration time effects
    ax9 = plt.subplot(3, 4, 9)
    
    integration_times = np.linspace(1, 50, 50)  # ms
    base_photons = 200  # photons/ms
    
    snr_vs_time = []
    for int_time in integration_times:
        signal = base_photons * int_time
        dark_current = pipeline.camera.fpa.dark_current_ref * int_time/1000
        total_noise = np.sqrt(signal + dark_current + read_noise**2)
        snr = signal / total_noise
        snr_vs_time.append(snr)
    
    ax9.plot(integration_times, snr_vs_time, 'g-', linewidth=3)
    ax9.axvline(x=pipeline.scene.int_time, color='red', linestyle='--', linewidth=2,
               label=f'Nominal: {pipeline.scene.int_time}ms')
    
    ax9.set_xlabel('Integration Time (ms)', fontsize=12)
    ax9.set_ylabel('Signal-to-Noise Ratio', fontsize=12)
    ax9.set_title('SNR vs Integration Time\n(Exposure Optimization)', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    # 10. Temperature effects on dark current
    ax10 = plt.subplot(3, 4, 10)
    
    temperatures = np.linspace(-40, 60, 100)  # °C
    ref_temp = pipeline.camera.fpa.dark_current_ref_temp
    dark_coeff = pipeline.camera.fpa.dark_current_coefficient
    
    dark_current_vs_temp = pipeline.camera.fpa.dark_current_ref + dark_coeff * (temperatures - ref_temp)
    dark_current_vs_temp = np.maximum(dark_current_vs_temp, 0)
    
    ax10.semilogy(temperatures, dark_current_vs_temp, 'purple', linewidth=3)
    ax10.axvline(x=pipeline.scene.temp, color='red', linestyle='--', linewidth=2,
                label=f'Operating: {pipeline.scene.temp}°C')
    
    ax10.set_xlabel('Temperature (°C)', fontsize=12)
    ax10.set_ylabel('Dark Current (e⁻/s)', fontsize=12)
    ax10.set_title('Dark Current vs Temperature\n(Thermal Noise Effects)', fontsize=14, fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.legend()
    
    # 11. Full-well capacity effects
    ax11 = plt.subplot(3, 4, 11)
    
    signal_range = np.linspace(0, pipeline.camera.fpa.full_well * 1.5, 100)
    
    # Linear response up to full well, then saturation
    linear_response = np.minimum(signal_range, pipeline.camera.fpa.full_well)
    
    ax11.plot(signal_range, linear_response, 'b-', linewidth=3, label='Detector Response')
    ax11.axhline(y=pipeline.camera.fpa.full_well, color='red', linestyle='--', linewidth=2,
                label=f'Full Well: {pipeline.camera.fpa.full_well:,}e⁻')
    ax11.axvline(x=pipeline.camera.fpa.full_well, color='red', linestyle='--', linewidth=2)
    
    # Mark operating points
    for i, (mag, color) in enumerate(zip(key_mags, key_colors)):
        star_obj = star(magnitude=mag, passband=pipeline.camera.passband)
        photons = calculate_optical_signal(star_obj, pipeline.camera, pipeline.scene)
        response = min(photons, pipeline.camera.fpa.full_well)
        ax11.plot(photons, response, 'o', color=color, markersize=8)
    
    ax11.set_xlabel('Input Signal (electrons)', fontsize=12)
    ax11.set_ylabel('Detected Signal (electrons)', fontsize=12)
    ax11.set_title('Detector Linearity\n(Saturation Effects)', fontsize=14, fontweight='bold')
    ax11.grid(True, alpha=0.3)
    ax11.legend()
    
    # 12. Summary - radiometry pipeline
    ax12 = plt.subplot(3, 4, 12)
    
    pipeline_steps = [
        'Star Magnitude',
        'Photon Flux',
        'Optical Collection',
        'Quantum Detection',
        'Electron Signal',
        'Noise Addition',
        'ADC Conversion',
        'Digital Image'
    ]
    
    y_positions = np.linspace(0.9, 0.1, len(pipeline_steps))
    
    for i, (step, y) in enumerate(zip(pipeline_steps, y_positions)):
        # Box
        box_color = plt.cm.viridis(i / len(pipeline_steps))
        box = plt.Rectangle((0.1, y-0.05), 0.8, 0.08, 
                           facecolor=box_color, alpha=0.7, edgecolor='black')
        ax12.add_patch(box)
        
        # Text
        ax12.text(0.5, y, step, ha='center', va='center', fontweight='bold', 
                 color='white', fontsize=11)
        
        # Arrow
        if i < len(pipeline_steps) - 1:
            ax12.arrow(0.5, y-0.06, 0, -0.07, head_width=0.03, head_length=0.015,
                      fc='black', ec='black')
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.set_title('Complete Radiometric Pipeline\n(Physical Signal Chain)', fontsize=14, fontweight='bold')
    ax12.axis('off')
    
    # Overall title
    fig.suptitle('Radiometry and Poisson Statistics Modeling\n' +
                 'Physics-Based Photon Detection with Hardware-Accurate Noise Models',
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_dir / 'radiometry_poisson_demonstration.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'radiometry_poisson_demonstration.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Radiometry demonstration saved to {output_dir}")
    print(f"Generated radiometry demonstration: {output_dir / 'radiometry_poisson_demonstration.png'}")

def main():
    """Generate radiometry and Poisson statistics demonstration."""
    
    print("=" * 80)
    print("RADIOMETRY AND POISSON STATISTICS DEMONSTRATION")
    print("=" * 80)
    print()
    
    create_radiometry_demonstration()
    
    print()
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("This demonstration showcases:")
    print("1. Complete radiometric modeling from magnitude to photons")
    print("2. Proper Poisson noise statistics and quantum effects")
    print("3. Hardware-accurate CMV4000 sensor parameters")
    print("4. Signal-to-noise analysis and detection limits")
    print("5. Temperature and integration time effects")
    print("6. Full detector response chain modeling")
    print()
    print("Perfect for showing the physical realism of your simulation!")

if __name__ == "__main__":
    main()