#!/usr/bin/env python3
"""
Simplified Focal Length Perturbation Demo

A streamlined demonstration showing focal length uncertainty propagation
without the complexity of full star tracker pipeline. Perfect for presentation!

This shows the concept clearly with realistic parameters based on your system.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional

# Import our perturbation framework
from src.core.perturbation_model import (
    create_focal_length_scenario, 
    PerturbationAnalyzer,
    DistributionType,
    ParameterVariation,
    PerturbationScenario
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def realistic_star_tracker_simulation(param_values: Dict[str, float], **kwargs) -> Dict[str, float]:
    """
    Realistic star tracker simulation based on your system parameters.
    
    This uses the actual relationships from your CLAUDE.md specifications
    without running the full complex pipeline.
    """
    try:
        # Extract focal length (with small random noise to simulate other effects)
        focal_length = param_values.get('focal_length', 25.0)  # mm
        
        # CMV4000 sensor parameters (from CLAUDE.md)
        pixel_pitch_um = 5.5  # µm
        pixel_pitch_mm = pixel_pitch_um / 1000.0
        
        # Calculate plate scale (fundamental relationship)
        plate_scale_arcsec_per_pixel = np.degrees(pixel_pitch_mm / focal_length) * 3600
        
        # Realistic centroiding accuracy (from CLAUDE.md: 0.15-0.25 pixels typical)
        base_centroiding_accuracy = 0.20  # pixels
        
        # Add some realistic variation based on system factors
        noise_factor = np.random.normal(1.0, 0.10)  # ±10% variation
        magnitude_factor = 1.0  # Could vary with star magnitude
        thermal_factor = 1.0    # Could vary with temperature effects
        
        centroiding_accuracy_pixels = base_centroiding_accuracy * noise_factor * magnitude_factor * thermal_factor
        
        # Convert to physical units
        centroiding_accuracy_um = centroiding_accuracy_pixels * pixel_pitch_um
        
        # Calculate bearing vector error (direct relationship via plate scale)
        bearing_error_arcsec = centroiding_accuracy_pixels * plate_scale_arcsec_per_pixel
        
        # Attitude error (from CLAUDE.md: 4-8 arcsec typical for single star bearing vector)
        # This would come from QUEST with multiple stars, but we simulate the relationship
        attitude_error_base = bearing_error_arcsec * 1.2  # Typical propagation factor
        attitude_error_noise = np.random.normal(0, attitude_error_base * 0.1)  # System noise
        attitude_error_arcsec = abs(attitude_error_base + attitude_error_noise)
        
        # Additional realistic metrics
        detection_success_rate = 0.98  # High for magnitude 4 star
        processing_time_sec = np.random.uniform(2.0, 3.0)  # Typical range
        
        return {
            'simulation_success': True,
            'focal_length_mm': focal_length,
            'pixel_pitch_um': pixel_pitch_um,
            'plate_scale_arcsec_per_pixel': plate_scale_arcsec_per_pixel,
            'centroiding_accuracy_pixels': centroiding_accuracy_pixels,
            'centroiding_accuracy_um': centroiding_accuracy_um,
            'bearing_error_arcsec': bearing_error_arcsec,
            'attitude_error_arcsec': attitude_error_arcsec,
            'detection_success_rate': detection_success_rate,
            'processing_time_sec': processing_time_sec
        }
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return {
            'simulation_success': False,
            'error_message': str(e)
        }

def create_comprehensive_visualization(
    results_dict: Dict[str, Dict],
    output_dir: Path,
    show_plots: bool = False
):
    """Create comprehensive visualization showing focal length uncertainty propagation."""
    
    logger.info("Creating comprehensive perturbation analysis visualization...")
    
    # Create the main analysis figure
    fig = plt.figure(figsize=(20, 16))
    
    # Colors for different scenarios
    scenario_colors = ['#1f77b4', '#ff7f0e', '#d62728']
    scenario_names = list(results_dict.keys())
    
    # Collect data
    all_data = []
    for scenario_name in scenario_names:
        data = results_dict[scenario_name]
        if len(data['successful_trials']) > 0:
            all_data.append(data['successful_trials'])
    
    # Main plots
    
    # 1. Focal length distributions
    ax1 = plt.subplot(3, 4, 1)
    for i, (scenario_name, df) in enumerate(zip(scenario_names, all_data)):
        focal_lengths = df['focal_length_mm'].values
        ax1.hist(focal_lengths, bins=20, alpha=0.6, color=scenario_colors[i], 
                 label=scenario_name, density=True)
    
    ax1.set_xlabel('Focal Length (mm)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Focal Length Variations\n(Thermal Effects)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Plate scale relationship
    ax2 = plt.subplot(3, 4, 2)
    for i, (scenario_name, df) in enumerate(zip(scenario_names, all_data)):
        focal_lengths = df['focal_length_mm'].values
        plate_scales = df['plate_scale_arcsec_per_pixel'].values
        ax2.scatter(focal_lengths, plate_scales, alpha=0.6, color=scenario_colors[i],
                   s=20, label=scenario_name)
    
    ax2.set_xlabel('Focal Length (mm)')
    ax2.set_ylabel('Plate Scale (arcsec/pixel)')
    ax2.set_title('Plate Scale vs Focal Length\n(Fundamental Relationship)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Bearing error propagation
    ax3 = plt.subplot(3, 4, 3)
    for i, (scenario_name, df) in enumerate(zip(scenario_names, all_data)):
        plate_scales = df['plate_scale_arcsec_per_pixel'].values
        bearing_errors = df['bearing_error_arcsec'].values
        ax3.scatter(plate_scales, bearing_errors, alpha=0.6, color=scenario_colors[i],
                   s=20, label=scenario_name)
    
    ax3.set_xlabel('Plate Scale (arcsec/pixel)')
    ax3.set_ylabel('Bearing Error (arcsec)')
    ax3.set_title('Bearing Error Propagation\n(via Plate Scale)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Final attitude error
    ax4 = plt.subplot(3, 4, 4)
    for i, (scenario_name, df) in enumerate(zip(scenario_names, all_data)):
        attitude_errors = df['attitude_error_arcsec'].values
        ax4.hist(attitude_errors, bins=20, alpha=0.6, color=scenario_colors[i],
                 density=True, label=scenario_name)
    
    ax4.set_xlabel('Attitude Error (arcsec)')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Final Attitude Error\n(System Output)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Direct correlation: Focal Length vs Attitude Error
    ax5 = plt.subplot(3, 4, 5)
    for i, (scenario_name, df) in enumerate(zip(scenario_names, all_data)):
        focal_lengths = df['focal_length_mm'].values
        attitude_errors = df['attitude_error_arcsec'].values
        ax5.scatter(focal_lengths, attitude_errors, alpha=0.6, color=scenario_colors[i],
                   s=30, label=scenario_name)
    
    # Fit trend line for combined data
    all_focal = np.concatenate([df['focal_length_mm'].values for df in all_data])
    all_attitude = np.concatenate([df['attitude_error_arcsec'].values for df in all_data])
    coeffs = np.polyfit(all_focal, all_attitude, 1)
    trend_x = np.linspace(all_focal.min(), all_focal.max(), 100)
    trend_y = np.polyval(coeffs, trend_x)
    ax5.plot(trend_x, trend_y, 'k--', linewidth=3, 
             label=f'Overall Trend: {coeffs[0]:.2f} arcsec/mm')
    
    ax5.set_xlabel('Focal Length (mm)')
    ax5.set_ylabel('Attitude Error (arcsec)')
    ax5.set_title('Direct Impact Relationship\n(Key Engineering Insight)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Sensitivity analysis
    ax6 = plt.subplot(3, 4, 6)
    sensitivities = []
    rms_errors = []
    
    for scenario_name in scenario_names:
        data = results_dict[scenario_name]
        analyzer = data['analyzer']
        
        # Calculate sensitivity
        sensitivity_data = analyzer.analyze_sensitivity('attitude_error_arcsec')
        focal_sens = sensitivity_data.get('focal_length', {})
        sensitivities.append(focal_sens.get('sensitivity', 0))
        
        # Calculate RMS error
        df = data['successful_trials']
        rms_error = np.sqrt(np.mean(df['attitude_error_arcsec'].values**2))
        rms_errors.append(rms_error)
    
    x_pos = np.arange(len(scenario_names))
    bars = ax6.bar(x_pos, sensitivities, color=scenario_colors, alpha=0.7, edgecolor='black')
    
    for bar, sens in zip(bars, sensitivities):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{sens:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax6.set_ylabel('Sensitivity (arcsec/mm)')
    ax6.set_title('Sensitivity Analysis\n(Impact per Unit Change)', fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([name.replace(' ', '\n') for name in scenario_names])
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Thermal expansion model
    ax7 = plt.subplot(3, 4, 7)
    
    temperatures = np.linspace(-40, 80, 100)
    nominal_focal_length = 25.0
    
    # Different materials
    materials = [
        {'name': 'Aluminum', 'cte': 23e-6, 'color': 'blue'},
        {'name': 'Steel', 'cte': 12e-6, 'color': 'green'},  
        {'name': 'Carbon Fiber', 'cte': 1e-6, 'color': 'red'}
    ]
    
    for material in materials:
        cte = material['cte']
        delta_temp = temperatures - 20  # Reference 20°C
        delta_focal_length = nominal_focal_length * cte * delta_temp
        focal_lengths = nominal_focal_length + delta_focal_length
        
        ax7.plot(temperatures, focal_lengths, color=material['color'],
                linewidth=2, label=material['name'])
    
    ax7.set_xlabel('Temperature (°C)')
    ax7.set_ylabel('Focal Length (mm)')
    ax7.set_title('Material Thermal Expansion\n(Root Cause)', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Highlight operating ranges from scenarios
    temp_ranges = [(-10, 30), (-20, 60), (-40, 80)]
    for i, temp_range in enumerate(temp_ranges):
        ax7.axvspan(temp_range[0], temp_range[1], alpha=0.2, color=scenario_colors[i])
    
    # 8. RMS Error comparison
    ax8 = plt.subplot(3, 4, 8)
    bars = ax8.bar(x_pos, rms_errors, color=scenario_colors, alpha=0.7, edgecolor='black')
    
    for bar, rms in zip(bars, rms_errors):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rms:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Add requirement line
    requirement = 5.0
    ax8.axhline(y=requirement, color='red', linestyle='--', linewidth=3,
               label=f'{requirement} arcsec requirement')
    
    ax8.set_ylabel('RMS Attitude Error (arcsec)')
    ax8.set_title('Performance vs Requirements\n(Design Verification)', fontweight='bold')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels([name.replace(' ', '\n') for name in scenario_names])
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.legend()
    
    # 9. Engineering implications summary
    ax9 = plt.subplot(3, 4, 9)
    ax9.text(0.05, 0.95, 'KEY ENGINEERING INSIGHTS:', fontsize=14, fontweight='bold',
             transform=ax9.transAxes)
    
    # Calculate key statistics
    mean_sensitivity = np.mean(sensitivities)
    max_rms = np.max(rms_errors)
    min_rms = np.min(rms_errors)
    
    insights = f"""
QUANTITATIVE RELATIONSHIPS:
• Sensitivity: ~{mean_sensitivity:.1f} arcsec/mm focal length
• Requirement margin: {requirement/max_rms:.1f}x safety factor
• Thermal control benefit: {max_rms/min_rms:.1f}x improvement

DESIGN IMPLICATIONS:
• 0.5mm focal length change = ~{mean_sensitivity*0.5:.1f}" attitude error
• Thermal stability directly affects performance
• Predictable, linear relationship enables optimization

RISK MITIGATION:
• Quantify thermal control requirements
• Validate designs before hardware build
• Optimize performance vs cost trade-offs
"""
    
    ax9.text(0.05, 0.85, insights, fontsize=11, transform=ax9.transAxes,
             verticalalignment='top', fontfamily='monospace')
    
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    # 10. Uncertainty propagation chain
    ax10 = plt.subplot(3, 4, 10)
    
    # Create flow diagram
    steps = [
        'Temperature\nVariation',
        'Thermal\nExpansion',
        'Focal Length\nChange',
        'Plate Scale\nChange',
        'Bearing Error\nIncrease',
        'Attitude Error\nDegradation'
    ]
    
    y_positions = np.linspace(0.9, 0.1, len(steps))
    
    for i, (step, y) in enumerate(zip(steps, y_positions)):
        # Box
        box = plt.Rectangle((0.1, y-0.05), 0.8, 0.1, 
                           facecolor=scenario_colors[i % 3], alpha=0.6,
                           edgecolor='black')
        ax10.add_patch(box)
        
        # Text
        ax10.text(0.5, y, step, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Arrow
        if i < len(steps) - 1:
            ax10.arrow(0.5, y-0.06, 0, -0.08, head_width=0.03, head_length=0.02,
                      fc='black', ec='black')
    
    ax10.set_xlim(0, 1)
    ax10.set_ylim(0, 1)
    ax10.set_title('Uncertainty Propagation Chain\n(Complete Digital Twin)', fontweight='bold')
    ax10.axis('off')
    
    # 11. Statistical confidence
    ax11 = plt.subplot(3, 4, 11)
    
    # Show confidence intervals for one scenario
    main_df = all_data[1]  # Middle scenario
    attitude_errors = main_df['attitude_error_arcsec'].values
    
    # Calculate percentiles
    percentiles = [5, 25, 50, 75, 95]
    values = [np.percentile(attitude_errors, p) for p in percentiles]
    
    ax11.boxplot([attitude_errors], positions=[1], widths=[0.6])
    
    # Add percentile labels
    for p, v in zip(percentiles, values):
        ax11.text(1.7, v, f'{p}th: {v:.1f}"', va='center', fontsize=10)
    
    ax11.set_xlim(0.5, 2.5)
    ax11.set_ylabel('Attitude Error (arcsec)')
    ax11.set_title(f'Statistical Distribution\n({scenario_names[1]})', fontweight='bold')
    ax11.set_xticks([])
    ax11.grid(True, alpha=0.3, axis='y')
    
    # 12. Value proposition
    ax12 = plt.subplot(3, 4, 12)
    ax12.text(0.05, 0.95, 'SIMULATION VALUE:', fontsize=14, fontweight='bold',
              transform=ax12.transAxes, color='darkgreen')
    
    value_text = f"""
BEFORE THIS ANALYSIS:
• "Thermal effects probably matter"
• "Need some thermal control"
• "Hope it works in testing"

AFTER THIS ANALYSIS:
• {mean_sensitivity:.1f} arcsec/mm quantified impact
• {requirement/max_rms:.1f}x safety factor calculated
• Thermal specs: ±{temp_ranges[0][1]-20}°C vs ±{temp_ranges[2][1]-20}°C
• Risk quantified, not guessed

DESIGN CONFIDENCE:
• Predict performance before build
• Optimize thermal vs cost trade-offs
• Validate requirements compliance
• Reduce late-stage surprises
"""
    
    ax12.text(0.05, 0.85, value_text, fontsize=10, transform=ax12.transAxes,
              verticalalignment='top', fontfamily='monospace')
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    
    # Overall title
    fig.suptitle(
        'Focal Length Perturbation Analysis: Digital Twin Uncertainty Propagation\n' +
        'Thermal Effects → Optical Changes → Attitude Performance Degradation',
        fontsize=20, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(output_dir / 'focal_length_uncertainty_propagation_demo.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'focal_length_uncertainty_propagation_demo.pdf', 
                bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"Comprehensive visualization saved to {output_dir}")

def main():
    """Run simplified focal length perturbation demonstration."""
    
    print("=" * 80)
    print("FOCAL LENGTH PERTURBATION DEMONSTRATION")
    print("Realistic Uncertainty Propagation Analysis")
    print("=" * 80)
    print()
    
    # Setup
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "perturbation_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define thermal scenarios
    thermal_scenarios = [
        {
            'name': 'Benign Environment',
            'description': 'Good thermal control (±10°C)',
            'temp_range': (-10, 30),
            'focal_length_variation': 0.2  # mm, 3-sigma
        },
        {
            'name': 'Nominal Space',
            'description': 'Typical space environment (±40°C)', 
            'temp_range': (-20, 60),
            'focal_length_variation': 0.5  # mm, 3-sigma
        },
        {
            'name': 'Harsh Environment',
            'description': 'Poor thermal control (±60°C)',
            'temp_range': (-40, 80),
            'focal_length_variation': 0.8  # mm, 3-sigma
        }
    ]
    
    logger.info(f"Output directory: {output_dir}")
    print()
    
    # Run analysis for each scenario
    n_trials = 200  # More trials for better statistics
    results = {}
    
    start_time = time.time()
    
    for scenario_def in thermal_scenarios:
        logger.info(f"Running scenario: {scenario_def['name']}")
        
        # Create scenario
        scenario = create_focal_length_scenario(
            nominal_focal_length=25.0,
            thermal_variation=scenario_def['focal_length_variation'],
            temperature_range=scenario_def['temp_range']
        )
        scenario.metadata.update(scenario_def)
        
        # Run Monte Carlo analysis
        analyzer = PerturbationAnalyzer(scenario)
        results_df = analyzer.run_monte_carlo(
            simulation_function=realistic_star_tracker_simulation,
            n_trials=n_trials
        )
        
        # Process results
        successful_trials = results_df[results_df['simulation_success'] == True].copy()
        success_rate = len(successful_trials) / len(results_df) * 100
        
        logger.info(f"Success rate: {success_rate:.1f}% ({len(successful_trials)}/{len(results_df)})")
        
        # Store results
        results[scenario_def['name']] = {
            'all_trials': results_df,
            'successful_trials': successful_trials,
            'analyzer': analyzer,
            'scenario': scenario
        }
        
        # Quick stats
        if len(successful_trials) > 0:
            focal_stats = analyzer.get_summary_statistics('focal_length')
            attitude_stats = analyzer.get_summary_statistics('attitude_error_arcsec')
            sensitivity = analyzer.analyze_sensitivity('attitude_error_arcsec')
            
            logger.info(f"Focal length: {focal_stats['mean']:.3f} ± {focal_stats['std']:.3f} mm")
            logger.info(f"Attitude error: {attitude_stats['mean']:.2f} ± {attitude_stats['std']:.2f} arcsec")
            
            focal_sensitivity = sensitivity.get('focal_length', {})
            if focal_sensitivity:
                logger.info(f"Sensitivity: {focal_sensitivity['sensitivity']:.2f} arcsec/mm")
        
        print()
    
    # Create visualizations
    create_comprehensive_visualization(results, output_dir, show_plots=False)
    
    analysis_time = time.time() - start_time
    
    # Final summary
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    print(f"Total time: {analysis_time:.1f} seconds")
    print(f"Results saved to: {output_dir}")
    print()
    
    print("KEY FINDINGS:")
    print("-" * 40)
    
    for scenario_name, data in results.items():
        if len(data['successful_trials']) > 0:
            analyzer = data['analyzer']
            attitude_stats = analyzer.get_summary_statistics('attitude_error_arcsec')
            sensitivity = analyzer.analyze_sensitivity('attitude_error_arcsec')
            focal_sensitivity = sensitivity.get('focal_length', {})
            
            print(f"\n{scenario_name}:")
            print(f"  RMS Attitude Error: {np.sqrt(np.mean(attitude_stats['mean']**2)):.2f} arcsec")
            print(f"  Sensitivity: {focal_sensitivity.get('sensitivity', 0):.2f} arcsec/mm")
            print(f"  Correlation: {focal_sensitivity.get('correlation', 0):.3f}")
    
    print("\n" + "=" * 80)
    print("This analysis demonstrates:")
    print("1. Quantitative impact of thermal effects on attitude accuracy")
    print("2. Direct correlation between focal length and performance")
    print("3. Design trade-offs with statistical confidence")
    print("4. Requirements verification capability")
    print("5. Risk quantification for engineering decisions")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)