#!/usr/bin/env python3
"""
Master script to generate all presentation figures.
Runs all visualization scripts and provides a comprehensive summary.
"""

import sys
import os
import subprocess
from pathlib import Path
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_name, description):
    """Run a visualization script and handle errors."""
    script_path = Path(__file__).parent / script_name
    
    logger.info(f"Running {description}...")
    start_time = time.time()
    
    try:
        # Run script with PYTHONPATH set
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent.parent.parent.parent)
        
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, env=env, 
                              cwd=str(Path(__file__).parent.parent.parent.parent))
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✓ {description} completed successfully ({duration:.1f}s)")
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
        else:
            logger.error(f"✗ {description} failed ({duration:.1f}s)")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ {description} failed with exception ({duration:.1f}s): {e}")
        return False
    
    return True

def check_output_directory():
    """Check and create output directory if needed."""
    output_dir = Path(__file__).parent.parent / "figures"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    return output_dir

def count_generated_files(output_dir):
    """Count generated PNG and PDF files."""
    png_files = list(output_dir.glob("*.png"))
    pdf_files = list(output_dir.glob("*.pdf"))
    
    return len(png_files), len(pdf_files)

def main():
    """Generate all presentation figures."""
    print("=" * 80)
    print("STAR TRACKER SIMULATION PRESENTATION FIGURE GENERATION")
    print("=" * 80)
    print()
    
    # Check output directory
    output_dir = check_output_directory()
    logger.info(f"Output directory: {output_dir}")
    
    # Count existing files
    initial_png, initial_pdf = count_generated_files(output_dir)
    logger.info(f"Existing files: {initial_png} PNG, {initial_pdf} PDF")
    print()
    
    # Define scripts to run
    scripts = [
        {
            'file': 'system_architecture.py',
            'description': 'System Architecture Diagrams',
            'expected_figures': ['pipeline_flowchart', 'multilevel_architecture', 'data_flow_diagram']
        },
        {
            'file': 'performance_characterization.py', 
            'description': 'Performance Characterization Plots',
            'expected_figures': ['centroiding_vs_magnitude', 'bearing_vector_vs_field_angle', 
                               'detection_success_rate', 'attitude_accuracy_vs_star_count', 'performance_summary_table']
        },
        {
            'file': 'physical_realism.py',
            'description': 'Physical Realism Demonstrations', 
            'expected_figures': ['psf_evolution', 'detector_response_comparison', 
                               'multi_star_scene', 'monte_carlo_error_propagation']
        },
        {
            'file': 'algorithm_validation.py',
            'description': 'Algorithm Validation Results',
            'expected_figures': ['bast_triangle_matching', 'quest_convergence', 
                               'end_to_end_validation', 'algorithm_validation_summary']
        },
        {
            'file': 'engineering_applications.py',
            'description': 'Engineering Applications',
            'expected_figures': ['sensor_trade_study', 'focal_length_optimization', 
                               'operational_envelope', 'requirements_verification']
        },
        {
            'file': 'development_timeline.py',
            'description': 'Development Timeline & Maturity',
            'expected_figures': ['development_timeline', 'validation_methodology', 
                               'capabilities_matrix', 'performance_benchmarks']
        }
    ]
    
    # Track results
    successful_scripts = 0
    total_expected_figures = sum(len(script['expected_figures']) for script in scripts)
    
    # Run each script
    start_total = time.time()
    
    for i, script in enumerate(scripts, 1):
        print(f"[{i}/{len(scripts)}] {script['description']}")
        print("-" * 60)
        
        success = run_script(script['file'], script['description'])
        
        if success:
            successful_scripts += 1
            # Check if expected figures were generated
            missing_figures = []
            for fig_name in script['expected_figures']:
                png_path = output_dir / f"{fig_name}.png"
                pdf_path = output_dir / f"{fig_name}.pdf"
                if not (png_path.exists() and pdf_path.exists()):
                    missing_figures.append(fig_name)
            
            if missing_figures:
                logger.warning(f"Missing expected figures: {', '.join(missing_figures)}")
        
        print()
    
    total_duration = time.time() - start_total
    
    # Final summary
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    
    final_png, final_pdf = count_generated_files(output_dir)
    generated_png = final_png - initial_png
    generated_pdf = final_pdf - initial_pdf
    
    print(f"Scripts run: {successful_scripts}/{len(scripts)}")
    print(f"Total execution time: {total_duration:.1f} seconds")
    print(f"Generated figures: {generated_png} PNG, {generated_pdf} PDF")
    print(f"Expected figures: {total_expected_figures * 2} total (PNG + PDF)")
    print(f"Output directory: {output_dir}")
    print()
    
    # List all generated files
    if generated_png > 0 or generated_pdf > 0:
        print("Generated files:")
        print("-" * 40)
        
        all_files = sorted(list(output_dir.glob("*.png")) + list(output_dir.glob("*.pdf")))
        for file_path in all_files:
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"  {file_path.name} ({file_size:.1f} KB)")
        
        print()
    
    # Success assessment
    success_rate = successful_scripts / len(scripts) * 100
    figure_rate = (generated_png + generated_pdf) / (total_expected_figures * 2) * 100
    
    print("SUCCESS ASSESSMENT:")
    print(f"  Script execution: {success_rate:.0f}% successful")
    print(f"  Figure generation: {figure_rate:.0f}% of expected figures")
    
    if success_rate == 100 and figure_rate >= 90:
        print("  Status: ✓ EXCELLENT - All scripts completed successfully")
    elif success_rate >= 80 and figure_rate >= 80:
        print("  Status: ✓ GOOD - Most scripts completed successfully")
    elif success_rate >= 60:
        print("  Status: ⚠ PARTIAL - Some scripts completed")
    else:
        print("  Status: ✗ POOR - Many scripts failed")
    
    print()
    print("USAGE INSTRUCTIONS:")
    print("-" * 40)
    print("1. All figures are saved in both PNG (presentation) and PDF (print) formats")
    print("2. Use PNG files for PowerPoint/Google Slides presentations")
    print("3. Use PDF files for LaTeX documents or high-quality prints")
    print("4. Figures are organized by category matching the presentation outline")
    print("5. Each script can be run independently if needed")
    
    return successful_scripts == len(scripts)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)