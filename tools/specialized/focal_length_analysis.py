#!/usr/bin/env python3
"""
focal_length_analysis.py - Analyze the effect of focal length on centroiding performance
"""

import os
import subprocess
import sys
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Find a suitable PSF file in the current directory
    psf_files = glob.glob("*_FieldFlattener_*.txt")
    
    if not psf_files:
        logger.error("ERROR: No PSF files found in current directory!")
        logger.error("Please ensure PSF files with '_FieldFlattener_' in their name exist.")
        sys.exit(1)
    
    # Use the first PSF file (preferably on-axis if available)
    on_axis_files = [f for f in psf_files if "0deg" in f or "0.0deg" in f]
    psf_file = on_axis_files[0] if on_axis_files else psf_files[0]
    
    # Directory to save results
    output_dir = "focal_length_analysis"
    
    # Command with default values
    cmd = [
        sys.executable,  # Path to Python interpreter
        "star_tracker_pipeline.py",
        "--psf", psf_file,
        "--focal-length-analysis",
        "--focal-lengths", "20", "24", "28", "32", "36", "40", "50",
        "--magnitude", "3.0",
        "--simulations", "30",
        "--output", output_dir
    ]
    
    logger.info(f"Running focal length analysis using PSF file: {psf_file}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Execute the command
        result = subprocess.run(cmd, check=True)
        logger.info(f"Focal length analysis completed successfully!")
        logger.info(f"Results saved to {output_dir} directory")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running focal length analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()