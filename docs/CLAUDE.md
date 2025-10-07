# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Directory Structure

The repository is organized as follows:

```
radiometry-pipeline/
├── src/                          # Core pipeline source code
│   ├── core/                     # Main pipeline components
│   │   ├── star_tracker_pipeline.py
│   │   ├── starcamera_model.py
│   │   ├── psf_plot.py
│   │   └── psf_photon_simulation.py
│   ├── multi_star/               # Multi-star functionality
│   └── BAST/                     # BAST algorithms
├── tools/                        # Analysis and utility scripts
│   ├── interactive/              # Main interactive analysis tools
│   ├── specialized/              # Specialized analysis tools
│   └── debug/                    # Debug and diagnostic tools
├── data/                         # Data directories
│   ├── PSF_sims/                 # PSF simulation data
│   ├── examples/                 # Example data and demos
│   └── tests/                    # Test data
├── docs/                         # Documentation
├── outputs/                      # Generated outputs
└── config/                       # Configuration files
```

**Important**: All tools must be run with `PYTHONPATH=.` to ensure proper import resolution.

## Core Architecture

This is a star tracker radiometry pipeline that simulates the complete signal chain from optical PSFs through photon detection to bearing vector calculation. The system is built around the **StarTrackerPipeline** class in `src/core/star_tracker_pipeline.py` which serves as the central orchestrator - all analysis flows through this class.

### Key Components

- **StarTrackerPipeline** (src/core/star_tracker_pipeline.py): Central orchestrator for all analysis
- **PSF Processing** (src/core/psf_plot.py): Parses PSF files and handles metadata extraction
- **Camera Modeling** (src/core/starcamera_model.py): CMV4000 sensor simulation and radiometric calculations
- **Photon Simulation** (src/core/psf_photon_simulation.py): Poisson noise and detector response modeling
- **Centroiding** (src/BAST/ modules): Star detection and subpixel centroiding algorithms

### Dual Analysis Architecture

The pipeline implements two parallel analysis paths:
1. **Original PSF Analysis**: High-resolution (128×128, 0.5µm/pixel) for optical design
2. **FPA Projected Analysis**: Realistic detector simulation (11×11, 5.5µm/pixel) for hardware validation

## Common Development Commands

### Basic Analysis
```bash
# Interactive single PSF analysis with performance comparison
PYTHONPATH=. python tools/interactive/debug_centroiding.py

# Field angle sweep across a PSF generation
PYTHONPATH=. python tools/interactive/angle_sweep.py

# Comprehensive system demonstration
PYTHONPATH=. python tools/specialized/full_fpa_demo.py data/PSF_sims/Gen_1/0_deg.txt --magnitude 3.0 --trials 5
```

### Testing and Validation
```bash
# Install dependencies
pip install -r config/requirements.txt

# Run diagnostic analysis
PYTHONPATH=. python tools/interactive/fpa_diagnostic.py data/PSF_sims/Gen_1/0_deg.txt

# Quality analysis of PSF files
PYTHONPATH=. python tools/specialized/analyze_psf_quality.py --psf-dir data/PSF_sims/
```

## PSF Data Structure

PSF files are organized in `data/PSF_sims/` with generation-based directories:
```
data/PSF_sims/
├── Gen_1/          # 128×128, 0.5µm/pixel
├── Gen_2/          # 32×32, 0.232µm/pixel  
└── Gen_N/          # New generations auto-detected
```

Files follow naming convention: `{angle}_deg.txt` (e.g., `0_deg.txt`, `8.49_deg.txt`)

## Development Guidelines

### Code Organization
- **StarTrackerPipeline class**: Never bypass this central orchestrator for core functionality
- **Function modularity**: Keep analysis functions focused on single parameters
- **Visualization pattern**: Use `visualize_*_results()` pattern for all plotting functions
- **Separate calculation and visualization logic**

### Algorithm Integration
- New centroiding algorithms must return `(x_centroid, y_centroid, total_intensity)` tuples
- Bearing vector calculations must maintain established focal length normalization
- All detector-related functions should support the CMV4000 specifications

### Physical Units (Always Required)
- **Pixels**: Image coordinates
- **Microns**: Physical dimensions  
- **Millimeters**: Optical parameters
- **Arcseconds**: Angular measurements
- Include unit comments in function signatures

### Logging Requirements
- Every function performing significant computation must include `logger.info()` calls
- Use module-level loggers following established patterns
- Log key parameters, processing steps, and results

### Output Structure
- Hierarchical directories: `{analysis_type}/{parameter_value}/` for parameter sweeps
- Standardized filenames: `centroid_error_vs_*.png`, `*_results.csv`
- Include both pixel and physical units in all outputs

## Key System Specifications

### CMV4000 Sensor Model
- **Resolution**: 2048×2048 pixels
- **Pixel Pitch**: 5.5µm
- **Quantum Efficiency**: ~60%
- **Full Well**: 13,500 e⁻
- **Read Noise**: 13.0 e⁻ RMS

### PSF Simulation Parameters
- **Gen 1**: 128×128 grid, 0.5µm spacing, 64×64µm area
- **Gen 2**: 32×32 grid, 0.232µm spacing
- **Wavelength**: 0.4050-1.0000µm
- **Strehl Ratio**: 0.055

### Detection Parameters
- **Threshold**: 5σ above background (adaptive for diffuse PSFs)
- **Minimum region**: 3 pixels
- **Maximum region**: 50 pixels (original), 24 pixels (FPA)
- **Centroiding**: Moment-based with brightness selection

## Performance Expectations

### Typical Results (Magnitude 3.0 star)
- **Centroiding accuracy**: 0.15-0.25 pixels (0.8-1.4µm)
- **Bearing vector accuracy**: 4-8 arcseconds
- **Detection success rate**: >95% (field angles 0-14°)
- **Processing time**: 2-5 seconds per analysis

### Analysis Scaling
- Single PSF analysis: 2-5 seconds
- Field angle sweep (11 angles): 2-3 minutes
- Memory usage: <100MB base, +33MB per full detector array

## Adding New PSF Generations

1. Create directory: `PSF_sims/Gen_N/`
2. Add PSF files following `{angle}_deg.txt` naming
3. Run interactive scripts - new generation will appear in menus
4. System automatically detects grid size and chooses appropriate projection method

## Multi-Star Development Context

This codebase is currently in Phase 2 development focusing on multi-star analysis capabilities. The `multi_star/` directory contains new development work. When working with multi-star features, maintain compatibility with the existing single-star pipeline architecture.

## Dependencies

Core packages (see requirements.txt):
- numpy>=1.19.0
- scipy>=1.5.0
- matplotlib>=3.3.0
- pandas>=1.1.0
- opencv-python>=4.5.0