# Streamlined Pipeline Organization

This document outlines a lightweight version of the radiometry pipeline codebase that maintains all core functionality while removing debugging and diagnostic files for presentation purposes.

## Current vs. Streamlined Structure

### Streamlined Directory Structure
```
radiometry-pipeline-streamlined/
├── src/                          # Core pipeline source code
│   ├── core/                     # Main pipeline components
│   │   ├── star_tracker_pipeline.py      # Central orchestrator [KEEP]
│   │   ├── starcamera_model.py           # CMV4000 sensor simulation [KEEP]
│   │   ├── psf_plot.py                   # PSF processing [KEEP]
│   │   └── psf_photon_simulation.py      # Poisson noise simulation [KEEP]
│   ├── multi_star/               # Multi-star functionality
│   │   ├── multi_star_pipeline.py        # Multi-star orchestrator [KEEP]
│   │   ├── scene_generator.py            # Scene generation [KEEP]
│   │   ├── multi_star_radiometry.py     # Multi-star radiometry [KEEP]
│   │   ├── attitude_transform.py        # Attitude transformations [KEEP]
│   │   ├── bijective_matching.py        # Optimal star matching [KEEP]
│   │   ├── monte_carlo_quest.py         # Attitude determination [KEEP]
│   │   ├── coordinate_validation.py     # Coordinate validation [KEEP]
│   │   └── peak_detection.py            # Peak detection algorithms [KEEP]
│   └── BAST/                     # BAST algorithms
│       ├── catalog.py                    # Star catalog handling [KEEP]
│       ├── identify.py                   # Star identification [KEEP]
│       ├── match.py                      # Triangle matching [KEEP]
│       ├── resolve.py                    # Attitude resolution [KEEP]
│       ├── calibrate.py                  # Calibration functions [KEEP]
│       ├── preprocess.py                 # Preprocessing [KEEP]
│       └── track.py                      # Tracking algorithms [KEEP]
├── tools/                        # Analysis and utility scripts
│   ├── interactive/              # Main interactive analysis tools
│   │   ├── debug_centroiding.py         # Interactive centroiding demo [KEEP]
│   │   ├── angle_sweep.py               # Field angle analysis [KEEP]
│   │   ├── fpa_diagnostic.py            # FPA diagnostic tool [KEEP]
│   │   └── generate_test_catalogs.py    # Catalog generation [KEEP]
│   ├── specialized/              # Specialized analysis tools
│   │   ├── full_fpa_demo.py             # Complete system demo [KEEP]
│   │   └── focal_length_analysis.py    # Optical analysis [KEEP]
│   └── visualization/            # NEW: Presentation visualizations
│       ├── pipeline_flowchart.py        # System architecture diagram
│       ├── performance_plots.py         # Performance visualization
│       ├── accuracy_analysis.py         # Accuracy vs conditions
│       └── comparison_plots.py          # Algorithm comparison
├── data/                         # Data directories [ALL KEEP]
│   ├── PSF_sims/                 # PSF simulation data
│   │   ├── Gen_1/                # 128×128, 0.5µm/pixel
│   │   └── Gen_2/                # 32×32, 0.232µm/pixel
│   ├── catalogs/                 # Star catalog files
│   │   ├── baseline_5_stars_spread.csv
│   │   └── hipparcos_subset.csv
│   └── examples/                 # Example data and demos
├── docs/                         # Documentation [STREAMLINED]
│   ├── README.md                         # Main documentation [KEEP]
│   ├── ARCHITECTURE.md                   # System architecture [KEEP]
│   ├── presentation/             # NEW: Presentation materials
│   │   ├── pipeline_organization.md     # This document
│   │   ├── outline.md                   # Presentation outline
│   │   ├── figures/              # Generated presentation figures
│   │   └── slides/               # Slide materials
│   └── technical/                # Technical documentation [SELECTIVE]
│       ├── ASSISTANT_CONTEXT.md         # AI development context [REMOVE]
│       ├── MULTI_STAR_ARCHITECTURE_OVERVIEW.md  [KEEP - rename to SYSTEM_OVERVIEW.md]
│       └── CMV4000 Technical Reference.md       [KEEP]
├── config/                       # Configuration files [KEEP]
│   ├── requirements.txt          # Python dependencies
│   └── camera_parameters.yaml    # Camera configuration
└── examples/                     # NEW: Self-contained examples
    ├── single_star_demo.py       # Basic single star analysis
    ├── multi_star_demo.py        # Multi-star scene simulation
    ├── attitude_demo.py          # Attitude determination demo
    └── performance_demo.py       # Performance characterization
```

## Files to Remove (Debugging/Development)

### Debug Tools (tools/debug/)
- `attitude_error_comparison.py` - Development debugging
- `bearing_vector_error_debugger.py` - Specific bug investigation
- `centroid_matching_debugger.py` - Matching algorithm debugging
- `complex_multi_star_visualization.py` - Development validation
- `coordinate_transformation_debugger.py` - Coordinate system debugging
- `debug_bast_matching.py` - BAST algorithm debugging
- `debug_fpa_issue.py` - FPA-specific debugging
- `debug_gen2_fpa.py` - Generation 2 PSF debugging
- `fpa_projection_debugger.py` - Projection debugging
- `simple_coordinate_test.py` - Basic coordinate tests
- `test_attitude_transformation.py` - Attitude testing
- `test_coordinate_fix.py` - Coordinate fix testing

### Development Documentation (docs/technical/)
- `ASSISTANT_CONTEXT.md` - AI development context
- `MULTI_STAR_DEV_DIARY.md` - Development diary
- `FULL_PIPELINE_DEV_DIARY.md` - Development diary
- `MULTI_STAR_HANDOFF.md` - Development handoff
- `MULTI_STAR_PHASE_1_PLAN.md` - Development planning
- `euler_angles.md` - Technical implementation notes
- Various `.backup` files - Development backups

### Development Scripts (Root Level)
- `test_*.py` files - Development testing scripts
- `fix_git_corruption.bat` - Development utility
- `*_checklist.md` - Development checklists
- `*_investigation.md` - Bug investigation notes

## Core Functionality Preserved

### 1. Complete Star Tracker Pipeline
- **Single Star Analysis**: Full PSF → detection → centroiding → bearing vector pipeline
- **Multi-Star Scenes**: Scene generation with realistic star fields and arbitrary attitudes
- **Physical Radiometry**: CMV4000 sensor modeling with Poisson noise and quantum efficiency
- **Attitude Determination**: BAST triangle matching + Monte Carlo QUEST algorithm

### 2. Analysis Capabilities
- **Performance Characterization**: Accuracy vs magnitude, field angle, noise conditions
- **Interactive Demonstrations**: Real-time parameter exploration and visualization
- **Comparison Studies**: Algorithm performance comparison and validation
- **System Validation**: Complete end-to-end pipeline verification

### 3. Data and Configuration
- **PSF Libraries**: Complete optical PSF simulation data (Gen 1 & 2)
- **Star Catalogs**: Hipparcos-based test catalogs with realistic star distributions
- **Camera Models**: CMV4000 specifications and configurable parameters
- **Example Scenarios**: Pre-configured test cases and demonstrations

## Import Path Compatibility

All import statements in the streamlined codebase will remain valid:
```python
# Core pipeline imports
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.core.starcamera_model import CMV4000Model

# Multi-star functionality
from src.multi_star.multi_star_pipeline import MultiStarPipeline
from src.multi_star.monte_carlo_quest import MonteCarloQUEST
from src.multi_star.bijective_matching import bijective_centroid_to_catalog_matching

# BAST algorithms
from src.BAST.catalog import from_csv
from src.BAST.match import match
from src.BAST.resolve import quest_algorithm
```

## New Additions for Presentation

### 1. Visualization Tools (tools/visualization/)
Purpose-built scripts for creating presentation-quality figures:
- **System Architecture Diagrams**: Pipeline flowcharts and data flow
- **Performance Visualizations**: Accuracy curves, noise analysis, field-of-view plots
- **Comparison Studies**: Algorithm performance side-by-side
- **Physical Realism Demonstrations**: PSF evolution, detector effects, radiometry

### 2. Self-Contained Examples (examples/)
Complete demonstration scripts that can be run independently:
- **Single Star Demo**: Basic functionality demonstration (5-minute runtime)
- **Multi-Star Demo**: Advanced capabilities showcase (10-minute runtime)
- **Attitude Demo**: QUEST algorithm and attitude determination
- **Performance Demo**: System characterization and validation

### 3. Presentation Materials (docs/presentation/)
- **Organized Documentation**: Architecture overview, system capabilities
- **Generated Figures**: Publication-quality plots and diagrams
- **Slide Templates**: Ready-to-use presentation materials

## File Count Reduction

- **Current Codebase**: ~150+ files including debugging, development logs, and temporary files
- **Streamlined Codebase**: ~50 core files + data + presentation materials
- **Functionality**: 100% preservation of core capabilities
- **Presentation Ready**: All remaining files serve the core mission or presentation needs

## Usage Compatibility

All existing workflows remain functional:
```bash
# Core functionality preserved
PYTHONPATH=. python tools/interactive/debug_centroiding.py
PYTHONPATH=. python tools/specialized/full_fpa_demo.py data/PSF_sims/Gen_1/0_deg.txt

# New presentation examples
PYTHONPATH=. python examples/single_star_demo.py
PYTHONPATH=. python examples/multi_star_demo.py
```

This streamlined structure maintains the sophisticated technical capabilities while presenting a clean, professional codebase suitable for engineering stakeholder presentation and potential productization.