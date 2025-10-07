# AI System Instructions for Star Tracker Radiometry Pipeline

## Project Overview

You are assisting with a sophisticated **star tracker radiometry simulation pipeline** - a complete digital twin of a BAST (Basic Angle Star Tracker) based star tracker system. This is production-ready engineering software used for spacecraft attitude determination system design and validation.

## Core Mission

This simulation enables **quantitative performance prediction** for star tracker systems before hardware development, providing:
- Physics-based radiometric modeling from photons to attitude solutions
- Algorithm validation and optimization
- Design trade-off analysis (sensors, optics, algorithms)
- Risk reduction for multi-million dollar spacecraft projects

## Technical Architecture

### System Flow
```
Stellar Scene → Optical PSF → CMV4000 Sensor → Star Detection → 
Centroiding → Bearing Vectors → Triangle Matching → Attitude Determination
```

### Key Components

**1. Physical Simulation Layer**
- **Optical PSFs**: Zemax-derived point spread functions with realistic aberrations
- **CMV4000 Sensor Model**: 2048×2048, 5.5µm pixels, quantum efficiency, read noise
- **Radiometric Chain**: Star magnitude → photon count → detector electrons → digital counts
- **Poisson Noise**: Realistic photon shot noise and detector characteristics

**2. Detection and Measurement**
- **Adaptive Thresholding**: Multi-scale background estimation with connected components
- **Sub-pixel Centroiding**: Moment-based algorithms achieving 0.15-0.25 pixel accuracy
- **Bearing Vector Calculation**: Pinhole camera model with focal length normalization
- **Multi-star Scene Generation**: Realistic star fields with arbitrary spacecraft attitudes

**3. Pattern Recognition and Attitude**
- **BAST Triangle Matching**: Geometric pattern recognition using inter-star angles
- **Bijective Matching**: Optimal one-to-one star assignment (Hungarian algorithm)
- **Monte Carlo QUEST**: Statistical attitude determination with uncertainty quantification
- **SVD-based Wahba Solution**: Optimal rotation matrix estimation

## Performance Characteristics

### Validated Capabilities
- **Centroiding Accuracy**: 0.8-1.4 µm (0.15-0.25 pixels on CMV4000)
- **Bearing Vector Accuracy**: 4-8 arcseconds typical performance
- **Attitude Accuracy**: 1-5 arcseconds with 3+ matched stars
- **Detection Success**: >95% for magnitude 3-6 stars (0-14° field angles)
- **Processing Speed**: Sub-second for most analysis scenarios

### System Specifications
- **Sensor**: CMV4000 (2048×2048, 5.5µm pitch, 13.5ke⁻ full well, 13e⁻ RMS read noise)
- **Optics**: 40.07mm focal length, F/# configurable
- **PSF Data**: Gen_1 (128×128, 0.5µm/pixel), Gen_2 (32×32, 0.232µm/pixel)
- **Wavelength Range**: 0.405-1.0 µm (visible/near-IR)
- **Star Catalogs**: Hipparcos-based with magnitude and position data

## Code Structure

### Core Modules (`src/`)
```
src/core/
├── star_tracker_pipeline.py     # Central orchestrator (CRITICAL)
├── starcamera_model.py          # CMV4000 physical modeling
├── psf_plot.py                  # PSF processing and metadata
└── psf_photon_simulation.py     # Poisson noise and radiometry

src/multi_star/
├── multi_star_pipeline.py       # Multi-star scene orchestrator
├── scene_generator.py           # Star field generation
├── attitude_transform.py        # Quaternion/Euler transformations
├── monte_carlo_quest.py         # Statistical attitude determination
└── bijective_matching.py        # Optimal star association

src/BAST/
├── catalog.py                   # Star catalog management
├── match.py                     # Triangle pattern matching
├── resolve.py                   # QUEST attitude algorithm
└── identify.py                  # Star identification logic
```

### Analysis Tools (`tools/`)
- **Interactive**: Real-time parameter exploration and visualization
- **Specialized**: Complete system demonstrations and validation
- **Visualization**: Presentation-quality figure generation (expandable)

### Data Assets (`data/`)
- **PSF_sims/**: Zemax-derived optical point spread functions
- **catalogs/**: Star position and magnitude databases
- **examples/**: Demonstration scenarios and test cases

## Mathematical Foundations

### Coordinate Systems
- **Inertial Frame**: Right-handed celestial coordinate system (RA/Dec)
- **Camera Frame**: Z-axis = boresight, X/Y in focal plane
- **Detector Frame**: Pixel coordinates with (0,0) at corner
- **Transformations**: Quaternions for attitude, DCM for vector rotations

### Key Algorithms
- **Centroiding**: `Σ(I(x,y) * [x,y]) / Σ(I(x,y))` (intensity-weighted moments)
- **Bearing Vectors**: `v = [u, v, f] / ||[u, v, f]||` (pinhole model)
- **Triangle Matching**: Angular distance preservation in 3D
- **QUEST**: `max(trace(λI - K))` where K is attitude profile matrix
- **SVD Solution**: `R = UV^T` for optimal rotation matrix

### Physical Models
- **Photon Flux**: `N = (F₀ * 10^(-0.4*mag) * A * Δt * QE)` 
- **Poisson Statistics**: `P(n) = λⁿe^(-λ)/n!` for shot noise
- **Detector Response**: `DN = (N_e + N_read) * gain + offset`
- **PSF Projection**: Optical → detector coordinate mapping with realistic sampling

## Engineering Applications

### Design Optimization
- **Sensor Trade Studies**: Pixel pitch, quantum efficiency, noise analysis
- **Optical Design**: Focal length vs accuracy/field-of-view optimization
- **Algorithm Comparison**: Detection thresholds, matching tolerances
- **Performance Prediction**: Mission-specific accuracy requirements

### Validation and Verification
- **Algorithm Testing**: Verify BAST matching rates and convergence
- **Error Budget Analysis**: End-to-end uncertainty propagation
- **Monte Carlo Validation**: Statistical performance characterization
- **Edge Case Analysis**: Low SNR, field edges, sparse star fields

## Critical Context for AI Assistance

### Code Philosophy
- **StarTrackerPipeline is Central**: All analysis flows through this orchestrator class
- **Physical Realism**: Never simplify to pure geometry - maintain radiometric accuracy
- **Modular Design**: Components can be swapped/upgraded while maintaining interfaces
- **Performance Focus**: Sub-second analysis for interactive use

### Development Principles
- **Unit Preservation**: Always specify pixels, µm, mm, arcseconds in comments
- **Logging Required**: All major functions must include informational logging
- **Validation First**: New algorithms must be validated against literature/ground truth
- **Backward Compatibility**: Maintain existing import paths and interfaces

### Mathematical Rigor
- **Proper Statistics**: Use Monte Carlo for uncertainty, avoid single-point estimates
- **Coordinate Consistency**: Maintain right-handed systems, proper transformations
- **Physical Constraints**: Unit vectors, positive photon counts, realistic noise levels
- **Numerical Stability**: Check matrix conditioning, handle edge cases gracefully

### Presentation Context
- **Audience**: Engineering managers and spacecraft system designers (not software developers)
- **Value Proposition**: Risk reduction and cost savings through simulation-first design
- **Technical Credibility**: Emphasize physics-based modeling and validated algorithms
- **Professional Maturity**: Production-ready tools, not research prototypes

## Common Request Categories

### Mathematical Documentation
- Derive governing equations for radiometric chain
- Document coordinate transformation mathematics
- Explain QUEST algorithm theoretical foundation
- Describe uncertainty propagation methodology

### Code Enhancement
- Implement additional sensor models (beyond CMV4000)
- Add environmental effects (temperature, radiation aging)
- Create advanced visualization tools
- Optimize computational performance

### Presentation Support
- Generate executive summary slides
- Create technical deep-dive presentations
- Develop performance comparison charts
- Design system architecture diagrams

### Validation and Testing
- Create comprehensive test suites
- Implement algorithm benchmarking
- Design validation against hardware data
- Develop regression testing framework

## Key Success Metrics

When helping with this project, prioritize:
1. **Technical Accuracy**: Physics and mathematics must be correct
2. **Engineering Utility**: Solutions must solve real spacecraft design problems
3. **Professional Quality**: Code and documentation suitable for stakeholder review
4. **Performance**: Maintain sub-second analysis capability
5. **Maintainability**: Clear, documented, modular implementation

## Critical Files to Reference

- **docs/white_paper/White_Paper_working-official.md**: The official white paper that comprehensively walks through the mathematics behind the code logic; AI assistants should familiarize themselves with this paper before offering assistance with the codebase
- **CLAUDE.md**: Development context and system architecture
- **README.md**: Quick start and capabilities overview
- **pipeline_organization.md**: Codebase structure rationale
- **outline.md**: Presentation strategy and key messages
- **STREAMLINED_STATUS.md**: Current system status and validation checklist

## Usage Patterns

Always assume:
- User runs commands with `PYTHONPATH=.` 
- Import paths start with `src.` (e.g., `from src.core.star_tracker_pipeline import StarTrackerPipeline`)
- Data files accessed via relative paths from project root
- Analysis outputs should include both numerical results and visualization

This is a sophisticated engineering simulation system representing years of development effort. Approach all assistance with the rigor and professionalism appropriate for spacecraft flight software development.