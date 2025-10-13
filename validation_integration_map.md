# Validation Integration Map
## Star Tracker Radiometry Pipeline - Validation Framework Architecture

### Current Codebase Architecture

#### Main Simulation Entry Point
- **StarTrackerPipeline** (`src/core/star_tracker_pipeline.py`): Central orchestrator class
  - Handles PSF loading, radiometric simulation, centroiding, and bearing vector calculation
  - Supports both original PSF analysis (128×128, 0.5µm/pixel) and FPA projection (11×11, 5.5µm/pixel)
  - Key methods: `process_psf()`, `calculate_bearing_vector()`, `generate_comprehensive_plots()`

#### BAST Algorithm Implementation
- **Star Catalog Interface** (`src/BAST/catalog.py`): Hipparcos catalog management with triplet generation
- **Pattern Matching** (`src/BAST/match.py`): Triangle matching algorithms  
- **Attitude Solver** (`src/BAST/resolve.py`): QUEST algorithm implementation with Davenport matrix
- **Star Identification** (`src/BAST/identify.py`): Centroiding and pixel grouping using OpenCV
- **Preprocessing** (`src/BAST/preprocess.py`): Image preprocessing and noise handling

#### Camera/Optics Module
- **StarCamera Model** (`src/core/starcamera_model.py`): CMV4000 sensor simulation
  - Specifications: 2048×2048 pixels, 5.5µm pitch, 13.5ke⁻ full well, 13e⁻ RMS read noise
  - Radiometric chain: magnitude → photon count → detector electrons → digital counts
- **PSF Processing** (`src/core/psf_plot.py`): Zemax-derived PSF parsing and metadata extraction
- **Photon Simulation** (`src/core/psf_photon_simulation.py`): Poisson noise and detector response

#### Image Generation and Rendering
- **Multi-Star Pipeline** (`src/multi_star/multi_star_pipeline.py`): Multi-star scene orchestration
- **Scene Generator** (`src/multi_star/scene_generator.py`): Realistic star field generation
- **Multi-Star Radiometry** (`src/multi_star/multi_star_radiometry.py`): Multi-source photon simulation

#### Attitude Solver Implementation
- **Monte Carlo QUEST** (`src/multi_star/monte_carlo_quest.py`): Statistical attitude determination
- **Attitude Transform** (`src/multi_star/attitude_transform.py`): Quaternion/Euler transformations
- **Bijective Matching** (`src/multi_star/bijective_matching.py`): Hungarian algorithm for optimal star assignment

#### Existing Test/Validation
- **Basic Validation** (`src/multi_star/validation.py`): Placeholder validation functions (needs implementation)
- **Perturbation Test** (`tools/specialized/test_perturbation_analysis.py`): Environmental effects testing
- **Example Catalogs** (`data/catalogs/`): Test star field configurations
- **Demo Scripts** (`examples/`): Single and multi-star demonstration code

#### Dependencies and Requirements
- **Core Requirements** (`config/requirements.txt`): numpy, scipy, matplotlib, pandas, opencv-python, astropy, numba
- **PSF Data** (`data/PSF_sims/`): 6 generations of Zemax-derived PSFs
- **Analysis Tools** (`tools/`): Interactive, specialized, and visualization utilities

### Proposed Validation Integration Points

#### 1. Non-Invasive Integration Strategy
- Create `validation/` directory alongside existing `src/`, `tools/`, `data/` structure
- Use **StarTrackerPipeline** as primary interface - do not modify core modules
- Leverage existing `monte_carlo_quest.py` for attitude statistics
- Extend `src/multi_star/validation.py` rather than replacing

#### 2. Interface Compatibility
- **Pipeline Interface**: All validation modules interface through `StarTrackerPipeline` class
- **Data Access**: Use existing PSF loading (`parse_psf_file`) and catalog interface (`src/BAST/catalog.py`)
- **Results Format**: Maintain existing output structure (`tools/` generates to `outputs/`)
- **Configuration**: Extend existing parameter handling patterns

#### 3. Module Integration Points

**Attitude Validation:**
- Interface: `StarTrackerPipeline.process_psf()` → `MonteCarloQUEST.determine_attitude()`
- Ground Truth: Use existing `attitude_transform.py` for quaternion generation
- Error Metrics: Leverage existing quaternion math in `src/multi_star/`

**Identification Validation:**
- Interface: `src/BAST/identify.py` centroiding + `src/BAST/match.py` triangle matching
- Catalog Interface: Extend `src/BAST/catalog.py` triplet generation
- Metrics: Build on existing `StarMatch` class from `src/BAST/match.py`

**Astrometric Validation:**
- Interface: `StarTrackerPipeline.calculate_bearing_vector()` 
- Camera Model: Use existing `src/core/starcamera_model.py` projection functions
- PSF Integration: Leverage `src/core/psf_photon_simulation.py` for ground truth

**Photometric Validation:**
- Interface: `calculate_optical_signal()` from `src/core/starcamera_model.py`
- Noise Model: Use existing Poisson simulation in `src/core/psf_photon_simulation.py`
- SNR Calculation: Build on existing radiometric chain

#### 4. Data Flow Integration
```
Existing: PSF Files → StarTrackerPipeline → Centroiding → Bearing Vectors
New:      PSF Files → Validation Framework → StarTrackerPipeline → Statistics → Reports
```

#### 5. Output Integration
- **Existing Pattern**: `tools/specialized/` → `outputs/{analysis_type}/`
- **Validation Pattern**: `validation/` → `validation/results/{module}/`
- **Report Generation**: Extend existing matplotlib plotting patterns
- **Data Export**: Follow existing CSV/HDF5 output conventions

#### 6. Configuration Integration
- **Current**: Hardcoded parameters in individual scripts
- **Proposed**: YAML configuration extending existing parameter patterns
- **Backward Compatibility**: Maintain existing command-line interfaces in `tools/`

### Critical Constraints

1. **No Core Module Modification**: Validation must be additive, not modify `src/core/` or `src/BAST/`
2. **Performance Preservation**: Maintain existing sub-second analysis capability  
3. **Interface Stability**: Do not break existing `tools/` or `examples/` imports
4. **Unit Consistency**: Follow existing physical unit patterns (pixels, µm, mm, arcsec)
5. **Logging Compatibility**: Extend existing logging patterns from core modules

### Implementation Sequence Dependencies

1. **Core Utilities First**: `metrics.py` and `monte_carlo.py` have no dependencies
2. **Pipeline Integration**: Attitude validation depends on `MonteCarloQUEST` interface
3. **BAST Integration**: Identification validation requires understanding existing match patterns
4. **Reporting Last**: All validation modules must be complete before aggregation

### Risk Mitigation

1. **Interface Stability**: Use composition over inheritance to avoid breaking changes
2. **Performance Impact**: Validation runs independently of production pipeline
3. **Data Compatibility**: Leverage existing PSF and catalog formats without modification
4. **Testing Strategy**: Each validation module includes unit tests that verify non-regression

This integration map provides the foundation for implementing the comprehensive validation framework while preserving the existing codebase architecture and performance characteristics.