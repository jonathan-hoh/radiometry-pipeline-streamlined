# Claude Code Instruction Set: Star Tracker Simulation Validation Framework

## Initialization Phase

**Task 1: Codebase Reconnaissance**
Examine the existing star tracker simulation repository structure. Identify and document:
- Main simulation entry point and pipeline orchestration
- BAST algorithm implementation location
- Camera model/optics module
- Catalog interface (current Hipparcos implementation)
- Attitude solver implementation
- Image generation and rendering modules
- Existing test/validation directories
- Dependencies and requirements files

Output a file `validation_integration_map.md` summarizing current architecture and proposed integration points.

---

## Module Development Phase

**Task 2: Create Validation Framework Structure**
Create the following directory structure:
```
validation/
├── __init__.py
├── attitude_validation.py
├── identification_validation.py
├── astrometric_validation.py
├── photometric_validation.py
├── noise_validation.py
├── monte_carlo.py
├── metrics.py
├── reporting.py
└── config/
    ├── validation_config.yaml
    └── test_scenarios.yaml
```

**Task 3: Implement Core Metrics Module** (`validation/metrics.py`)
Create functions for:
- Quaternion attitude error: `attitude_error_angle(q_true, q_solved)` returning cross-track error in arcsec
- Quaternion component errors: `quaternion_component_errors(q_true, q_solved)`
- Star identification rate: `identification_rate(detected_stars, matched_stars, catalog_stars)`
- Astrometric residuals: `astrometric_residuals(u_catalog, u_simulated)` returning Δx, Δy in pixels
- Centroid RMS: `centroid_rms(residuals)`
- SNR calculations: `calculate_snr(signal_electrons, noise_electrons)`

All functions must include:
- Comprehensive docstrings with equations in numpy-style format
- Type hints
- Input validation
- Unit tests in `validation/tests/test_metrics.py`

**Task 4: Monte Carlo Framework** (`validation/monte_carlo.py`)
Implement class `MonteCarloValidator`:
- `generate_random_attitudes(n_samples)`: uniform quaternion sampling over sphere
- `generate_test_scenarios(n_samples, ra_range, dec_range, roll_range)`: structured attitude grid
- `run_parallel(scenario_list, n_workers)`: parallel execution wrapper using multiprocessing
- `aggregate_results(result_list)`: statistical summary generation

Must support checkpointing to allow resume from interruption.

**Task 5: Attitude Validation Module** (`validation/attitude_validation.py`)
Implement class `AttitudeValidator`:

Methods:
- `__init__(simulation_pipeline, n_monte_carlo=1000)`
- `generate_ground_truth_images(attitudes)`: creates synthetic images with known q_true
- `run_pipeline_on_images(image_list)`: executes BAST + attitude solver
- `compute_error_statistics()`: returns mean, median, std, 95th percentile attitude errors
- `plot_error_distribution(output_path)`: histogram and CDF of attitude errors
- `compare_to_cramér_rao_bound(camera_params, star_mags)`: theoretical lower bound comparison

Integration requirements:
- Must interface with existing simulation pipeline without modifying core modules
- Save intermediate results (ground truth, solved attitudes, timing) to HDF5 format
- Generate plots using matplotlib, save to `validation/results/attitude/`

**Task 6: Star Identification Validation** (`validation/identification_validation.py`)
Implement class `IdentificationValidator`:

Methods:
- `__init__(bast_instance, catalog_interface)`
- `run_identification_sweep(star_densities, magnitude_ranges)`: vary field density and limiting magnitude
- `compute_confusion_matrix(true_ids, solved_ids)`: TP, FP, FN, TN counts
- `identification_rate_vs_density()`: plot η vs. stars in FOV
- `false_positive_analysis()`: characterize spurious matches
- `compare_to_benchmark_algorithms()`: if literature data available for Liebe, Pyramid algorithms

Output detailed CSV with per-image identification metrics.

**Task 7: Astrometric Validation** (`validation/astrometric_validation.py`)
Implement class `AstrometricValidator`:

Methods:
- `project_catalog_stars(catalog_stars, attitude, camera_model)`: ground truth pixel positions
- `compare_to_simulated_centroids(catalog_positions, simulated_positions)`
- `residual_analysis()`: mean, RMS, max residuals in x and y
- `radial_distortion_validation()`: residuals vs. radial distance from optical axis
- `plot_residual_vectors(output_path)`: quiver plot of Δu across FOV

Must verify projection model accuracy to sub-pixel level (<0.1 px RMS for well-modeled optics).

**Task 8: Photometric Validation** (`validation/photometric_validation.py`)
Implement class `PhotometricValidator`:

Methods:
- `magnitude_to_electrons_validation(star_magnitudes, exposure_time, camera_params)`
- `verify_psf_integration(test_stars)`: total integrated electrons vs. theoretical
- `snr_vs_magnitude_curve()`: empirical SNR relationship, compare to $SNR = 10^{-0.4(m_V - m_0)}$
- `limiting_magnitude_determination()`: magnitude where SNR = threshold (typically 5-7)
- `compare_to_detector_specs(vendor_quantum_efficiency)`: if specs available

Output includes SNR curves and magnitude limit determination.

**Task 9: Noise Characterization** (`validation/noise_validation.py`)
Implement class `NoiseValidator`:

Methods:
- `inject_noise_sources(image, read_noise_std, dark_current, shot_noise)`
- `centroid_error_vs_snr()`: empirical scaling, compare to $\sigma_{centroid} = \sigma_{pixel}/(SNR \cdot \sqrt{N_{pixels}})$
- `validate_noise_statistics(noise_realizations)`: verify Gaussian/Poisson characteristics
- `sensitivity_degradation(magnitude_range, noise_levels)`: performance vs. noise

Generate plots showing centroid error scaling and noise impact on identification rate.

**Task 10: Reporting Module** (`validation/reporting.py`)
Implement class `ValidationReporter`:

Methods:
- `generate_summary_report(validation_results_dict)`: aggregate all validation metrics
- `create_presentation_slides(template='saas_pitch')`: auto-generate figures for Felix
- `export_to_latex_table(metrics)`: publication-quality tables
- `save_json_summary(output_path)`: machine-readable results

Must produce:
- Executive summary PDF with key metrics
- Detailed technical report with all plots
- CSV/JSON files for programmatic access

**Task 11: Configuration System** (`validation/config/`)
Create YAML configuration files:

`validation_config.yaml`:
```yaml
monte_carlo:
  n_samples: 1000
  random_seed: 42
  parallel_workers: 8
  
camera:
  fov_deg: 20.0
  pixel_scale_arcsec: 15.0
  image_size: [1024, 1024]
  
validation_thresholds:
  attitude_error_arcsec: 1.0
  identification_rate_min: 0.95
  astrometric_rms_pixels: 0.1
```

`test_scenarios.yaml`: Define specific test cases (dense fields, sparse fields, ecliptic plane, galactic plane, etc.)

---

## Integration Phase

**Task 12: Main Validation Script** (`run_validation.py`)
Create top-level script that:
- Parses command line arguments (--module, --config, --output-dir)
- Loads configuration
- Instantiates validation modules
- Executes validation sequence
- Generates reports
- Returns exit code 0 if all thresholds met, 1 otherwise

**Task 13: CI/CD Integration**
Create `.github/workflows/validation.yml` (if using GitHub) or equivalent:
- Triggers on pull requests to main
- Runs subset of validation tests (reduced n_samples for speed)
- Posts summary comment with key metrics
- Fails if thresholds violated

**Task 14: Documentation**
Create `validation/README.md`:
- Overview of validation framework
- Usage examples for each module
- Interpretation guidelines for metrics
- SaaS pitch talking points derived from validation results

Update main repository README with validation section.

---

## Execution Sequence

**Phase 1**: Tasks 1-2 (reconnaissance and structure)
**Phase 2**: Tasks 3-4 (core utilities)
**Phase 3**: Tasks 5-9 (validation modules, can be parallel)
**Phase 4**: Tasks 10-11 (reporting and configuration)
**Phase 5**: Tasks 12-14 (integration and documentation)

After each task, run existing unit tests to ensure no regressions. Commit with descriptive messages following conventional commits format.

---

## Acceptance Criteria

Validation framework is complete when:
1. All modules pass their unit tests with >95% coverage
2. `run_validation.py --module all` executes without errors
3. Attitude error statistics show σ < 1 arcsec for nominal scenarios
4. Generated reports are presentation-ready
5. Runtime for full validation suite < 30 minutes on 8-core machine
6. All code follows PEP 8, passes flake8 linting

---

## Critical Notes for Claude Code

- Preserve existing simulation code - validation must be additive, not modify core pipeline
- Use existing catalog interface; extend Hipparcos query functions if needed
- Match existing code style and conventions observed in reconnaissance
- All file I/O must use pathlib for cross-platform compatibility
- Plots must be publication-quality: 300 DPI, vector formats where possible, consistent styling
- If existing modules lack necessary interfaces (e.g., attitude solver doesn't return covariance), create wrapper functions rather than modifying source
- Verify units consistently: quaternions, euler angles (which convention?), pixel coordinates (origin location?), catalog coordinates (J2000 epoch?)