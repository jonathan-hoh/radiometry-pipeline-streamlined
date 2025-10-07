# Star Tracker Presentation Visualization Scripts

This directory contains Python scripts that generate all figures needed for the star tracker simulation pipeline presentation. The scripts create professional-quality visualizations demonstrating the technical sophistication and practical utility of the simulation system.

## Quick Start

To generate all figures at once:

```bash
# From the root project directory
cd docs/presentation/visualization_scripts
PYTHONPATH=. python generate_all_figures.py
```

Individual scripts can be run separately:

```bash
PYTHONPATH=. python system_architecture.py
PYTHONPATH=. python performance_characterization.py
# ... etc
```

## Generated Figures

All figures are saved to `docs/presentation/figures/` in both PNG and PDF formats:
- **PNG files**: Optimized for presentations (PowerPoint, Google Slides)
- **PDF files**: Vector format for publications and high-quality prints

## Script Organization

### 1. `system_architecture.py`
**Purpose**: Show complete pipeline and physical realism architecture

**Generated Figures**:
- `pipeline_flowchart.png/pdf` - End-to-end data flow with physical processes
- `multilevel_architecture.png/pdf` - Optical → Electronic → Computational layers  
- `data_flow_diagram.png/pdf` - PSF files → sensor simulation → algorithm processing

### 2. `performance_characterization.py`
**Purpose**: Quantitative performance demonstration across operational parameters

**Generated Figures**:
- `centroiding_vs_magnitude.png/pdf` - Sub-pixel accuracy vs star brightness
- `bearing_vector_vs_field_angle.png/pdf` - Field-of-view performance effects
- `detection_success_rate.png/pdf` - Operational envelope boundaries
- `attitude_accuracy_vs_star_count.png/pdf` - Redundancy benefits
- `performance_summary_table.png/pdf` - Specification compliance table

### 3. `physical_realism.py`
**Purpose**: Demonstrate sophistication beyond simple geometric models

**Generated Figures**:
- `psf_evolution.png/pdf` - Realistic optical aberration effects across field
- `detector_response_comparison.png/pdf` - Photon noise vs clean geometric centroids
- `multi_star_scene.png/pdf` - Realistic detector images with multiple PSFs
- `monte_carlo_error_propagation.png/pdf` - Statistical uncertainty bounds

### 4. `algorithm_validation.py`
**Purpose**: Demonstrate technical credibility and algorithm accuracy

**Generated Figures**:
- `bast_triangle_matching.png/pdf` - Triangle matching performance and geometric accuracy
- `quest_convergence.png/pdf` - Attitude algorithm convergence analysis
- `end_to_end_validation.png/pdf` - Ground truth comparison results
- `algorithm_validation_summary.png/pdf` - Literature comparison table

### 5. `engineering_applications.py`
**Purpose**: Show practical utility for design decisions and optimization

**Generated Figures**:
- `sensor_trade_study.png/pdf` - CMV4000 vs alternatives comparison
- `focal_length_optimization.png/pdf` - Accuracy vs field-of-view trade-offs
- `operational_envelope.png/pdf` - Performance boundaries and design margins
- `requirements_verification.png/pdf` - Specification compliance analysis

### 6. `development_timeline.py`
**Purpose**: Show professional development process and current status

**Generated Figures**:
- `development_timeline.png/pdf` - Phased development approach with milestones
- `validation_methodology.png/pdf` - Systematic verification process flowchart
- `capabilities_matrix.png/pdf` - Implementation status by feature category
- `performance_benchmarks.png/pdf` - Quantitative development progress

## Dependencies

Required Python packages:
- `matplotlib >= 3.3.0` - Plotting and visualization
- `numpy >= 1.19.0` - Numerical computations
- `pandas >= 1.1.0` - Data handling (optional, used for some analyses)
- `scipy >= 1.5.0` - Scientific computing functions

Install dependencies:
```bash
pip install -r ../../config/requirements.txt
```

## Design Philosophy

### Visual Consistency
- **Color Schemes**: Consistent color coding across related figures
- **Typography**: Professional fonts and sizing for presentation clarity
- **Layout**: Standardized subplot arrangements and spacing

### Technical Accuracy
- **Data-Driven**: All performance curves based on realistic simulation parameters
- **Literature Validation**: Algorithm comparisons reference established benchmarks
- **Physical Realism**: PSF and sensor models use actual hardware specifications

### Presentation Ready
- **High Resolution**: 300 DPI for crisp presentation and print quality
- **Format Flexibility**: Both raster (PNG) and vector (PDF) formats provided
- **Clear Annotations**: Self-explanatory labels and legends for standalone use

## Customization

### Modifying Performance Parameters
Edit the `generate_performance_data()` functions in each script to adjust:
- Star magnitude ranges
- Field angle coverage
- Noise levels and sensor specifications
- Algorithm performance assumptions

### Styling Changes
Common styling variables at the top of each script:
- `output_dir`: Change figure save location
- Color palettes: Modify `colors` dictionaries
- Figure sizes: Adjust `figsize` parameters
- Font sizes: Modify `fontsize` parameters throughout

### Adding New Figures
1. Create new plotting functions following the established pattern:
   ```python
   def plot_new_analysis():
       fig, ax = plt.subplots(figsize=(12, 8))
       # ... plotting code ...
       plt.savefig(output_dir / 'new_analysis.png', dpi=300, bbox_inches='tight')
       plt.savefig(output_dir / 'new_analysis.pdf', bbox_inches='tight')
   ```
2. Add function call to `main()`
3. Update this README with figure description

## Troubleshooting

### Common Issues

**Import Errors**: Ensure `PYTHONPATH=.` is set when running scripts
```bash
PYTHONPATH=. python script_name.py
```

**Missing Figures**: Check console output for error messages. Most common causes:
- Missing dependencies
- Insufficient permissions for output directory
- Memory issues with large datasets

**Poor Quality Output**: Verify DPI settings (should be 300) and use vector PDF format for scaling

### Performance Optimization

For faster generation:
- Reduce Monte Carlo trial counts in `physical_realism.py`
- Decrease grid resolution in contour plots
- Comment out less critical subplots during development

### Memory Usage

Large figure generation may require:
- 4GB+ RAM for full Monte Carlo simulations
- Reduce data point density for memory-constrained systems
- Generate figures individually rather than all at once

## Integration with Presentation

### Slide Mapping
Figures are designed to support the presentation outline structure:
1. **Problem Statement** → Use architecture diagrams to show complexity
2. **Technical Capabilities** → Physical realism demonstrations
3. **Performance Demo** → Performance characterization plots
4. **Engineering Applications** → Trade study and requirements figures
5. **Technical Sophistication** → Algorithm validation and development timeline

### File Naming Convention
All files use descriptive names matching presentation content:
- `system_*` → Architecture and design overview
- `performance_*` → Quantitative results and analysis
- `validation_*` → Algorithm credibility and testing
- `engineering_*` → Practical applications and trade studies

## Maintenance

### Regular Updates
- Update performance data as simulation capabilities improve
- Refresh timeline figures to reflect current development status
- Add new validation results as they become available

### Version Control
- Commit both scripts and generated figures
- Use meaningful commit messages describing figure updates
- Tag releases corresponding to presentation milestones

## Support

For questions or issues with the visualization scripts:
1. Check this README for common solutions
2. Review console output for specific error messages
3. Verify all dependencies are installed and up to date
4. Ensure proper PYTHONPATH configuration

The scripts are designed to be self-contained and should work on any system with the required Python packages installed.