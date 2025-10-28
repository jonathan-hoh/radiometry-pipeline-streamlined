# Star Tracker GUI User Guide

Welcome to the Star Tracker Simulation GUI! This comprehensive desktop application provides an intuitive interface for configuring, running, and analyzing star tracker simulations.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages: `PyQt6`, `matplotlib`, `numpy`

### Installation
```bash
# Install dependencies
pip install PyQt6 matplotlib numpy

# Launch the GUI
python run_gui.py
```

### First Run
1. The GUI will automatically check dependencies and data files
2. If successful, you'll see: "✅ All dependencies available"
3. You're ready to start simulating!

---

## 🎯 Main Interface Overview

The GUI offers two interface modes:

### **Simple Interface** (Phase 1-2)
- Basic form with essential parameters
- Perfect for quick simulations
- Integrated progress tracking

### **Advanced Wizard Interface** (Phase 3-4) ⭐ *Recommended*
- Professional 3-tab wizard
- Comprehensive parameter control
- Real-time validation and feedback
- Advanced results visualization

*The GUI defaults to the advanced interface for the best experience.*

---

## 📋 Configuration Wizard (Advanced Interface)

### **Tab 1: Sensor Configuration**

Configure your star tracker's sensor characteristics:

| Parameter | Range | Description | Default |
|-----------|--------|-------------|---------|
| **Pixel Pitch** | 3-10 µm | Physical size of detector pixels | 5.6 µm |
| **Resolution** | 512²-2048² | Detector array dimensions | 2048×2048 |
| **Quantum Efficiency** | 0-100% | Photon-to-electron conversion rate | 60% |
| **Read Noise** | 0-50 e⁻ | Electronic noise during readout | 13 e⁻ |
| **Dark Current** | 0-200 e⁻/s | Thermal noise generation rate | 50 e⁻/s |

**💡 Tips:**
- Smaller pixels = better resolution, but potentially lower sensitivity
- Higher quantum efficiency = better faint star detection
- Lower read noise = improved performance for faint stars

### **Tab 2: Optics Configuration**

Configure your optical system:

| Parameter | Options | Description | Default |
|-----------|---------|-------------|---------|
| **Focal Length** | 10-100 mm | Distance from lens to detector | 35 mm |
| **Aperture** | f/1.2 to f/5.6 | f-number (focal length ÷ diameter) | f/1.4 |
| **Field of View** | *Auto-calculated* | Angular coverage of the detector | ~25° |
| **Distortion** | None/Minimal/Moderate | Optical distortion level | Minimal |

**💡 Tips:**
- Longer focal length = narrower FOV, better angular resolution
- Lower f-number = more light gathering, potential aberrations
- FOV updates automatically when you change sensor/optics parameters

### **Tab 3: Scenario Configuration**

Configure your simulation scenario:

| Parameter | Options | Description | Default |
|-----------|---------|-------------|---------|
| **Star Catalog** | Hipparcos/Gaia DR3 | Source star database | Hipparcos |
| **Magnitude Limit** | 0-10 | Faintest stars to include | 6.5 |
| **Attitude Profile** | 6 presets | Spacecraft motion pattern | Inertial (Fixed) |
| **Environment** | Deep Space/LEO/GEO | Operating environment | Deep Space |
| **Monte Carlo Trials** | 100-5000 | Number of simulation runs | 1000 |
| **PSF File** | *Auto-detected* | Point spread function data | 0_deg.txt |

**💡 Tips:**
- Higher magnitude limit = more stars, longer simulation time
- Complex environments (LEO/GEO) require more trials for good statistics
- Runtime estimate updates based on your selections

---

## ⚡ Running Simulations

### **Validation System**
- Each tab shows ✅/❌ indicators for parameter validity
- Progress bar shows configuration completeness (X of 3 sections)
- "Run Simulation" button enables only when all tabs are valid

### **Execution Process**
1. **Configure**: Set parameters across all 3 tabs
2. **Validate**: Ensure all parameters show ✅ indicators
3. **Run**: Click "🚀 Run Simulation" button
4. **Monitor**: Watch real-time progress in the right panel
5. **Analyze**: Results window opens automatically when complete

### **Progress Monitoring**
The right panel shows live updates during simulation:
- **Progress Bar**: Overall completion percentage
- **Live Metrics**: 
  - Trials completed (X/Y format)
  - Elapsed time
  - Estimated time remaining
  - Average time per trial
- **Detailed Log**: Timestamped execution messages
- **Cancel Option**: Stop simulation if needed

---

## 📊 Results Analysis

When simulation completes, a professional **Results Window** opens with three panels:

### **Left Panel: Statistics Analysis**

**🎯 Centroiding Performance**
- Mean Error, Standard Deviation, 95th Percentile, Maximum Error
- ✅ Pass indicator if within ±0.3 pixel criteria

**🎯 Attitude Performance** 
- Mean Error, Standard Deviation, 3σ Bound, Maximum Error
- ✅ Pass indicator if 3σ bound < 10 arcseconds

**🎯 Star Matching**
- Success Rate, Average Stars Detected/Matched, Match Time
- ✅ Pass indicator if success rate ≥ 95%

**🎯 Overall Performance**
- Trial statistics, execution time
- **Overall Score**: 0-100% weighted performance metric

### **Center Panel: Interactive Visualizations**

**📈 Four Plot Types:**
1. **Centroid Error**: Histogram with statistical overlays
2. **Attitude Error**: Distribution with requirement lines
3. **Star Field**: Catalog vs detected star positions
4. **Residuals**: 2D centroiding error patterns

**🛠️ Plot Controls:**
- **Interactive Tools**: Zoom, pan, save (matplotlib toolbar)
- **Customization**: Bin count, grid toggle
- **Refresh**: Update plots with current settings

### **Right Panel: Export & Sharing**

**📁 Export Options:**
- ✅ **Summary Report** (TXT): Comprehensive analysis document
- ✅ **All Plots** (PNG): High-resolution visualization images  
- ✅ **Raw Data** (CSV): Structured data tables for analysis
- ✅ **Configuration** (JSON): Simulation parameters and metadata

**⚙️ Export Settings:**
- **Plot DPI**: 72-600 DPI for image quality
- **Filename Prefix**: Customizable naming convention
- **Directory Selection**: Choose export location

**🚀 Quick Actions:**
- **💾 Quick Save**: One-click save to `outputs/` directory
- **📁 Export Selected**: Custom export with progress tracking

---

## ⌨️ Keyboard Shortcuts

### **Main Window**
- `Ctrl+Q`: Exit application
- `F1`: Show help/about dialog

### **Results Window**
- `Ctrl+S`: Quick save results
- `Ctrl+W`: Close results window
- `F5`: Refresh all plots

---

## 📁 File Organization

### **Input Files**
```
data/
├── PSF_sims/Gen_1/          # Point spread function data
│   ├── 0_deg.txt            # On-axis PSF
│   ├── 5_deg.txt            # 5-degree field angle
│   └── ...                  # Other field angles
└── catalogs/                # Star catalog data (if used)
```

### **Output Files**
```
outputs/
└── simulation_results_YYYYMMDD_HHMMSS/
    ├── simulation_summary.txt          # Comprehensive report
    ├── simulation_config.json          # Configuration parameters
    ├── plots/                          # Visualization images
    │   ├── centroid_histogram.png
    │   ├── attitude_histogram.png
    │   ├── star_field.png
    │   └── residuals.png
    └── data/                           # Raw data tables
        ├── centroid_errors.csv
        ├── attitude_errors.csv
        └── summary_statistics.csv
```

---

## 🔧 Troubleshooting

### **Common Issues**

**❌ "StarTrackerPipeline not available"**
- **Solution**: Run `PYTHONPATH=. python run_gui.py` to set Python path

**❌ "PSF data directory not found"**
- **Solution**: Ensure `data/PSF_sims/Gen_1/` exists with PSF files

**❌ "No PSF files found"**
- **Solution**: Check that PSF files (*.txt) are in the PSF directory

**❌ Plots not displaying**
- **Solution**: Install matplotlib: `pip install matplotlib`

**⚠️ Simulation runs slowly**
- **Solution**: Reduce trial count or use simpler environment settings

**❌ Export fails**
- **Solution**: Ensure target directory exists and has write permissions

### **Performance Tips**

**For Faster Simulations:**
- Start with fewer trials (100-500) for testing
- Use "Deep Space" environment (simplest)
- Choose "Inertial (Fixed)" attitude profile

**For Better Results:**
- Use 1000+ trials for publication-quality statistics
- Include environmental complexity for realistic scenarios
- Validate configuration thoroughly before running

### **Getting Help**

1. **Built-in Help**: Click menu → Help → About for version info
2. **Tooltips**: Hover over any parameter for detailed explanations
3. **Validation Messages**: Check ⚠️ indicators for specific issues
4. **Log Files**: Check `gui_application.log` for detailed error information

---

## 🎓 Usage Examples

### **Example 1: Quick Performance Check**
1. **Goal**: Fast evaluation of default configuration
2. **Settings**: Keep defaults, set trials to 100
3. **Expected Time**: ~15 seconds
4. **Use Case**: Initial system validation

### **Example 2: High-Precision Analysis**
1. **Goal**: Publication-quality performance analysis
2. **Settings**: Optimize all parameters, 2000+ trials
3. **Expected Time**: 5-10 minutes
4. **Use Case**: Research paper results

### **Example 3: Parameter Sensitivity Study**
1. **Goal**: Understanding pixel pitch impact
2. **Method**: Run multiple simulations with different pixel pitches
3. **Analysis**: Compare results across multiple results windows
4. **Use Case**: Design optimization

### **Example 4: Environmental Comparison**
1. **Goal**: Deep Space vs LEO performance
2. **Method**: Run identical configs with different environments
3. **Analysis**: Use export data for quantitative comparison
4. **Use Case**: Mission planning

---

## 🔮 Advanced Features

### **Multiple Results Windows**
- Open multiple results simultaneously for comparison
- Each window has unique job ID and timestamp
- Independent analysis and export for each result set

### **Real-Time Validation**
- Parameters validate as you type
- Cross-dependencies automatically update (e.g., FOV recalculation)
- Color-coded feedback with specific error messages

### **Professional Export**
- Threaded export operations don't block the interface
- Timestamped directories prevent data overwrites
- Multiple formats support different analysis workflows

### **Error Recovery**
- Graceful handling of missing files or dependencies
- Simulation cancellation with proper cleanup
- Automatic retry suggestions for common issues

---

## 🏆 Best Practices

### **Configuration Strategy**
1. **Start Simple**: Use defaults for initial familiarization
2. **Understand Dependencies**: Learn how parameters interact
3. **Validate Thoroughly**: Ensure all tabs show ✅ before running
4. **Document Changes**: Use meaningful export prefixes for organization

### **Simulation Management**
1. **Test Runs**: Start with fewer trials for parameter exploration
2. **Production Runs**: Use higher trial counts for final analysis
3. **Monitor Progress**: Watch live metrics for early problem detection
4. **Save Everything**: Export results immediately after completion

### **Results Analysis**
1. **Check Statistics**: Review pass/fail indicators for requirements compliance
2. **Examine Plots**: Look for patterns or anomalies in visualizations
3. **Export Data**: Save raw data for external analysis if needed
4. **Compare Results**: Use multiple windows for parameter studies

---

## 📚 Technical Notes

### **Simulation Algorithm**
The GUI interfaces with the existing `StarTrackerPipeline` from `src/core/star_tracker_pipeline.py`, providing a user-friendly front-end to the sophisticated radiometric simulation engine.

### **Threading Architecture**
- Main GUI thread handles user interface
- Background worker thread runs simulations
- Qt signals/slots enable safe cross-thread communication
- Progress updates maintain responsive interface

### **Validation System**
- Real-time parameter validation with visual feedback
- Cross-dependency checking between configuration tabs
- Industry-standard thresholds for performance assessment
- Comprehensive error reporting with specific guidance

---

**🎉 Congratulations!** You're now ready to use the Star Tracker Simulation GUI effectively. The interface combines powerful simulation capabilities with an intuitive user experience, making complex star tracker analysis accessible to both experts and newcomers.

For additional technical details, see the `GUI/GUI_dev_diary.md` file which documents the complete development process and architecture decisions.