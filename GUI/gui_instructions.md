## PHASED LOCAL GUI DEVELOPMENT INSTRUCTIONS FOR CLAUDE CODE

### CONTEXT
Build a desktop GUI for star tracker simulation using existing Python codebase. GUI runs entirely locally using Python GUI framework (PyQt6 or Tkinter) with no web server required. Implements 3-screen workflow: Configuration → Execution → Results. Direct integration with StarTrackerPipeline class from `src/core/star_tracker_pipeline.py`.

### ARCHITECTURE
- **Framework**: PyQt6 (recommended) or Tkinter (simpler but less polished)
- **Threading**: QThreadPool/threading for non-blocking simulation execution
- **Plotting**: matplotlib embedded in GUI widgets
- **State**: In-memory Python objects, optional JSON file persistence
- **File I/O**: Direct filesystem access to existing `data/` and `outputs/` directories

---

### PHASE 1: BASIC SINGLE-WINDOW FORM (Days 1-2)

**Objective**: Single window with input fields that triggers simulation and displays numerical results

**Setup**:
```bash
# Install dependencies
pip install PyQt6 matplotlib numpy

# Create GUI structure
mkdir -p gui/{widgets,utils}
touch gui/__init__.py gui/main_window.py gui/config_form.py
```

**Tasks**:

1. **Main Application** (`gui/main_window.py`):
```python
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from src.core.star_tracker_pipeline import StarTrackerPipeline

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Star Tracker Simulator")
        self.setGeometry(100, 100, 800, 600)
        # Setup UI components
```

2. **Configuration Form** (`gui/config_form.py`):
   - Create QFormLayout with 5 input fields:
     - Focal length: QDoubleSpinBox (10-100mm, default 35)
     - Pixel pitch: QDoubleSpinBox (3-10µm, default 5.6)
     - Resolution: QComboBox ["512x512", "1024x1024", "2048x2048"]
     - Magnitude: QDoubleSpinBox (0-10, default 3.0)
     - Num trials: QSpinBox (10-5000, default 100)
   - "Run Simulation" QPushButton
   - QTextEdit for displaying results (read-only)

3. **Simulation Integration**:
```python
def run_simulation(self):
    config = {
        'focal_length': self.focal_length_input.value(),
        'pixel_pitch': self.pixel_pitch_input.value(),
        # ... other params
    }
    
    pipeline = StarTrackerPipeline(
        psf_file="data/PSF_sims/Gen_1/0_deg.txt",
        magnitude=config['magnitude'],
        num_simulations=config['num_trials']
    )
    results = pipeline.process_psf()
    
    self.results_text.setText(
        f"Centroid Error Mean: {results['centroid_error_mean']:.3f} px\n"
        f"Attitude Error: {results['attitude_error_rms']:.2f} arcsec"
    )
```

4. **Launch Script** (`run_gui.py`):
```python
import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

**Success Criteria**: Run `python run_gui.py`, fill form, click button, see numerical results appear

---

### PHASE 2: NON-BLOCKING EXECUTION WITH PROGRESS (Days 3-4)

**Objective**: Move simulation to background thread, add progress bar and live metrics

**Tasks**:

1. **Worker Thread** (`gui/utils/simulation_worker.py`):
```python
from PyQt6.QtCore import QThread, pyqtSignal

class SimulationWorker(QThread):
    progress_update = pyqtSignal(int, dict)  # (iteration, current_stats)
    finished = pyqtSignal(dict)  # results
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def run(self):
        # Modified pipeline that emits progress
        for i in range(self.config['num_trials']):
            # Run one iteration
            stats = {...}  # Current stats
            self.progress_update.emit(i, stats)
        
        self.finished.emit(final_results)
```

2. **Progress UI** (`gui/widgets/progress_panel.py`):
   - QProgressBar showing completion percentage
   - 4 QLabel widgets for live metrics:
     - Stars detected: 18
     - Centroids valid: 18
     - Match status: Success
     - Current attitude error: 8.1 arcsec
   - QLabel for time elapsed/remaining

3. **Main Window Integration**:
```python
def run_simulation(self):
    self.run_button.setEnabled(False)
    self.progress_panel.show()
    
    self.worker = SimulationWorker(self.get_config())
    self.worker.progress_update.connect(self.update_progress)
    self.worker.finished.connect(self.display_results)
    self.worker.start()

def update_progress(self, iteration, stats):
    percent = int((iteration / self.total_trials) * 100)
    self.progress_bar.setValue(percent)
    self.stars_label.setText(f"Stars: {stats['stars_detected']}")
    # Update other live metrics
```

4. **Cancel Button**:
   - Add "Cancel" button that calls `self.worker.terminate()`
   - Clean up partial results, re-enable form

**Success Criteria**: Simulation runs in background, GUI remains responsive, progress updates smoothly

---

### PHASE 3: TABBED WIZARD INTERFACE (Days 5-7)

**Objective**: Replace single form with tabbed wizard: Sensor | Optics | Scenario

**Tasks**:

1. **Tab Widget** (`gui/main_window.py`):
```python
from PyQt6.QtWidgets import QTabWidget

self.config_tabs = QTabWidget()
self.config_tabs.addTab(SensorConfigWidget(), "1. Sensor")
self.config_tabs.addTab(OpticsConfigWidget(), "2. Optics")
self.config_tabs.addTab(ScenarioConfigWidget(), "3. Scenario")
```

2. **Sensor Configuration Tab** (`gui/widgets/sensor_config.py`):
   - QGroupBox "Sensor Parameters"
   - 5 input fields with QLabel + QToolButton (ℹ️) tooltips:
     - Pixel pitch: QDoubleSpinBox + QSlider (3-10µm)
     - Resolution: QComboBox
     - Quantum efficiency: QSpinBox (0-100%)
     - Read noise: QDoubleSpinBox (0-50 e⁻)
     - Dark current: QDoubleSpinBox (0-200 e⁻/s)
   - Validation indicators (✓ green QLabel when valid)

3. **Optics Configuration Tab** (`gui/widgets/optics_config.py`):
   - Focal length: QDoubleSpinBox + QSlider (10-100mm)
   - Aperture: QComboBox [f/1.2, f/1.4, f/2.0, f/2.8]
   - FOV display: QLabel (auto-calculated, read-only)
   - Distortion: QComboBox [None, Minimal, Moderate]

4. **Scenario Configuration Tab** (`gui/widgets/scenario_config.py`):
   - Star catalog: QComboBox [Hipparcos, Gaia DR3]
   - Magnitude limit: QDoubleSpinBox + QSlider (0-10)
   - Attitude profile: QComboBox presets
   - Environment: QButtonGroup with 3 QRadioButtons [Deep space, LEO, GEO]
   - Monte Carlo trials: QSpinBox (100-5000)

5. **Navigation Buttons**:
   - Bottom toolbar: [< Back] [Next >] [Run Simulation]
   - "Back" disabled on first tab, "Next" disabled on last tab
   - "Run Simulation" only enabled when all tabs validated

6. **Validation System** (`gui/utils/validator.py`):
```python
def validate_sensor_config(config):
    if not 3 <= config['pixel_pitch'] <= 10:
        return False, "Pixel pitch must be 3-10µm"
    # ... other checks
    return True, ""
```

**Success Criteria**: Navigate 3 tabs, validation indicators update, complete config runs simulation

---

### PHASE 4: RESULTS VISUALIZATION (Days 8-10)

**Objective**: Display results in new window with statistics and embedded matplotlib plots

**Tasks**:

1. **Results Window** (`gui/results_window.py`):
```python
class ResultsWindow(QMainWindow):
    def __init__(self, results, config):
        super().__init__()
        self.setWindowTitle("Simulation Results")
        self.setGeometry(150, 150, 1200, 800)
        
        # 3-panel layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.create_stats_panel())
        splitter.addWidget(self.create_plot_panel())
        splitter.addWidget(self.create_export_panel())
        splitter.setSizes([300, 700, 200])
```

2. **Statistics Panel** (Left, `gui/widgets/stats_panel.py`):
   - QScrollArea with QVBoxLayout containing 3 QGroupBoxes:
     - **Centroiding**: Mean, Std, 95th percentile with checkmarks
     - **Attitude**: Mean, Std, 3σ bound with pass/fail
     - **Matching**: Success rate with threshold indicator
   - Use rich text in QLabel for formatting:
```python
stats_text = """
<b>Centroiding Performance</b><br>
Mean error: 0.153 px<br>
Std deviation: 0.081 px<br>
95th percentile: 0.298 px<br>
<span style='color:green'>✓ Within ±0.3 px criteria</span>
"""
self.stats_label.setText(stats_text)
```

3. **Plot Panel** (Center, `gui/widgets/plot_panel.py`):
   - QTabWidget with 4 tabs: Centroid Error | Attitude Error | Star Field | Residuals
   - Each tab contains `FigureCanvas` (matplotlib widget):
```python
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(figsize=(8, 6))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        
    def plot_centroid_histogram(self, errors):
        self.axes.hist(errors, bins=50, alpha=0.7)
        self.axes.set_xlabel('Centroid Error (pixels)')
        self.axes.set_ylabel('Frequency')
        self.axes.set_title('Centroid Error Distribution')
        self.draw()
```

4. **Export Panel** (Right, `gui/widgets/export_panel.py`):
   - QGroupBox "Export Options"
   - 4 QCheckBox widgets (pre-checked):
     - Summary report (PDF)
     - All plots (PNG)
     - Raw data (CSV)
     - Configuration (JSON)
   - QPushButton "Export Selected"
   - QPushButton "Save Results..."
   - Export logic:
```python
def export_results(self):
    dialog = QFileDialog()
    save_dir = dialog.getExistingDirectory(self, "Select Export Directory")
    
    if self.pdf_checkbox.isChecked():
        self.generate_pdf_report(save_dir)
    if self.plots_checkbox.isChecked():
        self.save_all_plots(save_dir)
    # ... other exports
    
    QMessageBox.information(self, "Export Complete", 
                          f"Results exported to {save_dir}")
```

5. **Window Management**:
   - Results window opens modeless (doesn't block main window)
   - Multiple results windows can be open simultaneously
   - Each window has unique job_id in title: "Results - Job_20251013_142305"

**Success Criteria**: Results window displays statistics, interactive plots, functional export to user-selected directory

---

### PHASE 5: PRESET TEMPLATES (Days 11-12)

**Objective**: Add template loading for common star tracker configurations

**Tasks**:

1. **Template Storage** (`gui/data/templates.json`):
```json
{
  "blue_canyon_nst": {
    "name": "Blue Canyon NST",
    "sensor": {
      "pixel_pitch": 5.6,
      "resolution": "2048x2048",
      "quantum_efficiency": 60,
      "read_noise": 13,
      "dark_current": 50
    },
    "optics": {
      "focal_length": 35,
      "aperture": "f/1.4"
    },
    "scenario": {
      "magnitude_limit": 6.5,
      "trials": 1000
    }
  },
  "sinclair_st16": {...},
  "custom": {...}
}
```

2. **Template Manager** (`gui/utils/template_manager.py`):
```python
import json
from pathlib import Path

class TemplateManager:
    def __init__(self):
        self.templates_file = Path("gui/data/templates.json")
        self.templates = self.load_templates()
    
    def load_templates(self):
        with open(self.templates_file) as f:
            return json.load(f)
    
    def get_template_names(self):
        return [t['name'] for t in self.templates.values()]
    
    def load_template(self, template_id):
        return self.templates[template_id]
```

3. **Template Dropdown** (`gui/main_window.py`):
   - Add QComboBox at top of config tabs: "Load Template:"
   - Populate with template names + "Custom (Empty)"
   - On selection changed:
```python
def on_template_selected(self, index):
    template_id = list(self.templates.keys())[index]
    config = self.template_manager.load_template(template_id)
    
    # Populate all form fields
    self.sensor_tab.set_config(config['sensor'])
    self.optics_tab.set_config(config['optics'])
    self.scenario_tab.set_config(config['scenario'])
    
    # Show info dialog
    QMessageBox.information(self, "Template Loaded", 
                          f"Loaded configuration: {config['name']}")
```

4. **Save Custom Template**:
   - Add "Save as Template..." button
   - QInputDialog for template name
   - Serialize current config to templates.json
   - Refresh dropdown to show new template

**Success Criteria**: Select "Blue Canyon NST" from dropdown, all fields auto-populate, simulation runs with preset

---

### PHASE 6: INPUT VALIDATION & ERROR HANDLING (Days 13-14)

**Objective**: Robust validation with visual feedback and graceful error recovery

**Tasks**:

1. **Real-Time Validation** (`gui/widgets/validated_input.py`):
```python
class ValidatedDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, min_val, max_val, tooltip):
        super().__init__()
        self.setRange(min_val, max_val)
        self.setToolTip(tooltip)
        self.valueChanged.connect(self.validate)
        
    def validate(self):
        if self.hasAcceptableInput():
            self.setStyleSheet("border: 2px solid green;")
            self.validation_icon.setText("✓")
            self.validation_icon.setStyleSheet("color: green;")
        else:
            self.setStyleSheet("border: 2px solid red;")
            self.validation_icon.setText("✗")
            self.validation_icon.setStyleSheet("color: red;")
```

2. **Form-Level Validation**:
   - Add `is_valid()` method to each config widget
   - Main window checks all tabs before enabling "Run Simulation"
   - Display validation summary in status bar:
```python
def check_all_validation(self):
    errors = []
    if not self.sensor_tab.is_valid():
        errors.append("Sensor configuration incomplete")
    if not self.optics_tab.is_valid():
        errors.append("Optics configuration incomplete")
    
    if errors:
        self.statusBar().showMessage(" | ".join(errors))
        self.run_button.setEnabled(False)
    else:
        self.statusBar().showMessage("Ready to run simulation")
        self.run_button.setEnabled(True)
```

3. **Error Dialogs** (`gui/utils/error_handler.py`):
```python
class ErrorHandler:
    @staticmethod
    def handle_simulation_error(exception):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Simulation Error")
        msg.setText("Simulation failed to complete")
        msg.setInformativeText(str(exception))
        msg.setDetailedText(traceback.format_exc())
        msg.exec()
    
    @staticmethod
    def handle_file_not_found(filepath):
        QMessageBox.warning(None, "File Not Found",
                          f"Could not locate: {filepath}\n\n"
                          f"Please check data directory structure.")
```

4. **Graceful Failure Recovery**:
   - Wrap simulation execution in try/except
   - On failure: Log error, show dialog, re-enable form
   - Auto-save last valid config to `~/.star_tracker_gui/last_config.json`
   - On startup, offer to restore last config

5. **Loading States**:
   - Disable all inputs during simulation: `setEnabled(False)`
   - Show modal progress dialog with cancel button:
```python
progress = QProgressDialog("Running simulation...", "Cancel", 
                          0, self.num_trials, self)
progress.setWindowModality(Qt.WindowModal)
progress.canceled.connect(self.worker.terminate)
```

**Success Criteria**: Invalid inputs show red borders + tooltips, simulation errors display friendly dialogs, cancel works cleanly

---

### PHASE 7: COMPARISON MODE (Days 15-16)

**Objective**: Side-by-side comparison of two simulation results

**Tasks**:

1. **Results History** (`gui/utils/results_cache.py`):
```python
import pickle
from pathlib import Path

class ResultsCache:
    def __init__(self):
        self.cache_dir = Path("outputs/.gui_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def save_result(self, job_id, results, config):
        cache_file = self.cache_dir / f"{job_id}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump({'results': results, 'config': config}, f)
    
    def load_result(self, job_id):
        cache_file = self.cache_dir / f"{job_id}.pkl"
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def list_results(self):
        return [f.stem for f in self.cache_dir.glob("*.pkl")]
```

2. **Comparison Window** (`gui/comparison_window.py`):
   - QSplitter with 2 ResultsWindow-like panels (left and right)
   - Top bar with 2 QComboBox: "Baseline:" and "Current:"
   - Load cached results into each panel
   - Center divider shows delta metrics:
```python
delta_widget = QWidget()
delta_layout = QVBoxLayout()
delta_layout.addWidget(QLabel(f"Δ Centroid Error: {delta_centroid:+.3f} px"))
delta_layout.addWidget(QLabel(f"Δ Attitude Error: {delta_attitude:+.2f} arcsec"))
# Highlight significant changes in red/green
```

3. **Comparison Plots**:
   - Overlay mode: Both histograms on same axes (different colors)
   - Difference mode: Plot baseline - current
   - Toggle with QRadioButton: [Overlay] [Difference]

4. **Launch Comparison**:
   - Add "Compare..." button to results window
   - Opens comparison window with current result pre-selected
   - User selects baseline from dropdown of cached results

**Success Criteria**: Open comparison window, select two runs, see side-by-side stats and overlayed plots

---

### PHASE 8: PARAMETER SWEEP (Days 17-19, Optional)

**Objective**: Run multiple simulations varying one parameter, display sweep plot

**Tasks**:

1. **Sweep Configuration** (`gui/widgets/sweep_config.py`):
   - Add "Run Parameter Sweep" QCheckBox to scenario tab
   - When checked, show sweep configuration panel:
     - Parameter to vary: QComboBox [Focal length, Pixel pitch, Magnitude]
     - Start value: QDoubleSpinBox
     - End value: QDoubleSpinBox
     - Step size: QDoubleSpinBox
     - Preview: QLabel "Will run X simulations"

2. **Batch Execution** (`gui/utils/batch_runner.py`):
```python
class BatchRunner(QThread):
    progress_update = pyqtSignal(int, int)  # (current, total)
    batch_complete = pyqtSignal(list)  # results
    
    def run(self):
        results = []
        param_values = np.arange(self.start, self.end, self.step)
        
        for i, value in enumerate(param_values):
            config = self.base_config.copy()
            config[self.param_name] = value
            
            result = run_single_simulation(config)
            results.append({'param_value': value, 'result': result})
            
            self.progress_update.emit(i+1, len(param_values))
        
        self.batch_complete.emit(results)
```

3. **Sweep Results Window** (`gui/sweep_results_window.py`):
   - QTableWidget showing all runs:
     - Columns: Focal Length | Centroid Error | Attitude Error | Status
     - Sortable by clicking headers
   - Matplotlib plot: Parameter (x-axis) vs Performance (y-axis)
   - Dual y-axis: Centroid error (left), Attitude error (right)
   - Add trendline/curve fit

4. **Export Sweep Results**:
   - "Export Sweep Data" button → CSV with all runs
   - "Export Sweep Plot" button → High-res PNG

**Success Criteria**: Configure sweep (focal length 25-50mm, step 5mm), runs 6 simulations, displays plot showing performance vs parameter

---

### PHASE 9: PSF FILE BROWSER (Days 20-21, Optional)

**Objective**: Visual browser for selecting PSF files from `data/PSF_sims/`

**Tasks**:

1. **PSF Browser Dialog** (`gui/widgets/psf_browser.py`):
   - QDialog with file tree on left, preview on right
   - QTreeWidget showing PSF directory structure:
     - Gen_1/
       - 0_deg.txt
       - 5_deg.txt
       - ...
     - Gen_2/
       - 0_deg.txt
       - ...

2. **PSF Preview Panel**:
   - Show PSF metadata when file selected:
     - Grid size: 128×128
     - Pixel spacing: 0.5µm
     - Field angle: 0°
     - Total energy: 94.3%
   - Thumbnail plot: 2D heatmap of PSF intensity

3. **Integration**:
   - Add "Browse PSF Files..." button to scenario config
   - Opens browser dialog, returns selected file path
   - Auto-populate related fields (field angle, etc.)

**Success Criteria**: Click browse, navigate tree, see previews, select PSF file, path populates in form

---

### PHASE 10: SETTINGS & PREFERENCES (Days 22-23, Optional)

**Objective**: Persistent user preferences and configuration

**Tasks**:

1. **Settings Dialog** (`gui/settings_dialog.py`):
   - QDialog with QTabWidget:
     - General: Default output directory, auto-save results
     - Plotting: Default DPI, figure size, color scheme
     - Performance: Number of CPU cores, memory limit
     - Advanced: Debug logging, cache size limit

2. **Settings Storage** (`gui/utils/settings.py`):
```python
from PyQt6.QtCore import QSettings

class AppSettings:
    def __init__(self):
        self.settings = QSettings("SOE", "StarTrackerGUI")
    
    def get(self, key, default=None):
        return self.settings.value(key, default)
    
    def set(self, key, value):
        self.settings.setValue(key, value)
```

3. **Menu Bar Integration**:
   - Main window menu: File | Edit | View | Tools | Help
   - Edit menu: "Preferences..." → Opens settings dialog
   - File menu: "Recent Results" submenu (last 10)

4. **Apply Settings**:
   - Load settings on startup: `self.settings.get('output_dir')`
   - Apply to simulation: Use preferred CPU count
   - Apply to plots: Use saved DPI and color scheme

**Success Criteria**: Open preferences, change output directory, setting persists after restart

---

### TESTING CHECKLIST (Throughout Development)

After each phase:
- **Manual Testing**: Click all buttons, fill all fields, trigger all paths
- **Edge Cases**: Empty inputs, extreme values, cancel mid-simulation
- **Error Injection**: Delete PSF file, corrupt config, fill disk
- **Performance**: Monitor memory usage during long simulations
- **Cross-Platform**: Test on Windows/Mac/Linux if targeting multiple OS

---

### PACKAGING & DISTRIBUTION (Post-Development)

**Option 1: PyInstaller (Recommended)**
```bash
# Create standalone executable
pip install pyinstaller
pyinstaller --name="StarTrackerGUI" \
            --windowed \
            --onefile \
            --add-data "data:data" \
            --add-data "gui/data:gui/data" \
            --icon="gui/assets/icon.ico" \
            run_gui.py

# Output: dist/StarTrackerGUI.exe (Windows) or dist/StarTrackerGUI (Mac/Linux)
```

**Option 2: Simple Python Distribution**
```bash
# Create distributable package
python setup.py sdist bdist_wheel

# Users install with:
pip install star_tracker_gui-1.0.0-py3-none-any.whl
python -m star_tracker_gui
```

**Installer Creation**:
- Windows: Use Inno Setup or NSIS
- Mac: Create .app bundle with py2app
- Linux: Create .deb or .rpm package

---

### CRITICAL INTEGRATION POINTS

**Direct File System Access**:
```python
# PSF files
PSF_DIR = Path("data/PSF_sims/Gen_1")
psf_files = list(PSF_DIR.glob("*_deg.txt"))

# Star catalogs
CATALOG_DIR = Path("data/catalogs")
catalog_file = CATALOG_DIR / "hipparcos.csv"

# Output directory
OUTPUT_DIR = Path("outputs")
job_dir = OUTPUT_DIR / f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
job_dir.mkdir(exist_ok=True)
```

**Pipeline Integration**:
```python
# Import existing code
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.multi_star.multi_star_pipeline import run_multi_star_analysis
from src.BAST.match import match_triangles

# Use directly - no API layer needed
pipeline = StarTrackerPipeline(
    psf_file=str(psf_file),
    magnitude=magnitude,
    num_simulations=trials
)
results = pipeline.process_psf()

# Access results immediately
centroid_errors = results['centroid_errors']
attitude_solution = results['attitude_quaternion']
```

**Threading Pattern**:
```python
# Safe pattern for updating GUI from worker thread
class SimulationWorker(QThread):
    result_ready = pyqtSignal(dict)  # Only emit, don't call GUI directly
    
    def run(self):
        results = self.pipeline.process_psf()
        self.result_ready.emit(results)  # Signal back to main thread

# Main window connects signal
self.worker.result_ready.connect(self.display_results)  # Runs on main thread
```

---

### MINIMAL STARTUP EXAMPLE

**Quick Test** (`minimal_gui.py`):
```python
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                              QPushButton, QLabel, QWidget)
from src.core.star_tracker_pipeline import StarTrackerPipeline

class MinimalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Star Tracker - Minimal Test")
        
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.label = QLabel("Click to run simulation")
        self.button = QPushButton("Run Default Simulation")
        self.button.clicked.connect(self.run_sim)
        
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        widget.setLayout(layout)
        self.setCentralWidget(widget)
    
    def run_sim(self):
        self.label.setText("Running...")
        self.button.setEnabled(False)
        
        pipeline = StarTrackerPipeline(
            psf_file="data/PSF_sims/Gen_1/0_deg.txt",
            magnitude=3.0,
            num_simulations=50
        )
        results = pipeline.process_psf()
        
        self.label.setText(f"Done! Centroid error: {results['centroid_error_mean']:.3f} px")
        self.button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MinimalGUI()
    window.show()
    sys.exit(app.exec())
```

Test with: `python minimal_gui.py`

---

Use this phased approach to build desktop GUI iteratively. PyQt6 provides more professional appearance than Tkinter. Prioritize Phases 1-4 for functional MVP, add advanced features (7-10) as time permits. Local architecture is simpler than web version: no server, no async jobs, direct filesystem access.