#!/usr/bin/env python3
"""
gui/main_window.py - Main Application Window

Main window for the Star Tracker GUI application.
Integrates configuration form with simulation execution.
Part of Phase 1 implementation.
"""

import sys
import logging
import traceback
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QStatusBar, QMenuBar,
    QMessageBox, QApplication, QSplitter
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon

# Import simulation components
try:
    from src.core.star_tracker_pipeline import StarTrackerPipeline
except ImportError as e:
    logging.error(f"Failed to import StarTrackerPipeline: {e}")
    StarTrackerPipeline = None

# Import GUI components
from .config_form import ConfigForm
from .widgets.progress_panel import ProgressPanel
from .utils.simulation_worker import SimulationWorker

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """
    Main application window for Star Tracker GUI.
    
    Provides:
    - Configuration form interface
    - Simulation execution
    - Results display
    - Basic menu system
    """
    
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.simulation_worker = None
        self.setup_ui()
        self.setup_menu()
        self.setup_connections()
        self.check_dependencies()
        
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Star Tracker Simulator v1.0")
        self.setGeometry(100, 100, 800, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter for Phase 2
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Create splitter for config form and progress panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Configuration form (left side)
        self.config_form = ConfigForm()
        splitter.addWidget(self.config_form)
        
        # Progress panel (right side)
        self.progress_panel = ProgressPanel()
        splitter.addWidget(self.progress_panel)
        
        # Set splitter proportions (60% config, 40% progress)
        splitter.setSizes([480, 320])
        
        layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Apply application styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #ffffff;
            }
        """)
        
    def setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # About action
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_connections(self):
        """Setup signal connections."""
        self.config_form.run_simulation_requested.connect(self.run_simulation)
        self.progress_panel.cancel_requested.connect(self.cancel_simulation)
        
    def check_dependencies(self):
        """Check if required dependencies are available."""
        if StarTrackerPipeline is None:
            self.status_bar.showMessage("⚠️ StarTrackerPipeline not available - check imports")
            QMessageBox.warning(
                self,
                "Import Error",
                "Failed to import StarTrackerPipeline.\n\n"
                "Please ensure the simulation modules are available in the Python path.\n"
                "Try running: PYTHONPATH=. python run_gui.py"
            )
            return False
            
        # Check for PSF data files
        psf_dir = Path("data/PSF_sims/Gen_1")
        if not psf_dir.exists():
            self.status_bar.showMessage("⚠️ PSF data directory not found")
            QMessageBox.warning(
                self,
                "Data Directory Missing",
                f"PSF data directory not found: {psf_dir}\n\n"
                "Please ensure the data directory structure is in place."
            )
            return False
            
        # Check for at least one PSF file
        psf_files = list(psf_dir.glob("*_deg.txt"))
        if not psf_files:
            self.status_bar.showMessage("⚠️ No PSF files found")
            QMessageBox.warning(
                self,
                "PSF Files Missing",
                f"No PSF files found in: {psf_dir}\n\n"
                "Please ensure PSF simulation data files are available."
            )
            return False
            
        self.status_bar.showMessage("✅ All dependencies available")
        logger.info(f"Found {len(psf_files)} PSF files")
        return True
        
    def run_simulation(self, config):
        """Run star tracker simulation with given configuration (Phase 2 - threaded)."""
        logger.info(f"Starting threaded simulation with config: {config}")
        
        try:
            # Check if simulation is already running
            if self.simulation_worker and self.simulation_worker.isRunning():
                QMessageBox.warning(
                    self,
                    "Simulation Running",
                    "A simulation is already running. Please wait for it to complete or cancel it first."
                )
                return
                
            # Initialize pipeline
            psf_file = "data/PSF_sims/Gen_1/0_deg.txt"  # Default PSF file
            
            if not Path(psf_file).exists():
                raise FileNotFoundError(f"PSF file not found: {psf_file}")
                
            self.pipeline = StarTrackerPipeline(
                psf_file=psf_file,
                magnitude=config['magnitude'],
                num_simulations=config['num_trials']
            )
            
            # Create worker thread
            self.simulation_worker = SimulationWorker(self.pipeline, config)
            
            # Connect worker signals
            self.simulation_worker.progress_updated.connect(self.progress_panel.update_progress)
            self.simulation_worker.metrics_updated.connect(self.progress_panel.update_metrics)
            self.simulation_worker.simulation_finished.connect(self.on_simulation_finished)
            self.simulation_worker.simulation_failed.connect(self.on_simulation_failed)
            
            # Update UI state
            self.config_form.set_simulation_running(True)
            self.progress_panel.start_simulation()
            self.status_bar.showMessage("Running simulation in background...")
            
            # Start the worker thread
            self.simulation_worker.start()
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to start simulation: {error_msg}")
            logger.error(traceback.format_exc())
            
            # Show error to user
            self.status_bar.showMessage(f"❌ Failed to start simulation: {error_msg}")
            QMessageBox.critical(
                self,
                "Simulation Startup Error",
                f"Failed to start simulation:\n\n{error_msg}\n\n"
                "Please check the configuration and try again."
            )
            
            # Reset UI state
            self.config_form.set_simulation_running(False)
            

    def cancel_simulation(self):
        """Cancel the currently running simulation."""
        if self.simulation_worker and self.simulation_worker.isRunning():
            logger.info("User requested simulation cancellation")
            reply = QMessageBox.question(
                self, "Cancel Simulation",
                "Are you sure you want to cancel the running simulation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.simulation_worker.cancel()
                self.progress_panel.simulation_cancelled()
                self.status_bar.showMessage("Cancelling simulation...")
                if self.simulation_worker.wait(3000):
                    logger.info("Simulation cancelled successfully")
                    self.status_bar.showMessage("Simulation cancelled")
                else:
                    logger.warning("Simulation thread did not respond to cancellation")
                    self.status_bar.showMessage("⚠️ Simulation thread unresponsive")
                self.config_form.set_simulation_running(False)

    def on_simulation_finished(self, results):
        """Handle successful simulation completion."""
        logger.info("Simulation completed successfully")
        self.progress_panel.simulation_finished(success=True)
        if isinstance(results, dict):
            results["execution_time"] = getattr(self.pipeline, "execution_time", 0.0)
            results["configuration"] = self.simulation_worker.config if self.simulation_worker else {}
        self.config_form.update_results(results)
        self.status_bar.showMessage("✅ Simulation completed successfully")
        self.config_form.set_simulation_running(False)

    def on_simulation_failed(self, error_message):
        """Handle simulation failure."""
        logger.error(f"Simulation failed: {error_message}")
        self.progress_panel.simulation_finished(success=False, message=error_message)
        self.config_form.update_results({"error": error_message})
        self.status_bar.showMessage(f"❌ Simulation failed: {error_message}")
        QMessageBox.critical(
            self, "Simulation Error",
            f"The simulation failed with the following error:\n\n{error_message}\n\n"
            "Please check the configuration and try again."
        )
        self.config_form.set_simulation_running(False)
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Star Tracker Simulator",
            "<h3>Star Tracker Simulator</h3>"
            "<p>Desktop GUI for star tracker radiometry simulation.</p>"
            "<p><b>Version:</b> 1.0.0</p>"
            "<p><b>Author:</b> SOE SaaS Physics Team</p>"
            "<p><b>Framework:</b> PyQt6</p>"
            "<br>"
            "<p>This application provides a graphical interface for configuring "
            "and running star tracker simulations with real-time results visualization.</p>"
        )
        
    def closeEvent(self, event):
        """Handle application close event."""
        # Check if simulation is running
        if not self.config_form.run_button.isEnabled():
            reply = QMessageBox.question(
                self,
                "Simulation Running",
                "A simulation is currently running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
                
        # Accept close event
        logger.info("Application closing")
        event.accept()

def main():
    """Main application entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('gui_application.log')
        ]
    )
    
    logger.info("Starting Star Tracker GUI application")
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Star Tracker Simulator")
    app.setApplicationVersion("1.0.0")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    exit_code = app.exec()
    logger.info(f"Application exited with code {exit_code}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())