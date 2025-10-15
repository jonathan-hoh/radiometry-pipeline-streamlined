#!/usr/bin/env python3
"""
gui/config_form.py - Configuration Form Widget

Basic configuration form with input fields for star tracker simulation parameters.
Part of Phase 1 implementation.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QTextEdit,
    QLabel, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import logging

logger = logging.getLogger(__name__)

class ConfigForm(QWidget):
    """
    Configuration form widget for star tracker simulation parameters.
    
    Provides input fields for:
    - Focal length (10-100mm)
    - Pixel pitch (3-10µm) 
    - Resolution (512x512, 1024x1024, 2048x2048)
    - Magnitude (0-10)
    - Number of trials (10-5000)
    """
    
    # Signal emitted when simulation should be run
    run_simulation_requested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Star Tracker Simulation Configuration")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Configuration form
        config_group = QGroupBox("Simulation Parameters")
        form_layout = QFormLayout(config_group)
        
        # Focal length input
        self.focal_length_input = QDoubleSpinBox()
        self.focal_length_input.setRange(10.0, 100.0)
        self.focal_length_input.setValue(35.0)
        self.focal_length_input.setSuffix(" mm")
        self.focal_length_input.setDecimals(1)
        self.focal_length_input.setToolTip("Focal length of the optical system (10-100mm)")
        form_layout.addRow("Focal Length:", self.focal_length_input)
        
        # Pixel pitch input
        self.pixel_pitch_input = QDoubleSpinBox()
        self.pixel_pitch_input.setRange(3.0, 10.0)
        self.pixel_pitch_input.setValue(5.6)
        self.pixel_pitch_input.setSuffix(" µm")
        self.pixel_pitch_input.setDecimals(1)
        self.pixel_pitch_input.setToolTip("Pixel pitch of the detector (3-10µm)")
        form_layout.addRow("Pixel Pitch:", self.pixel_pitch_input)
        
        # Resolution input
        self.resolution_input = QComboBox()
        self.resolution_input.addItems(["512x512", "1024x1024", "2048x2048"])
        self.resolution_input.setCurrentText("2048x2048")
        self.resolution_input.setToolTip("Detector resolution")
        form_layout.addRow("Resolution:", self.resolution_input)
        
        # Magnitude input
        self.magnitude_input = QDoubleSpinBox()
        self.magnitude_input.setRange(0.0, 10.0)
        self.magnitude_input.setValue(3.0)
        self.magnitude_input.setDecimals(1)
        self.magnitude_input.setToolTip("Star magnitude for simulation (0-10)")
        form_layout.addRow("Star Magnitude:", self.magnitude_input)
        
        # Number of trials input
        self.num_trials_input = QSpinBox()
        self.num_trials_input.setRange(10, 5000)
        self.num_trials_input.setValue(100)
        self.num_trials_input.setToolTip("Number of Monte Carlo trials (10-5000)")
        form_layout.addRow("Number of Trials:", self.num_trials_input)
        
        layout.addWidget(config_group)
        
        # Run simulation button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.run_button = QPushButton("Run Simulation")
        self.run_button.setMinimumHeight(40)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        button_layout.addWidget(self.run_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Results display
        results_group = QGroupBox("Simulation Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setPlaceholderText("Simulation results will appear here...")
        self.results_text.setStyleSheet("""
            QTextEdit {
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        results_layout.addWidget(self.results_text)
        layout.addWidget(results_group)
        
    def setup_connections(self):
        """Setup signal connections."""
        self.run_button.clicked.connect(self.on_run_simulation)
        
    def on_run_simulation(self):
        """Handle run simulation button click."""
        config = self.get_configuration()
        logger.info(f"Running simulation with config: {config}")
        self.run_simulation_requested.emit(config)
        
    def get_configuration(self):
        """Get current configuration from form inputs."""
        return {
            'focal_length': self.focal_length_input.value(),
            'pixel_pitch': self.pixel_pitch_input.value(),
            'resolution': self.resolution_input.currentText(),
            'magnitude': self.magnitude_input.value(),
            'num_trials': self.num_trials_input.value()
        }
    
    def set_configuration(self, config):
        """Set form inputs from configuration dictionary."""
        if 'focal_length' in config:
            self.focal_length_input.setValue(config['focal_length'])
        if 'pixel_pitch' in config:
            self.pixel_pitch_input.setValue(config['pixel_pitch'])
        if 'resolution' in config:
            self.resolution_input.setCurrentText(config['resolution'])
        if 'magnitude' in config:
            self.magnitude_input.setValue(config['magnitude'])
        if 'num_trials' in config:
            self.num_trials_input.setValue(config['num_trials'])
    
    def set_simulation_running(self, running):
        """Enable/disable form during simulation."""
        self.run_button.setEnabled(not running)
        self.focal_length_input.setEnabled(not running)
        self.pixel_pitch_input.setEnabled(not running)
        self.resolution_input.setEnabled(not running)
        self.magnitude_input.setEnabled(not running)
        self.num_trials_input.setEnabled(not running)
        
        if running:
            self.run_button.setText("Running...")
            self.results_text.setText("Simulation running...\n")
        else:
            self.run_button.setText("Run Simulation")
    
    def update_results(self, results):
        """Update results display with simulation results."""
        if 'error' in results:
            self.results_text.setText(f"❌ Simulation failed: {results['error']}")
            return
            
        # Format results for display
        results_text = "✅ Simulation completed successfully!\n\n"
        
        if 'centroid_error_mean' in results:
            results_text += f"Centroid Error Mean: {results['centroid_error_mean']:.3f} pixels\n"
        if 'centroid_error_std' in results:
            results_text += f"Centroid Error Std: {results['centroid_error_std']:.3f} pixels\n"
        if 'attitude_error_rms' in results:
            results_text += f"Attitude Error RMS: {results['attitude_error_rms']:.2f} arcseconds\n"
        if 'execution_time' in results:
            results_text += f"Execution Time: {results['execution_time']:.1f} seconds\n"
        if 'num_successful' in results:
            results_text += f"Successful Trials: {results['num_successful']}/{results.get('num_total', 'N/A')}\n"
        
        # Add raw results if available
        if 'raw_stats' in results:
            results_text += "\n--- Detailed Statistics ---\n"
            for key, value in results['raw_stats'].items():
                if isinstance(value, (int, float)):
                    results_text += f"{key}: {value:.3f}\n"
                else:
                    results_text += f"{key}: {value}\n"
        
        self.results_text.setText(results_text)
        logger.info("Results display updated")
    
    def clear_results(self):
        """Clear the results display."""
        self.results_text.clear()
        self.results_text.setPlaceholderText("Simulation results will appear here...")