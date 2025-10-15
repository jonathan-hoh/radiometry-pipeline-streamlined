#!/usr/bin/env python3
"""
sensor_config.py - Sensor Configuration Tab Widget

Provides sensor parameter configuration interface for the tabbed wizard.
Part of Phase 3 implementation.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QSlider, QLabel, 
    QToolButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import logging

logger = logging.getLogger(__name__)


class SensorConfigWidget(QWidget):
    """
    Sensor configuration tab for star tracker simulation parameters.
    
    Provides input fields for:
    - Pixel pitch (3-10µm) with slider
    - Resolution (512x512, 1024x1024, 2048x2048)
    - Quantum efficiency (0-100%)
    - Read noise (0-50 e⁻)
    - Dark current (0-200 e⁻/s)
    """
    
    # Signal emitted when configuration changes
    config_changed = pyqtSignal()
    validation_changed = pyqtSignal(bool)  # True if valid
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.setup_validation()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("1. Sensor Configuration")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Configure the sensor characteristics of your star tracker.")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Sensor parameters group
        sensor_group = QGroupBox("Sensor Parameters")
        form_layout = QFormLayout(sensor_group)
        
        # Pixel pitch with slider
        pixel_row = QHBoxLayout()
        self.pixel_pitch_input = QDoubleSpinBox()
        self.pixel_pitch_input.setRange(3.0, 10.0)
        self.pixel_pitch_input.setValue(5.6)
        self.pixel_pitch_input.setSuffix(" µm")
        self.pixel_pitch_input.setDecimals(1)
        self.pixel_pitch_input.setMinimumWidth(100)
        
        self.pixel_pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pixel_pitch_slider.setRange(30, 100)  # 3.0-10.0 * 10
        self.pixel_pitch_slider.setValue(56)  # 5.6 * 10
        self.pixel_pitch_slider.setMinimumWidth(150)
        
        pixel_info_btn = self._create_info_button(
            "Physical size of each pixel element. Smaller pixels provide "
            "better angular resolution but may have lower sensitivity."
        )
        
        pixel_row.addWidget(self.pixel_pitch_input)
        pixel_row.addWidget(self.pixel_pitch_slider)
        pixel_row.addWidget(pixel_info_btn)
        pixel_row.addWidget(self._create_validation_indicator("pixel_pitch"))
        pixel_row.addStretch()
        
        form_layout.addRow("Pixel Pitch:", pixel_row)
        
        # Resolution
        resolution_row = QHBoxLayout()
        self.resolution_input = QComboBox()
        self.resolution_input.addItems(["512x512", "1024x1024", "2048x2048"])
        self.resolution_input.setCurrentText("2048x2048")
        self.resolution_input.setMinimumWidth(120)
        
        resolution_info_btn = self._create_info_button(
            "Detector array size. Higher resolution provides better "
            "field coverage but increases processing time."
        )
        
        resolution_row.addWidget(self.resolution_input)
        resolution_row.addWidget(resolution_info_btn)
        resolution_row.addWidget(self._create_validation_indicator("resolution"))
        resolution_row.addStretch()
        
        form_layout.addRow("Resolution:", resolution_row)
        
        # Quantum efficiency with slider
        qe_row = QHBoxLayout()
        self.quantum_efficiency_input = QSpinBox()
        self.quantum_efficiency_input.setRange(0, 100)
        self.quantum_efficiency_input.setValue(60)
        self.quantum_efficiency_input.setSuffix("%")
        self.quantum_efficiency_input.setMinimumWidth(100)
        
        self.qe_slider = QSlider(Qt.Orientation.Horizontal)
        self.qe_slider.setRange(0, 100)
        self.qe_slider.setValue(60)
        self.qe_slider.setMinimumWidth(150)
        
        qe_info_btn = self._create_info_button(
            "Percentage of incident photons converted to electrons. "
            "Higher values improve sensitivity to faint stars."
        )
        
        qe_row.addWidget(self.quantum_efficiency_input)
        qe_row.addWidget(self.qe_slider)
        qe_row.addWidget(qe_info_btn)
        qe_row.addWidget(self._create_validation_indicator("quantum_efficiency"))
        qe_row.addStretch()
        
        form_layout.addRow("Quantum Efficiency:", qe_row)
        
        # Read noise
        read_noise_row = QHBoxLayout()
        self.read_noise_input = QDoubleSpinBox()
        self.read_noise_input.setRange(0.0, 50.0)
        self.read_noise_input.setValue(13.0)
        self.read_noise_input.setSuffix(" e⁻")
        self.read_noise_input.setDecimals(1)
        self.read_noise_input.setMinimumWidth(100)
        
        read_noise_info_btn = self._create_info_button(
            "Electronic noise introduced during readout. Lower values "
            "improve detection of faint stars."
        )
        
        read_noise_row.addWidget(self.read_noise_input)
        read_noise_row.addWidget(read_noise_info_btn)
        read_noise_row.addWidget(self._create_validation_indicator("read_noise"))
        read_noise_row.addStretch()
        
        form_layout.addRow("Read Noise:", read_noise_row)
        
        # Dark current
        dark_current_row = QHBoxLayout()
        self.dark_current_input = QDoubleSpinBox()
        self.dark_current_input.setRange(0.0, 200.0)
        self.dark_current_input.setValue(50.0)
        self.dark_current_input.setSuffix(" e⁻/s")
        self.dark_current_input.setDecimals(1)
        self.dark_current_input.setMinimumWidth(100)
        
        dark_current_info_btn = self._create_info_button(
            "Thermally generated electrons per second. Lower values "
            "reduce background noise, especially for long exposures."
        )
        
        dark_current_row.addWidget(self.dark_current_input)
        dark_current_row.addWidget(dark_current_info_btn)
        dark_current_row.addWidget(self._create_validation_indicator("dark_current"))
        dark_current_row.addStretch()
        
        form_layout.addRow("Dark Current:", dark_current_row)
        
        layout.addWidget(sensor_group)
        
        # Add overall validation status
        self.validation_summary = QLabel()
        self.validation_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.validation_summary.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self.validation_summary)
        
        layout.addStretch()
        
    def _create_info_button(self, tooltip_text):
        """Create an information button with tooltip."""
        btn = QToolButton()
        btn.setText("ℹ️")
        btn.setToolTip(tooltip_text)
        btn.setMaximumSize(25, 25)
        btn.setStyleSheet("""
            QToolButton {
                border: 1px solid #ccc;
                border-radius: 12px;
                background-color: #f0f0f0;
                font-size: 12px;
            }
            QToolButton:hover {
                background-color: #e0e0e0;
            }
        """)
        return btn
        
    def _create_validation_indicator(self, field_name):
        """Create a validation indicator label."""
        indicator = QLabel("✓")
        indicator.setObjectName(f"validation_{field_name}")
        indicator.setStyleSheet("color: green; font-weight: bold; font-size: 16px;")
        indicator.setMinimumSize(20, 20)
        indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return indicator
        
    def setup_connections(self):
        """Setup signal connections."""
        # Connect input changes to validation
        self.pixel_pitch_input.valueChanged.connect(self._on_config_changed)
        self.pixel_pitch_slider.valueChanged.connect(self._on_pixel_slider_changed)
        self.resolution_input.currentTextChanged.connect(self._on_config_changed)
        self.quantum_efficiency_input.valueChanged.connect(self._on_config_changed)
        self.qe_slider.valueChanged.connect(self._on_qe_slider_changed)
        self.read_noise_input.valueChanged.connect(self._on_config_changed)
        self.dark_current_input.valueChanged.connect(self._on_config_changed)
        
    def setup_validation(self):
        """Setup initial validation state."""
        self._validate_all()
        
    def _on_pixel_slider_changed(self, value):
        """Handle pixel pitch slider changes."""
        # Convert slider value (30-100) to actual value (3.0-10.0)
        actual_value = value / 10.0
        self.pixel_pitch_input.setValue(actual_value)
        
    def _on_qe_slider_changed(self, value):
        """Handle quantum efficiency slider changes."""
        self.quantum_efficiency_input.setValue(value)
        
    def _on_config_changed(self):
        """Handle configuration changes."""
        # Sync sliders with spin boxes
        self.pixel_pitch_slider.setValue(int(self.pixel_pitch_input.value() * 10))
        self.qe_slider.setValue(self.quantum_efficiency_input.value())
        
        self._validate_all()
        self.config_changed.emit()
        
    def _validate_all(self):
        """Validate all input fields."""
        is_valid = True
        
        # Validate each field
        validation_results = {
            'pixel_pitch': self._validate_pixel_pitch(),
            'resolution': self._validate_resolution(),
            'quantum_efficiency': self._validate_quantum_efficiency(),
            'read_noise': self._validate_read_noise(),
            'dark_current': self._validate_dark_current()
        }
        
        # Update validation indicators
        for field, valid in validation_results.items():
            indicator = self.findChild(QLabel, f"validation_{field}")
            if indicator:
                if valid:
                    indicator.setText("✓")
                    indicator.setStyleSheet("color: green; font-weight: bold; font-size: 16px;")
                else:
                    indicator.setText("✗")
                    indicator.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
                    is_valid = False
        
        # Update validation summary
        if is_valid:
            self.validation_summary.setText("✅ Sensor configuration is valid")
            self.validation_summary.setStyleSheet("color: green; font-weight: bold; padding: 10px;")
        else:
            self.validation_summary.setText("⚠️ Please check highlighted fields")
            self.validation_summary.setStyleSheet("color: orange; font-weight: bold; padding: 10px;")
            
        self.validation_changed.emit(is_valid)
        
    def _validate_pixel_pitch(self):
        """Validate pixel pitch value."""
        value = self.pixel_pitch_input.value()
        return 3.0 <= value <= 10.0
        
    def _validate_resolution(self):
        """Validate resolution selection."""
        return self.resolution_input.currentText() in ["512x512", "1024x1024", "2048x2048"]
        
    def _validate_quantum_efficiency(self):
        """Validate quantum efficiency value."""
        value = self.quantum_efficiency_input.value()
        return 0 <= value <= 100
        
    def _validate_read_noise(self):
        """Validate read noise value."""
        value = self.read_noise_input.value()
        return 0.0 <= value <= 50.0
        
    def _validate_dark_current(self):
        """Validate dark current value."""
        value = self.dark_current_input.value()
        return 0.0 <= value <= 200.0
        
    def is_valid(self):
        """Check if all configuration is valid."""
        return (self._validate_pixel_pitch() and 
                self._validate_resolution() and
                self._validate_quantum_efficiency() and
                self._validate_read_noise() and
                self._validate_dark_current())
                
    def get_config(self):
        """Get current sensor configuration."""
        return {
            'pixel_pitch': self.pixel_pitch_input.value(),
            'resolution': self.resolution_input.currentText(),
            'quantum_efficiency': self.quantum_efficiency_input.value(),
            'read_noise': self.read_noise_input.value(),
            'dark_current': self.dark_current_input.value()
        }
        
    def set_config(self, config):
        """Set sensor configuration from dictionary."""
        if 'pixel_pitch' in config:
            self.pixel_pitch_input.setValue(config['pixel_pitch'])
        if 'resolution' in config:
            self.resolution_input.setCurrentText(config['resolution'])
        if 'quantum_efficiency' in config:
            self.quantum_efficiency_input.setValue(config['quantum_efficiency'])
        if 'read_noise' in config:
            self.read_noise_input.setValue(config['read_noise'])
        if 'dark_current' in config:
            self.dark_current_input.setValue(config['dark_current'])
            
        self._validate_all()