#!/usr/bin/env python3
"""
optics_config.py - Optics Configuration Tab Widget

Provides optical system parameter configuration interface for the tabbed wizard.
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
import math

logger = logging.getLogger(__name__)


class OpticsConfigWidget(QWidget):
    """
    Optics configuration tab for star tracker simulation parameters.
    
    Provides input fields for:
    - Focal length (10-100mm) with slider
    - Aperture (f-numbers)
    - FOV display (auto-calculated, read-only)
    - Distortion selection
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
        title_label = QLabel("2. Optics Configuration")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Configure the optical system characteristics of your star tracker.")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Optics parameters group
        optics_group = QGroupBox("Optical Parameters")
        form_layout = QFormLayout(optics_group)
        
        # Focal length with slider
        focal_row = QHBoxLayout()
        self.focal_length_input = QDoubleSpinBox()
        self.focal_length_input.setRange(10.0, 100.0)
        self.focal_length_input.setValue(35.0)
        self.focal_length_input.setSuffix(" mm")
        self.focal_length_input.setDecimals(1)
        self.focal_length_input.setMinimumWidth(100)
        
        self.focal_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.focal_length_slider.setRange(100, 1000)  # 10.0-100.0 * 10
        self.focal_length_slider.setValue(350)  # 35.0 * 10
        self.focal_length_slider.setMinimumWidth(150)
        
        focal_info_btn = self._create_info_button(
            "Distance from the lens/mirror to the detector. Longer focal "
            "lengths provide narrower field of view but better angular resolution."
        )
        
        focal_row.addWidget(self.focal_length_input)
        focal_row.addWidget(self.focal_length_slider)
        focal_row.addWidget(focal_info_btn)
        focal_row.addWidget(self._create_validation_indicator("focal_length"))
        focal_row.addStretch()
        
        form_layout.addRow("Focal Length:", focal_row)
        
        # Aperture (f-number)
        aperture_row = QHBoxLayout()
        self.aperture_input = QComboBox()
        self.aperture_input.addItems(["f/1.2", "f/1.4", "f/2.0", "f/2.8", "f/4.0", "f/5.6"])
        self.aperture_input.setCurrentText("f/1.4")
        self.aperture_input.setMinimumWidth(120)
        
        aperture_info_btn = self._create_info_button(
            "Ratio of focal length to aperture diameter. Lower f-numbers "
            "gather more light but may have more optical aberrations."
        )
        
        aperture_row.addWidget(self.aperture_input)
        aperture_row.addWidget(aperture_info_btn)
        aperture_row.addWidget(self._create_validation_indicator("aperture"))
        aperture_row.addStretch()
        
        form_layout.addRow("Aperture (f-number):", aperture_row)
        
        # FOV display (calculated automatically)
        fov_row = QHBoxLayout()
        self.fov_display = QLabel("25.4° × 25.4°")
        self.fov_display.setStyleSheet("""
            QLabel {
                font-family: monospace;
                font-size: 12px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 5px;
                min-width: 120px;
            }
        """)
        
        fov_info_btn = self._create_info_button(
            "Field of view calculated from focal length and detector size. "
            "This determines how much sky the star tracker can observe."
        )
        
        fov_row.addWidget(self.fov_display)
        fov_row.addWidget(fov_info_btn)
        fov_row.addStretch()
        
        form_layout.addRow("Field of View:", fov_row)
        
        # Distortion
        distortion_row = QHBoxLayout()
        self.distortion_input = QComboBox()
        self.distortion_input.addItems(["None", "Minimal", "Moderate"])
        self.distortion_input.setCurrentText("Minimal")
        self.distortion_input.setMinimumWidth(120)
        
        distortion_info_btn = self._create_info_button(
            "Level of optical distortion. Real optical systems have some "
            "distortion that affects star position accuracy."
        )
        
        distortion_row.addWidget(self.distortion_input)
        distortion_row.addWidget(distortion_info_btn)
        distortion_row.addWidget(self._create_validation_indicator("distortion"))
        distortion_row.addStretch()
        
        form_layout.addRow("Distortion Level:", distortion_row)
        
        layout.addWidget(optics_group)
        
        # Optical performance summary
        performance_group = QGroupBox("Calculated Performance")
        perf_layout = QFormLayout(performance_group)
        
        # Angular resolution
        self.angular_resolution_display = QLabel("--")
        self.angular_resolution_display.setStyleSheet("""
            QLabel {
                font-family: monospace;
                font-size: 11px;
                color: #333;
            }
        """)
        perf_layout.addRow("Angular Resolution:", self.angular_resolution_display)
        
        # Light gathering power
        self.light_gathering_display = QLabel("--")
        self.light_gathering_display.setStyleSheet("""
            QLabel {
                font-family: monospace;
                font-size: 11px;
                color: #333;
            }
        """)
        perf_layout.addRow("Relative Light Gathering:", self.light_gathering_display)
        
        layout.addWidget(performance_group)
        
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
        # Connect input changes to validation and FOV calculation
        self.focal_length_input.valueChanged.connect(self._on_config_changed)
        self.focal_length_slider.valueChanged.connect(self._on_focal_slider_changed)
        self.aperture_input.currentTextChanged.connect(self._on_config_changed)
        self.distortion_input.currentTextChanged.connect(self._on_config_changed)
        
    def setup_validation(self):
        """Setup initial validation state."""
        self._calculate_derived_values()
        self._validate_all()
        
    def _on_focal_slider_changed(self, value):
        """Handle focal length slider changes."""
        # Convert slider value (100-1000) to actual value (10.0-100.0)
        actual_value = value / 10.0
        self.focal_length_input.setValue(actual_value)
        
    def _on_config_changed(self):
        """Handle configuration changes."""
        # Sync slider with spin box
        self.focal_length_slider.setValue(int(self.focal_length_input.value() * 10))
        
        self._calculate_derived_values()
        self._validate_all()
        self.config_changed.emit()
        
    def _calculate_derived_values(self):
        """Calculate derived values like FOV and performance metrics."""
        focal_length = self.focal_length_input.value()  # mm
        
        # Calculate FOV assuming 2048x2048 sensor with 5.6µm pixels (can be made dynamic later)
        sensor_size = 2048 * 5.6e-3  # mm (default values)
        fov_rad = 2 * math.atan(sensor_size / (2 * focal_length))
        fov_deg = math.degrees(fov_rad)
        
        self.fov_display.setText(f"{fov_deg:.1f}° × {fov_deg:.1f}°")
        
        # Calculate angular resolution (pixel size in arcseconds)
        pixel_size = 5.6e-3  # mm (default)
        angular_res_rad = pixel_size / focal_length
        angular_res_arcsec = math.degrees(angular_res_rad) * 3600
        
        self.angular_resolution_display.setText(f"{angular_res_arcsec:.1f} arcsec/pixel")
        
        # Calculate relative light gathering power
        aperture_text = self.aperture_input.currentText()
        f_number = float(aperture_text.replace("f/", ""))
        aperture_diameter = focal_length / f_number  # mm
        relative_area = (aperture_diameter / 25.0) ** 2  # Relative to 25mm aperture
        
        self.light_gathering_display.setText(f"{relative_area:.2f}× (vs f/1.4, 35mm)")
        
    def _validate_all(self):
        """Validate all input fields."""
        is_valid = True
        
        # Validate each field
        validation_results = {
            'focal_length': self._validate_focal_length(),
            'aperture': self._validate_aperture(),
            'distortion': self._validate_distortion()
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
            self.validation_summary.setText("✅ Optics configuration is valid")
            self.validation_summary.setStyleSheet("color: green; font-weight: bold; padding: 10px;")
        else:
            self.validation_summary.setText("⚠️ Please check highlighted fields")
            self.validation_summary.setStyleSheet("color: orange; font-weight: bold; padding: 10px;")
            
        self.validation_changed.emit(is_valid)
        
    def _validate_focal_length(self):
        """Validate focal length value."""
        value = self.focal_length_input.value()
        return 10.0 <= value <= 100.0
        
    def _validate_aperture(self):
        """Validate aperture selection."""
        return self.aperture_input.currentText() in ["f/1.2", "f/1.4", "f/2.0", "f/2.8", "f/4.0", "f/5.6"]
        
    def _validate_distortion(self):
        """Validate distortion selection."""
        return self.distortion_input.currentText() in ["None", "Minimal", "Moderate"]
        
    def is_valid(self):
        """Check if all configuration is valid."""
        return (self._validate_focal_length() and 
                self._validate_aperture() and
                self._validate_distortion())
                
    def get_config(self):
        """Get current optics configuration."""
        return {
            'focal_length': self.focal_length_input.value(),
            'aperture': self.aperture_input.currentText(),
            'distortion': self.distortion_input.currentText(),
            'calculated_fov': self.fov_display.text(),
            'calculated_angular_resolution': self.angular_resolution_display.text()
        }
        
    def set_config(self, config):
        """Set optics configuration from dictionary."""
        if 'focal_length' in config:
            self.focal_length_input.setValue(config['focal_length'])
        if 'aperture' in config:
            self.aperture_input.setCurrentText(config['aperture'])
        if 'distortion' in config:
            self.distortion_input.setCurrentText(config['distortion'])
            
        self._calculate_derived_values()
        self._validate_all()
        
    def update_sensor_params(self, pixel_pitch, resolution):
        """Update FOV calculations when sensor parameters change."""
        # Parse resolution
        if 'x' in resolution:
            width, height = resolution.split('x')
            sensor_pixels = int(width)
        else:
            sensor_pixels = 2048  # default
            
        # Recalculate FOV with new sensor parameters
        focal_length = self.focal_length_input.value()  # mm
        sensor_size = sensor_pixels * pixel_pitch * 1e-3  # Convert µm to mm
        fov_rad = 2 * math.atan(sensor_size / (2 * focal_length))
        fov_deg = math.degrees(fov_rad)
        
        self.fov_display.setText(f"{fov_deg:.1f}° × {fov_deg:.1f}°")
        
        # Update angular resolution
        angular_res_rad = (pixel_pitch * 1e-3) / focal_length  # pixel_pitch in µm -> mm
        angular_res_arcsec = math.degrees(angular_res_rad) * 3600
        self.angular_resolution_display.setText(f"{angular_res_arcsec:.1f} arcsec/pixel")