#!/usr/bin/env python3
"""
scenario_config.py - Scenario Configuration Tab Widget

Provides simulation scenario parameter configuration interface for the tabbed wizard.
Part of Phase 3 implementation.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QSlider, QLabel, 
    QToolButton, QFrame, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ScenarioConfigWidget(QWidget):
    """
    Scenario configuration tab for star tracker simulation parameters.
    
    Provides input fields for:
    - Star catalog selection
    - Magnitude limit with slider
    - Attitude profile presets
    - Environment selection
    - Monte Carlo trials
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
        title_label = QLabel("3. Scenario Configuration")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #000000;")
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Configure the simulation scenario and environment.")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #666666; margin-bottom: 10px; background-color: transparent;")
        layout.addWidget(desc_label)
        
        # Star catalog group
        catalog_group = QGroupBox("Star Catalog")
        catalog_layout = QFormLayout(catalog_group)
        
        # Star catalog selection
        catalog_row = QHBoxLayout()
        self.catalog_input = QComboBox()
        self.catalog_input.addItems(["Hipparcos", "Gaia DR3", "Custom"])
        self.catalog_input.setCurrentText("Hipparcos")
        self.catalog_input.setMinimumWidth(120)
        
        catalog_info_btn = self._create_info_button(
            "Star catalog to use for simulation. Hipparcos contains ~120,000 stars, "
            "Gaia DR3 has over 1 billion stars with higher precision."
        )
        
        catalog_row.addWidget(self.catalog_input)
        catalog_row.addWidget(catalog_info_btn)
        catalog_row.addWidget(self._create_validation_indicator("catalog"))
        catalog_row.addStretch()
        
        catalog_layout.addRow("Catalog:", catalog_row)
        
        # Magnitude limit with slider
        mag_row = QHBoxLayout()
        self.magnitude_limit_input = QDoubleSpinBox()
        self.magnitude_limit_input.setRange(0.0, 10.0)
        self.magnitude_limit_input.setValue(6.5)
        self.magnitude_limit_input.setDecimals(1)
        self.magnitude_limit_input.setMinimumWidth(100)
        self.magnitude_limit_input.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        
        self.magnitude_slider = QSlider(Qt.Orientation.Horizontal)
        self.magnitude_slider.setRange(0, 100)  # 0.0-10.0 * 10
        self.magnitude_slider.setValue(65)  # 6.5 * 10
        self.magnitude_slider.setMinimumWidth(150)
        
        mag_info_btn = self._create_info_button(
            "Faintest star magnitude to include. Lower values = brighter stars only. "
            "Typical values: 6.5 for naked eye limit, 8.0 for good sensitivity."
        )
        
        mag_row.addWidget(self.magnitude_limit_input)
        mag_row.addWidget(self.magnitude_slider)
        mag_row.addWidget(mag_info_btn)
        mag_row.addWidget(self._create_validation_indicator("magnitude"))
        mag_row.addStretch()
        
        catalog_layout.addRow("Magnitude Limit:", mag_row)
        
        layout.addWidget(catalog_group)
        
        # Attitude and environment group
        attitude_group = QGroupBox("Attitude & Environment")
        attitude_layout = QFormLayout(attitude_group)
        
        # Attitude profile
        attitude_row = QHBoxLayout()
        self.attitude_input = QComboBox()
        self.attitude_input.addItems([
            "Inertial (Fixed)",
            "Slow Rotation (0.1°/s)",
            "Normal Slewing (1°/s)",
            "Fast Slewing (5°/s)",
            "Tumbling Recovery",
            "Sun Avoidance Maneuver"
        ])
        self.attitude_input.setCurrentText("Inertial (Fixed)")
        self.attitude_input.setMinimumWidth(180)
        
        attitude_info_btn = self._create_info_button(
            "Spacecraft attitude motion during simulation. Fixed attitude is simplest, "
            "slewing motion tests tracking during spacecraft maneuvers."
        )
        
        attitude_row.addWidget(self.attitude_input)
        attitude_row.addWidget(attitude_info_btn)
        attitude_row.addWidget(self._create_validation_indicator("attitude"))
        attitude_row.addStretch()
        
        attitude_layout.addRow("Attitude Profile:", attitude_row)
        
        # Environment selection
        env_layout = QVBoxLayout()
        env_label = QLabel("Operating Environment:")
        env_label.setStyleSheet("color: #000000; background-color: transparent;")
        env_layout.addWidget(env_label)
        
        self.environment_group = QButtonGroup()
        self.deep_space_radio = QRadioButton("Deep Space")
        self.deep_space_radio.setToolTip("Minimal radiation, stable thermal environment")
        self.deep_space_radio.setChecked(True)
        
        self.leo_radio = QRadioButton("Low Earth Orbit (LEO)")
        self.leo_radio.setToolTip("Earth radiation belts, thermal cycling, atmospheric drag")
        
        self.geo_radio = QRadioButton("Geostationary Orbit (GEO)")
        self.geo_radio.setToolTip("High radiation environment, eclipse periods")
        
        self.environment_group.addButton(self.deep_space_radio, 0)
        self.environment_group.addButton(self.leo_radio, 1)
        self.environment_group.addButton(self.geo_radio, 2)
        
        env_radio_layout = QHBoxLayout()
        env_radio_layout.addWidget(self.deep_space_radio)
        env_radio_layout.addWidget(self.leo_radio)
        env_radio_layout.addWidget(self.geo_radio)
        env_radio_layout.addStretch()
        
        env_layout.addLayout(env_radio_layout)
        attitude_layout.addRow(env_layout)
        
        layout.addWidget(attitude_group)
        
        # Simulation parameters group
        sim_group = QGroupBox("Simulation Parameters")
        sim_layout = QFormLayout(sim_group)
        
        # Monte Carlo trials
        trials_row = QHBoxLayout()
        self.trials_input = QSpinBox()
        self.trials_input.setRange(100, 5000)
        self.trials_input.setValue(1000)
        self.trials_input.setMinimumWidth(100)
        self.trials_input.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        
        trials_info_btn = self._create_info_button(
            "Number of Monte Carlo trials to run. More trials give better "
            "statistics but take longer to complete."
        )
        
        trials_row.addWidget(self.trials_input)
        trials_row.addWidget(trials_info_btn)
        trials_row.addWidget(self._create_validation_indicator("trials"))
        trials_row.addStretch()
        
        sim_layout.addRow("Monte Carlo Trials:", trials_row)
        
        # PSF file selection (simplified for now)
        psf_row = QHBoxLayout()
        self.psf_input = QComboBox()
        self._populate_psf_files()
        
        psf_info_btn = self._create_info_button(
            "Point Spread Function file to use for simulation. Different files "
            "represent different field angles and optical conditions."
        )
        
        psf_row.addWidget(self.psf_input)
        psf_row.addWidget(psf_info_btn)
        psf_row.addWidget(self._create_validation_indicator("psf"))
        psf_row.addStretch()
        
        sim_layout.addRow("PSF File:", psf_row)
        
        layout.addWidget(sim_group)
        
        # Simulation summary
        summary_group = QGroupBox("Estimated Runtime")
        summary_layout = QVBoxLayout(summary_group)
        
        self.runtime_estimate = QLabel("Estimated time: ~2.5 minutes")
        self.runtime_estimate.setStyleSheet("""
            QLabel {
                font-family: monospace;
                font-size: 12px;
                color: #333333;
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 8px;
            }
        """)
        summary_layout.addWidget(self.runtime_estimate)
        
        layout.addWidget(summary_group)
        
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
                border: 1px solid #cccccc;
                border-radius: 12px;
                background-color: #f0f0f0;
                color: #000000;
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
        indicator.setStyleSheet("color: green; font-weight: bold; font-size: 16px; background-color: transparent;")
        indicator.setMinimumSize(20, 20)
        indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return indicator
        
    def _populate_psf_files(self):
        """Populate PSF file dropdown with available files."""
        psf_dir = Path("data/PSF_sims/Gen_1")
        if psf_dir.exists():
            psf_files = list(psf_dir.glob("*_deg.txt"))
            if psf_files:
                # Sort by field angle
                sorted_files = sorted([f.name for f in psf_files])
                self.psf_input.addItems(sorted_files)
                self.psf_input.setCurrentText("0_deg.txt")  # Default to on-axis
            else:
                self.psf_input.addItem("No PSF files found")
        else:
            self.psf_input.addItem("PSF directory not found")
            
    def setup_connections(self):
        """Setup signal connections."""
        # Connect input changes to validation and estimates
        self.catalog_input.currentTextChanged.connect(self._on_config_changed)
        self.magnitude_limit_input.valueChanged.connect(self._on_config_changed)
        self.magnitude_slider.valueChanged.connect(self._on_magnitude_slider_changed)
        self.attitude_input.currentTextChanged.connect(self._on_config_changed)
        self.environment_group.buttonClicked.connect(self._on_config_changed)
        self.trials_input.valueChanged.connect(self._on_config_changed)
        self.psf_input.currentTextChanged.connect(self._on_config_changed)
        
    def setup_validation(self):
        """Setup initial validation state."""
        self._update_runtime_estimate()
        self._validate_all()
        
    def _on_magnitude_slider_changed(self, value):
        """Handle magnitude slider changes."""
        # Convert slider value (0-100) to actual value (0.0-10.0)
        actual_value = value / 10.0
        self.magnitude_limit_input.setValue(actual_value)
        
    def _on_config_changed(self):
        """Handle configuration changes."""
        # Sync slider with spin box
        self.magnitude_slider.setValue(int(self.magnitude_limit_input.value() * 10))
        
        self._update_runtime_estimate()
        self._validate_all()
        self.config_changed.emit()
        
    def _update_runtime_estimate(self):
        """Update estimated runtime based on current configuration."""
        trials = self.trials_input.value()
        
        # Rough time estimation (this could be made more sophisticated)
        base_time_per_trial = 0.15  # seconds per trial (rough estimate)
        
        # Adjust for environment complexity
        env_id = self.environment_group.checkedId()
        env_multipliers = {0: 1.0, 1: 1.2, 2: 1.1}  # Deep space, LEO, GEO
        env_multiplier = env_multipliers.get(env_id, 1.0)
        
        # Adjust for attitude complexity
        attitude_text = self.attitude_input.currentText()
        if "Fixed" in attitude_text:
            attitude_multiplier = 1.0
        elif "Slow" in attitude_text:
            attitude_multiplier = 1.1
        elif "Normal" in attitude_text:
            attitude_multiplier = 1.3
        elif "Fast" in attitude_text or "Tumbling" in attitude_text:
            attitude_multiplier = 1.5
        else:
            attitude_multiplier = 1.2
            
        total_time = trials * base_time_per_trial * env_multiplier * attitude_multiplier
        
        if total_time < 60:
            time_str = f"~{total_time:.0f} seconds"
        elif total_time < 3600:
            minutes = total_time / 60
            time_str = f"~{minutes:.1f} minutes"
        else:
            hours = total_time / 3600
            time_str = f"~{hours:.1f} hours"
            
        self.runtime_estimate.setText(f"Estimated time: {time_str}")
        
    def _validate_all(self):
        """Validate all input fields."""
        is_valid = True
        
        # Validate each field
        validation_results = {
            'catalog': self._validate_catalog(),
            'magnitude': self._validate_magnitude(),
            'attitude': self._validate_attitude(),
            'trials': self._validate_trials(),
            'psf': self._validate_psf()
        }
        
        # Update validation indicators
        for field, valid in validation_results.items():
            indicator = self.findChild(QLabel, f"validation_{field}")
            if indicator:
                if valid:
                    indicator.setText("✓")
                    indicator.setStyleSheet("color: green; font-weight: bold; font-size: 16px; background-color: transparent;")
                else:
                    indicator.setText("✗")
                    indicator.setStyleSheet("color: red; font-weight: bold; font-size: 16px; background-color: transparent;")
                    is_valid = False
        
        # Update validation summary
        if is_valid:
            self.validation_summary.setText("✅ Scenario configuration is valid")
            self.validation_summary.setStyleSheet("color: green; font-weight: bold; padding: 10px; background-color: transparent;")
        else:
            self.validation_summary.setText("⚠️ Please check highlighted fields")
            self.validation_summary.setStyleSheet("color: orange; font-weight: bold; padding: 10px; background-color: transparent;")
            
        self.validation_changed.emit(is_valid)
        
    def _validate_catalog(self):
        """Validate catalog selection."""
        return self.catalog_input.currentText() in ["Hipparcos", "Gaia DR3", "Custom"]
        
    def _validate_magnitude(self):
        """Validate magnitude limit value."""
        value = self.magnitude_limit_input.value()
        return 0.0 <= value <= 10.0
        
    def _validate_attitude(self):
        """Validate attitude selection."""
        return self.attitude_input.currentText() != ""
        
    def _validate_trials(self):
        """Validate trials value."""
        value = self.trials_input.value()
        return 100 <= value <= 5000
        
    def _validate_psf(self):
        """Validate PSF file selection."""
        return (self.psf_input.currentText() != "" and 
                "not found" not in self.psf_input.currentText())
        
    def is_valid(self):
        """Check if all configuration is valid."""
        return (self._validate_catalog() and 
                self._validate_magnitude() and
                self._validate_attitude() and
                self._validate_trials() and
                self._validate_psf())
                
    def get_config(self):
        """Get current scenario configuration."""
        return {
            'catalog': self.catalog_input.currentText(),
            'magnitude_limit': self.magnitude_limit_input.value(),
            'attitude_profile': self.attitude_input.currentText(),
            'environment': self._get_selected_environment(),
            'trials': self.trials_input.value(),
            'psf_file': self.psf_input.currentText(),
            'estimated_runtime': self.runtime_estimate.text()
        }
        
    def _get_selected_environment(self):
        """Get selected environment."""
        env_id = self.environment_group.checkedId()
        environments = {0: "Deep Space", 1: "LEO", 2: "GEO"}
        return environments.get(env_id, "Deep Space")
        
    def set_config(self, config):
        """Set scenario configuration from dictionary."""
        if 'catalog' in config:
            self.catalog_input.setCurrentText(config['catalog'])
        if 'magnitude_limit' in config:
            self.magnitude_limit_input.setValue(config['magnitude_limit'])
        if 'attitude_profile' in config:
            self.attitude_input.setCurrentText(config['attitude_profile'])
        if 'environment' in config:
            env_name = config['environment']
            if env_name == "Deep Space":
                self.deep_space_radio.setChecked(True)
            elif env_name == "LEO":
                self.leo_radio.setChecked(True)
            elif env_name == "GEO":
                self.geo_radio.setChecked(True)
        if 'trials' in config:
            self.trials_input.setValue(config['trials'])
        if 'psf_file' in config:
            self.psf_input.setCurrentText(config['psf_file'])
            
        self._update_runtime_estimate()
        self._validate_all()