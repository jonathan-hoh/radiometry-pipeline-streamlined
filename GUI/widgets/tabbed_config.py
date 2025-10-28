#!/usr/bin/env python3
"""
tabbed_config.py - Tabbed Configuration Widget

Provides the main tabbed wizard interface with navigation and validation.
Part of Phase 3 implementation.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QLabel, QFrame, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import logging

# Import the tab widgets
from .sensor_config import SensorConfigWidget
from .optics_config import OpticsConfigWidget
from .scenario_config import ScenarioConfigWidget
from ..utils.validator import ConfigValidator

logger = logging.getLogger(__name__)


class TabbedConfigWidget(QWidget):
    """
    Main tabbed configuration widget with wizard-style navigation.
    
    Provides:
    - Three configuration tabs (Sensor, Optics, Scenario)
    - Navigation buttons with validation
    - Overall validation status
    - Run simulation signal when ready
    """
    
    # Signal emitted when ready to run simulation
    run_simulation_requested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.setup_validation()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Star Tracker Simulation Configuration Wizard")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2196F3; margin-bottom: 10px; background-color: transparent;")
        layout.addWidget(title_label)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: white;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                color: #000000;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 150px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #000000;
                border-color: #2196F3;
                border-bottom: 2px solid white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e0e0e0;
                color: #000000;
            }
        """)
        
        # Create and add tabs
        self.sensor_tab = SensorConfigWidget()
        self.optics_tab = OpticsConfigWidget()
        self.scenario_tab = ScenarioConfigWidget()
        
        self.tab_widget.addTab(self.sensor_tab, "1. Sensor")
        self.tab_widget.addTab(self.optics_tab, "2. Optics") 
        self.tab_widget.addTab(self.scenario_tab, "3. Scenario")
        
        # Add tab indicators
        self._update_tab_indicators()
        
        layout.addWidget(self.tab_widget)
        
        # Navigation and validation section
        nav_frame = QFrame()
        nav_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #f8f9fa;
                color: #000000;
                padding: 10px;
            }
        """)
        nav_layout = QVBoxLayout(nav_frame)
        
        # Overall validation status
        self.overall_validation = QLabel()
        self.overall_validation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overall_validation.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px; background-color: transparent;")
        nav_layout.addWidget(self.overall_validation)
        
        # Progress indicator
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Progress:")
        progress_label.setStyleSheet("color: #000000; background-color: transparent;")
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 3)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 2px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        nav_layout.addLayout(progress_layout)
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        self.back_button = QPushButton("◀ Back")
        self.back_button.setMinimumWidth(100)
        self.back_button.setEnabled(False)
        self.back_button.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #f0f0f0;
                color: #000000;
            }
            QPushButton:hover:enabled {
                background-color: #e0e0e0;
                color: #000000;
            }
            QPushButton:disabled {
                color: #999999;
                background-color: #f0f0f0;
            }
        """)
        
        self.next_button = QPushButton("Next ▶")
        self.next_button.setMinimumWidth(100)
        self.next_button.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #2196F3;
                border-radius: 4px;
                background-color: #2196F3;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #cccccc;
                color: #999999;
            }
        """)
        
        self.run_button = QPushButton("🚀 Run Simulation")
        self.run_button.setMinimumWidth(150)
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                border: 2px solid #4CAF50;
                border-radius: 6px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover:enabled {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #cccccc;
                color: #999999;
            }
        """)
        
        button_layout.addWidget(self.back_button)
        button_layout.addStretch()
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.run_button)
        
        nav_layout.addLayout(button_layout)
        layout.addWidget(nav_frame)
        
    def setup_connections(self):
        """Setup signal connections."""
        # Tab change handling
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        
        # Navigation buttons
        self.back_button.clicked.connect(self._go_back)
        self.next_button.clicked.connect(self._go_next)
        self.run_button.clicked.connect(self._run_simulation)
        
        # Configuration change handling
        self.sensor_tab.config_changed.connect(self._on_config_changed)
        self.sensor_tab.validation_changed.connect(self._on_validation_changed)
        
        self.optics_tab.config_changed.connect(self._on_config_changed)
        self.optics_tab.validation_changed.connect(self._on_validation_changed)
        
        self.scenario_tab.config_changed.connect(self._on_config_changed)
        self.scenario_tab.validation_changed.connect(self._on_validation_changed)
        
        # Cross-tab updates (optics FOV depends on sensor params)
        self.sensor_tab.config_changed.connect(self._update_cross_dependencies)
        
    def setup_validation(self):
        """Setup initial validation state."""
        self._validate_all_tabs()
        self._update_navigation_state()
        
    def _on_tab_changed(self, index):
        """Handle tab change."""
        self._update_navigation_state()
        self._update_progress()
        logger.info(f"Switched to tab {index}")
        
    def _go_back(self):
        """Go to previous tab."""
        current_index = self.tab_widget.currentIndex()
        if current_index > 0:
            self.tab_widget.setCurrentIndex(current_index - 1)
            
    def _go_next(self):
        """Go to next tab."""
        current_index = self.tab_widget.currentIndex()
        if current_index < self.tab_widget.count() - 1:
            self.tab_widget.setCurrentIndex(current_index + 1)
            
    def _run_simulation(self):
        """Emit signal to run simulation with current configuration."""
        if self._validate_complete_configuration():
            config = self.get_complete_configuration()
            logger.info("Running simulation with tabbed configuration")
            self.run_simulation_requested.emit(config)
        else:
            QMessageBox.warning(
                self, "Configuration Invalid",
                "Please complete and validate all configuration tabs before running the simulation."
            )
            
    def _on_config_changed(self):
        """Handle configuration changes."""
        self._validate_all_tabs()
        self._update_navigation_state()
        
    def _on_validation_changed(self, is_valid):
        """Handle validation state changes."""
        self._validate_all_tabs()
        self._update_navigation_state()
        
    def _update_cross_dependencies(self):
        """Update cross-dependencies between tabs."""
        # Update optics FOV calculation when sensor params change
        sensor_config = self.sensor_tab.get_config()
        self.optics_tab.update_sensor_params(
            sensor_config['pixel_pitch'],
            sensor_config['resolution']
        )
        
    def _validate_all_tabs(self):
        """Validate all tabs and update overall status."""
        sensor_valid = self.sensor_tab.is_valid()
        optics_valid = self.optics_tab.is_valid()
        scenario_valid = self.scenario_tab.is_valid()
        
        # Count valid tabs
        valid_count = sum([sensor_valid, optics_valid, scenario_valid])
        self._update_progress_value(valid_count)
        
        # Update overall validation display
        if valid_count == 3:
            self.overall_validation.setText("✅ All configurations valid - Ready to simulate!")
            self.overall_validation.setStyleSheet("color: green; font-weight: bold; font-size: 14px; background-color: transparent;")
            
            # Run complete cross-validation
            if self._validate_complete_configuration():
                self.run_button.setEnabled(True)
            else:
                self.run_button.setEnabled(False)
                self.overall_validation.setText("⚠️ Configuration compatibility issues detected")
                self.overall_validation.setStyleSheet("color: orange; font-weight: bold; font-size: 14px; background-color: transparent;")
        else:
            invalid_tabs = []
            if not sensor_valid:
                invalid_tabs.append("Sensor")
            if not optics_valid:
                invalid_tabs.append("Optics")
            if not scenario_valid:
                invalid_tabs.append("Scenario")
                
            self.overall_validation.setText(f"⚠️ Incomplete: {', '.join(invalid_tabs)}")
            self.overall_validation.setStyleSheet("color: orange; font-weight: bold; font-size: 14px; background-color: transparent;")
            self.run_button.setEnabled(False)
            
    def _validate_complete_configuration(self):
        """Validate complete configuration across all tabs."""
        try:
            sensor_config = self.sensor_tab.get_config()
            optics_config = self.optics_tab.get_config()
            scenario_config = self.scenario_tab.get_config()
            
            is_valid, error_msg = ConfigValidator.validate_complete_config(
                sensor_config, optics_config, scenario_config
            )
            
            if not is_valid:
                logger.warning(f"Complete configuration validation failed: {error_msg}")
                
            return is_valid
            
        except Exception as e:
            logger.error(f"Error in complete configuration validation: {e}")
            return False
            
    def _update_navigation_state(self):
        """Update navigation button states."""
        current_index = self.tab_widget.currentIndex()
        tab_count = self.tab_widget.count()
        
        # Back button
        self.back_button.setEnabled(current_index > 0)
        
        # Next button
        self.next_button.setEnabled(current_index < tab_count - 1)
        
        # Update next button text based on current tab
        if current_index == tab_count - 1:
            self.next_button.setText("Complete")
        else:
            self.next_button.setText("Next ▶")
            
    def _update_progress(self):
        """Update progress indicator."""
        current_index = self.tab_widget.currentIndex()
        self.progress_bar.setFormat(f"Step {current_index + 1} of 3")
        
    def _update_progress_value(self, valid_count):
        """Update progress bar value based on valid tabs."""
        self.progress_bar.setValue(valid_count)
        if valid_count == 3:
            self.progress_bar.setFormat("Configuration Complete!")
        else:
            self.progress_bar.setFormat(f"{valid_count} of 3 sections complete")
            
    def _update_tab_indicators(self):
        """Update tab titles with validation indicators."""
        # This will be called after validation to update tab titles
        # For now, keep it simple - indicators are shown within each tab
        pass
        
    def get_complete_configuration(self):
        """Get complete configuration from all tabs."""
        config = {
            'sensor': self.sensor_tab.get_config(),
            'optics': self.optics_tab.get_config(),
            'scenario': self.scenario_tab.get_config()
        }
        
        # Add some derived values for simulation
        config['combined'] = {
            'focal_length': config['optics']['focal_length'],
            'pixel_pitch': config['sensor']['pixel_pitch'],
            'resolution': config['sensor']['resolution'],
            'magnitude': config['scenario']['magnitude_limit'],
            'num_trials': config['scenario']['trials'],
            'psf_file': f"data/PSF_sims/Gen_1/{config['scenario']['psf_file']}"
        }
        
        return config
        
    def set_complete_configuration(self, config):
        """Set complete configuration across all tabs."""
        if 'sensor' in config:
            self.sensor_tab.set_config(config['sensor'])
        if 'optics' in config:
            self.optics_tab.set_config(config['optics'])
        if 'scenario' in config:
            self.scenario_tab.set_config(config['scenario'])
            
        self._validate_all_tabs()
        self._update_navigation_state()
        
    def get_current_tab_name(self):
        """Get name of currently active tab."""
        current_index = self.tab_widget.currentIndex()
        tab_names = ["Sensor", "Optics", "Scenario"]
        return tab_names[current_index] if 0 <= current_index < len(tab_names) else "Unknown"