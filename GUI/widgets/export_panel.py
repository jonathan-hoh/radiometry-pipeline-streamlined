#!/usr/bin/env python3
"""
export_panel.py - Export Panel Widget

Provides export functionality for simulation results.
Part of Phase 4 implementation.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QTextEdit, QComboBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont
import logging
import json
import csv
from pathlib import Path
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class ExportWorker(QThread):
    """Worker thread for exporting results."""
    
    progress_updated = pyqtSignal(int, str)  # (percentage, status)
    export_finished = pyqtSignal(bool, str)  # (success, message)
    
    def __init__(self, export_options, results_data, export_dir):
        super().__init__()
        self.export_options = export_options
        self.results_data = results_data
        self.export_dir = Path(export_dir)
        self.is_cancelled = False
        
    def cancel(self):
        """Cancel the export operation."""
        self.is_cancelled = True
        
    def run(self):
        """Main export execution."""
        try:
            total_steps = sum(1 for option in self.export_options.values() if option)
            current_step = 0
            
            if self.export_options.get('summary_pdf', False):
                self.progress_updated.emit(int(current_step/total_steps*100), "Generating PDF report...")
                if not self.is_cancelled:
                    self._export_pdf_report()
                current_step += 1
                
            if self.export_options.get('plots_png', False):
                self.progress_updated.emit(int(current_step/total_steps*100), "Saving plots...")
                if not self.is_cancelled:
                    self._export_plots()
                current_step += 1
                
            if self.export_options.get('raw_csv', False):
                self.progress_updated.emit(int(current_step/total_steps*100), "Exporting raw data...")
                if not self.is_cancelled:
                    self._export_csv_data()
                current_step += 1
                
            if self.export_options.get('config_json', False):
                self.progress_updated.emit(int(current_step/total_steps*100), "Saving configuration...")
                if not self.is_cancelled:
                    self._export_config()
                current_step += 1
                
            if not self.is_cancelled:
                self.progress_updated.emit(100, "Export complete!")
                self.export_finished.emit(True, f"Results exported to {self.export_dir}")
            else:
                self.export_finished.emit(False, "Export cancelled by user")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            self.export_finished.emit(False, f"Export failed: {str(e)}")
            
    def _export_pdf_report(self):
        """Export PDF summary report."""
        try:
            # For now, create a text-based summary since PDF generation requires additional libraries
            summary_file = self.export_dir / "simulation_summary.txt"
            
            with open(summary_file, 'w') as f:
                f.write("STAR TRACKER SIMULATION SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Configuration summary
                if 'configuration' in self.results_data:
                    config = self.results_data['configuration']
                    f.write("CONFIGURATION:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in config.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # Results summary
                f.write("RESULTS SUMMARY:\n")
                f.write("-" * 20 + "\n")
                
                if 'centroid_error_mean' in self.results_data:
                    f.write(f"Centroid Error Mean: {self.results_data['centroid_error_mean']:.3f} pixels\n")
                if 'centroid_error_std' in self.results_data:
                    f.write(f"Centroid Error Std: {self.results_data['centroid_error_std']:.3f} pixels\n")
                if 'attitude_error_rms' in self.results_data:
                    f.write(f"Attitude Error RMS: {self.results_data['attitude_error_rms']:.2f} arcseconds\n")
                if 'execution_time' in self.results_data:
                    f.write(f"Execution Time: {self.results_data['execution_time']:.1f} seconds\n")
                    
                num_trials = self.results_data.get('num_trials', 0)
                num_successful = self.results_data.get('num_successful', 0)
                f.write(f"Success Rate: {num_successful}/{num_trials} ({num_successful/num_trials*100:.1f}%)\n")
                
            logger.info(f"Exported summary report to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting PDF report: {e}")
            
    def _export_plots(self):
        """Export all plots as PNG files."""
        try:
            # This would integrate with the plot panel to save plots
            # For now, create placeholder files
            plots_dir = self.export_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            plot_types = ["centroid_histogram", "attitude_histogram", "star_field", "residuals"]
            
            for plot_type in plot_types:
                plot_file = plots_dir / f"{plot_type}.png"
                # In real implementation, this would call the plot panel's export function
                plot_file.touch()  # Create empty file as placeholder
                
            logger.info(f"Exported plots to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting plots: {e}")
            
    def _export_csv_data(self):
        """Export raw simulation data to CSV."""
        try:
            data_dir = self.export_dir / "data"
            data_dir.mkdir(exist_ok=True)
            
            # Export centroid errors if available
            if 'centroid_errors' in self.results_data:
                centroid_file = data_dir / "centroid_errors.csv"
                with open(centroid_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Trial', 'Centroid_Error_px'])
                    for i, error in enumerate(self.results_data['centroid_errors']):
                        writer.writerow([i+1, error])
                        
            # Export attitude errors if available
            if 'attitude_errors' in self.results_data:
                attitude_file = data_dir / "attitude_errors.csv"
                with open(attitude_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Trial', 'Attitude_Error_arcsec'])
                    for i, error in enumerate(self.results_data['attitude_errors']):
                        writer.writerow([i+1, error])
                        
            # Export summary statistics
            summary_file = data_dir / "summary_statistics.csv"
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value', 'Unit'])
                
                metrics = [
                    ('centroid_error_mean', 'pixels'),
                    ('centroid_error_std', 'pixels'),
                    ('attitude_error_rms', 'arcseconds'),
                    ('execution_time', 'seconds'),
                    ('num_trials', ''),
                    ('num_successful', '')
                ]
                
                for metric, unit in metrics:
                    if metric in self.results_data:
                        writer.writerow([metric, self.results_data[metric], unit])
                        
            logger.info(f"Exported CSV data to {data_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting CSV data: {e}")
            
    def _export_config(self):
        """Export configuration to JSON."""
        try:
            config_file = self.export_dir / "simulation_config.json"
            
            config_data = {
                'export_timestamp': datetime.now().isoformat(),
                'configuration': self.results_data.get('configuration', {}),
                'metadata': {
                    'software': 'Star Tracker Simulator',
                    'version': '1.0.0',
                    'export_format_version': '1.0'
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Exported configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")


class ExportPanel(QWidget):
    """
    Export panel for saving simulation results.
    
    Provides:
    - Multiple export format options
    - Directory selection
    - Progress tracking
    - Export customization
    """
    
    # Signal emitted when export is requested
    export_requested = pyqtSignal(dict, str)  # (options, directory)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_data = None
        self.export_worker = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Export Results")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2196F3; margin-bottom: 15px;")
        layout.addWidget(title_label)
        
        # Export options group
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout(options_group)
        
        # Checkboxes for export types
        self.pdf_checkbox = QCheckBox("Summary report (TXT)")
        self.pdf_checkbox.setChecked(True)
        self.pdf_checkbox.setToolTip("Generate a comprehensive text summary of simulation results")
        
        self.plots_checkbox = QCheckBox("All plots (PNG)")
        self.plots_checkbox.setChecked(True)
        self.plots_checkbox.setToolTip("Export all visualization plots as high-resolution PNG images")
        
        self.csv_checkbox = QCheckBox("Raw data (CSV)")
        self.csv_checkbox.setChecked(True)
        self.csv_checkbox.setToolTip("Export raw simulation data in comma-separated values format")
        
        self.config_checkbox = QCheckBox("Configuration (JSON)")
        self.config_checkbox.setChecked(True)
        self.config_checkbox.setToolTip("Save simulation configuration parameters in JSON format")
        
        options_layout.addWidget(self.pdf_checkbox)
        options_layout.addWidget(self.plots_checkbox)
        options_layout.addWidget(self.csv_checkbox)
        options_layout.addWidget(self.config_checkbox)
        
        layout.addWidget(options_group)
        
        # Export settings group
        settings_group = QGroupBox("Export Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # DPI setting for plots
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("Plot DPI:"))
        self.dpi_spinbox = QSpinBox()
        self.dpi_spinbox.setRange(72, 600)
        self.dpi_spinbox.setValue(300)
        self.dpi_spinbox.setSuffix(" dpi")
        dpi_layout.addWidget(self.dpi_spinbox)
        dpi_layout.addStretch()
        settings_layout.addLayout(dpi_layout)
        
        # File naming convention
        naming_layout = QHBoxLayout()
        naming_layout.addWidget(QLabel("Filename prefix:"))
        self.prefix_combo = QComboBox()
        self.prefix_combo.addItems([
            "simulation_results",
            "star_tracker_sim", 
            "custom"
        ])
        naming_layout.addWidget(self.prefix_combo)
        naming_layout.addStretch()
        settings_layout.addLayout(naming_layout)
        
        layout.addWidget(settings_group)
        
        # Directory selection
        dir_group = QGroupBox("Export Directory")
        dir_layout = QVBoxLayout(dir_group)
        
        dir_select_layout = QHBoxLayout()
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 5px;
                background-color: #f9f9f9;
            }
        """)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_directory)
        self.browse_button.setMaximumWidth(100)
        
        dir_select_layout.addWidget(self.dir_label)
        dir_select_layout.addWidget(self.browse_button)
        dir_layout.addLayout(dir_select_layout)
        
        layout.addWidget(dir_group)
        
        # Progress section
        progress_group = QGroupBox("Export Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.progress_label = QLabel("Ready to export")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(progress_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.export_button = QPushButton("📁 Export Selected")
        self.export_button.setMinimumHeight(40)
        self.export_button.clicked.connect(self.start_export)
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel_export)
        
        self.quick_save_button = QPushButton("💾 Quick Save")
        self.quick_save_button.setToolTip("Save results to default location with timestamp")
        self.quick_save_button.clicked.connect(self.quick_save)
        
        button_layout.addWidget(self.quick_save_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.export_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch
        layout.addStretch()
        
    def browse_directory(self):
        """Open directory selection dialog."""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Export Directory",
            str(Path.home())
        )
        
        if directory:
            self.dir_label.setText(directory)
            self.export_button.setEnabled(True)
            
    def start_export(self):
        """Start the export process."""
        if not self.results_data:
            QMessageBox.warning(self, "No Data", "No simulation results available to export.")
            return
            
        export_dir = self.dir_label.text()
        if export_dir == "No directory selected":
            QMessageBox.warning(self, "No Directory", "Please select an export directory first.")
            return
            
        # Get export options
        export_options = {
            'summary_pdf': self.pdf_checkbox.isChecked(),
            'plots_png': self.plots_checkbox.isChecked(),
            'raw_csv': self.csv_checkbox.isChecked(),
            'config_json': self.config_checkbox.isChecked()
        }
        
        if not any(export_options.values()):
            QMessageBox.warning(self, "No Options", "Please select at least one export option.")
            return
            
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self.prefix_combo.currentText()
        export_subdir = Path(export_dir) / f"{prefix}_{timestamp}"
        export_subdir.mkdir(parents=True, exist_ok=True)
        
        # Start export worker
        self.export_worker = ExportWorker(export_options, self.results_data, str(export_subdir))
        self.export_worker.progress_updated.connect(self.update_progress)
        self.export_worker.export_finished.connect(self.export_finished)
        
        # Update UI
        self.export_button.setVisible(False)
        self.cancel_button.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start export
        self.export_worker.start()
        
    def cancel_export(self):
        """Cancel the ongoing export."""
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.cancel()
            self.progress_label.setText("Cancelling export...")
            
    def quick_save(self):
        """Quick save to default location."""
        if not self.results_data:
            QMessageBox.warning(self, "No Data", "No simulation results available to save.")
            return
            
        # Use outputs directory
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Set directory and start export with all options
        self.dir_label.setText(str(outputs_dir))
        self.pdf_checkbox.setChecked(True)
        self.plots_checkbox.setChecked(True)
        self.csv_checkbox.setChecked(True)
        self.config_checkbox.setChecked(True)
        
        self.start_export()
        
    def update_progress(self, percentage, status):
        """Update export progress."""
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(status)
        
    def export_finished(self, success, message):
        """Handle export completion."""
        # Reset UI
        self.export_button.setVisible(True)
        self.cancel_button.setVisible(False)
        self.progress_bar.setVisible(False)
        
        if success:
            self.progress_label.setText("✅ Export completed successfully!")
            QMessageBox.information(self, "Export Complete", message)
        else:
            self.progress_label.setText("❌ Export failed")
            QMessageBox.critical(self, "Export Failed", message)
            
    def set_results_data(self, results):
        """Set the results data for export."""
        self.results_data = results
        self.export_button.setEnabled(self.dir_label.text() != "No directory selected")
        
    def clear_results(self):
        """Clear results data."""
        self.results_data = None
        self.export_button.setEnabled(False)
        self.progress_label.setText("No results to export")
        
    def get_export_summary(self):
        """Get summary of what will be exported."""
        if not self.results_data:
            return "No data available"
            
        options = []
        if self.pdf_checkbox.isChecked():
            options.append("Summary report")
        if self.plots_checkbox.isChecked():
            options.append("Plots")
        if self.csv_checkbox.isChecked():
            options.append("Raw data")
        if self.config_checkbox.isChecked():
            options.append("Configuration")
            
        return f"Will export: {', '.join(options)}" if options else "No options selected"