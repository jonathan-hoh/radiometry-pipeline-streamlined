"""
progress_panel.py - Progress tracking and metrics display widget

Provides real-time progress updates and live metrics during simulation execution.
"""

import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QProgressBar, QPushButton, QTextEdit, QGroupBox)
from PyQt6.QtCore import pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import time

logger = logging.getLogger(__name__)


class ProgressPanel(QWidget):
    """Widget for displaying simulation progress and live metrics."""
    
    # Signal emitted when user requests to cancel simulation
    cancel_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the progress panel UI components."""
        layout = QVBoxLayout(self)
        
        # Progress section
        progress_group = QGroupBox("Simulation Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Status label
        self.status_label = QLabel("Ready to start simulation")
        self.status_label.setFont(QFont("Arial", 10))
        progress_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel Simulation")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        progress_layout.addLayout(button_layout)
        
        layout.addWidget(progress_group)
        
        # Metrics section
        metrics_group = QGroupBox("Live Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # Create metrics display grid
        self.metrics_layout = QVBoxLayout()
        self.metrics_labels = {}
        
        # Initialize common metric labels
        metric_names = [
            ("trials_completed", "Trials Completed"),
            ("elapsed_time", "Elapsed Time"),
            ("estimated_time_remaining", "Est. Time Remaining"),
            ("avg_time_per_trial", "Avg. Time/Trial")
        ]
        
        for key, display_name in metric_names:
            metric_row = QHBoxLayout()
            label = QLabel(f"{display_name}:")
            label.setMinimumWidth(150)
            value_label = QLabel("--")
            value_label.setFont(QFont("Courier", 9))  # Monospace for numbers
            
            metric_row.addWidget(label)
            metric_row.addWidget(value_label)
            metric_row.addStretch()
            
            self.metrics_labels[key] = value_label
            self.metrics_layout.addLayout(metric_row)
            
        metrics_layout.addLayout(self.metrics_layout)
        layout.addWidget(metrics_group)
        
        # Detailed log section (collapsible)
        log_group = QGroupBox("Detailed Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(100)
        self.log_display.setFont(QFont("Courier", 8))
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        
        layout.addWidget(log_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
    def start_simulation(self):
        """Called when simulation starts."""
        self.start_time = time.time()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting simulation...")
        self.cancel_button.setEnabled(True)
        self.log_display.clear()
        self._reset_metrics()
        self._log_message("Simulation started")
        
    def update_progress(self, percentage, status_message):
        """Update progress bar and status."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(status_message)
        self._log_message(f"[{percentage:3d}%] {status_message}")
        
    def update_metrics(self, metrics_dict):
        """Update live metrics display."""
        for key, value in metrics_dict.items():
            if key in self.metrics_labels:
                if isinstance(value, (int, float)):
                    if key == "trials_completed":
                        total = metrics_dict.get('total_trials', '?')
                        display_value = f"{value} / {total}"
                    elif 'time' in key and isinstance(value, (int, float)):
                        display_value = f"{value:.1f}s"
                    else:
                        display_value = str(value)
                else:
                    display_value = str(value)
                    
                self.metrics_labels[key].setText(display_value)
                
    def simulation_finished(self, success=True, message=""):
        """Called when simulation finishes."""
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText("Simulation completed successfully!")
            self._log_message("✅ Simulation completed successfully")
        else:
            self.status_label.setText(f"Simulation failed: {message}")
            self._log_message(f"❌ Simulation failed: {message}")
            
        self.cancel_button.setEnabled(False)
        
        # Calculate total time
        if self.start_time:
            total_time = time.time() - self.start_time
            self.metrics_labels['elapsed_time'].setText(f"{total_time:.1f}s")
            self._log_message(f"Total execution time: {total_time:.1f}s")
            
    def simulation_cancelled(self):
        """Called when simulation is cancelled."""
        self.status_label.setText("Simulation cancelled by user")
        self.cancel_button.setEnabled(False)
        self._log_message("⚠️ Simulation cancelled by user")
        
    def _reset_metrics(self):
        """Reset all metric displays to default values."""
        for label in self.metrics_labels.values():
            label.setText("--")
            
    def _log_message(self, message):
        """Add a timestamped message to the detailed log."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_display.append(log_entry)
        
        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def get_current_status(self):
        """Get current progress status for external queries."""
        return {
            'progress_percent': self.progress_bar.value(),
            'status_message': self.status_label.text(),
            'is_running': self.cancel_button.isEnabled(),
            'start_time': self.start_time
        }