#!/usr/bin/env python3
"""
results_window.py - Results Display Window

Main results window with 3-panel layout for displaying simulation results.
Part of Phase 4 implementation.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QStatusBar, QMessageBox, QLabel, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFont
import logging
from datetime import datetime

# Import panel components
from .widgets.stats_panel import StatsPanel
from .widgets.plot_panel import PlotPanel
from .widgets.export_panel import ExportPanel

logger = logging.getLogger(__name__)


class ResultsWindow(QMainWindow):
    """
    Results display window with 3-panel layout.
    
    Provides:
    - Statistics panel (left) - Performance metrics and indicators
    - Plot panel (center) - Interactive matplotlib visualizations
    - Export panel (right) - Data export functionality
    """
    
    # Signal emitted when window is closed
    window_closed = pyqtSignal()
    
    def __init__(self, results, config, job_id=None, parent=None):
        super().__init__(parent)
        self.results = results
        self.config = config
        self.job_id = job_id or f"Job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.setup_ui()
        self.setup_menu()
        self.load_results()
        
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle(f"Simulation Results - {self.job_id}")
        self.setGeometry(150, 150, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Ultra-compact title section - minimal space
        title_frame = QFrame()
        title_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 2px;
                padding: 2px 6px;
            }
        """)
        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(6, 2, 6, 2)

        # Compact title
        title_label = QLabel("Star Tracker Results")
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2196F3;")

        # Compact job info
        subtitle_label = QLabel(f"{self.job_id}")
        subtitle_label.setStyleSheet("color: #666; font-size: 9px;")

        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(subtitle_label)

        layout.addWidget(title_frame)
        
        # Create 3-panel splitter layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #cccccc;
                width: 2px;
                margin: 2px;
            }
            QSplitter::handle:hover {
                background-color: #2196F3;
            }
        """)
        
        # Left panel: Statistics
        self.stats_panel = StatsPanel()
        stats_frame = self._create_panel_frame("Statistics", self.stats_panel)
        main_splitter.addWidget(stats_frame)
        
        # Center panel: Plots
        self.plot_panel = PlotPanel()
        plots_frame = self._create_panel_frame("Visualizations", self.plot_panel)
        main_splitter.addWidget(plots_frame)
        
        # Right panel: Export
        self.export_panel = ExportPanel()
        export_frame = self._create_panel_frame("Export", self.export_panel)
        main_splitter.addWidget(export_frame)
        
        # Set splitter proportions (25% stats, 50% plots, 25% export)
        main_splitter.setSizes([350, 700, 350])
        
        layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Results loaded successfully")
        
        # Apply window styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
        """)
        
    def _create_panel_frame(self, title, widget):
        """Create a framed panel with title."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("""
            QFrame {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Panel title
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #495057;
                padding: 8px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                margin-bottom: 5px;
            }
        """)
        
        layout.addWidget(title_label)
        layout.addWidget(widget)
        
        return frame
        
    def setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Save results action
        save_action = QAction('Save Results...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.export_panel.quick_save)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Export actions
        export_plots_action = QAction('Export Plots...', self)
        export_plots_action.triggered.connect(self._export_plots)
        file_menu.addAction(export_plots_action)
        
        export_data_action = QAction('Export Data...', self)
        export_data_action.triggered.connect(self._export_data)
        file_menu.addAction(export_data_action)
        
        file_menu.addSeparator()
        
        # Close action
        close_action = QAction('Close Window', self)
        close_action.setShortcut('Ctrl+W')
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        # Refresh action
        refresh_action = QAction('Refresh Plots', self)
        refresh_action.setShortcut('F5')
        refresh_action.triggered.connect(self.plot_panel.refresh_plots)
        view_menu.addAction(refresh_action)
        
        view_menu.addSeparator()
        
        # Panel visibility toggles (could be implemented later)
        show_stats_action = QAction('Show Statistics Panel', self)
        show_stats_action.setCheckable(True)
        show_stats_action.setChecked(True)
        view_menu.addAction(show_stats_action)
        
        show_export_action = QAction('Show Export Panel', self)
        show_export_action.setCheckable(True)
        show_export_action.setChecked(True)
        view_menu.addAction(show_export_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About Results Viewer', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def load_results(self):
        """Load results into all panels."""
        try:
            logger.info(f"Loading results into panels for job {self.job_id}")
            
            # Update statistics panel
            self.stats_panel.update_results(self.results)
            
            # Update plot panel
            self.plot_panel.update_plots(self.results)
            
            # Set results data for export panel
            self.export_panel.set_results_data(self.results)
            
            # Update status
            num_trials = self.results.get('num_trials', 0)
            num_successful = self.results.get('num_successful', 0)
            success_rate = (num_successful / num_trials * 100) if num_trials > 0 else 0
            
            self.status_bar.showMessage(
                f"Results: {num_successful}/{num_trials} trials successful ({success_rate:.1f}%)"
            )
            
            logger.info("Results loaded successfully into all panels")
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            self.status_bar.showMessage(f"Error loading results: {str(e)}")
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load results:\n\n{str(e)}"
            )
            
    def _export_plots(self):
        """Export plots via file dialog."""
        # This could open a specific plot export dialog
        self.export_panel.plots_checkbox.setChecked(True)
        self.export_panel.pdf_checkbox.setChecked(False)
        self.export_panel.csv_checkbox.setChecked(False)
        self.export_panel.config_checkbox.setChecked(False)
        
        if self.export_panel.dir_label.text() == "No directory selected":
            self.export_panel.browse_directory()
            
        if self.export_panel.dir_label.text() != "No directory selected":
            self.export_panel.start_export()
            
    def _export_data(self):
        """Export data via file dialog."""
        self.export_panel.plots_checkbox.setChecked(False)
        self.export_panel.pdf_checkbox.setChecked(True)
        self.export_panel.csv_checkbox.setChecked(True)
        self.export_panel.config_checkbox.setChecked(True)
        
        if self.export_panel.dir_label.text() == "No directory selected":
            self.export_panel.browse_directory()
            
        if self.export_panel.dir_label.text() != "No directory selected":
            self.export_panel.start_export()
            
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Results Viewer",
            "<h3>Star Tracker Results Viewer</h3>"
            "<p>Advanced visualization and analysis tool for star tracker simulation results.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Comprehensive performance statistics</li>"
            "<li>Interactive matplotlib visualizations</li>"
            "<li>Multi-format data export</li>"
            "<li>Professional report generation</li>"
            "</ul>"
            "<br>"
            "<p><b>Version:</b> 1.0.0</p>"
            "<p><b>Author:</b> SOE SaaS Physics Team</p>"
        )
        
    def closeEvent(self, event):
        """Handle window close event."""
        logger.info(f"Closing results window for job {self.job_id}")
        self.window_closed.emit()
        event.accept()
        
    def get_results_summary(self):
        """Get a summary of the results for display."""
        try:
            summary = {
                'job_id': self.job_id,
                'timestamp': datetime.now().isoformat(),
                'trials': self.results.get('num_trials', 0),
                'successful': self.results.get('num_successful', 0),
                'centroid_error': self.results.get('centroid_error_mean', 0),
                'attitude_error': self.results.get('attitude_error_rms', 0),
                'execution_time': self.results.get('execution_time', 0)
            }
            return summary
        except Exception as e:
            logger.error(f"Error generating results summary: {e}")
            return {'error': str(e)}
            
    def update_results(self, new_results, new_config=None):
        """Update the window with new results data."""
        try:
            self.results = new_results
            if new_config:
                self.config = new_config
                
            # Reload all panels
            self.load_results()
            
            # Update window title
            self.setWindowTitle(f"Simulation Results - {self.job_id} (Updated)")
            
            logger.info("Results window updated with new data")
            
        except Exception as e:
            logger.error(f"Error updating results: {e}")
            QMessageBox.critical(
                self, "Update Error",
                f"Failed to update results:\n\n{str(e)}"
            )
            
    def clear_results(self):
        """Clear all results from the panels."""
        self.stats_panel.clear_results()
        self.plot_panel.clear_plots()
        self.export_panel.clear_results()
        self.status_bar.showMessage("Results cleared")
        
    def save_window_state(self):
        """Save window state for restoration."""
        return {
            'geometry': self.geometry(),
            'job_id': self.job_id,
            'timestamp': datetime.now().isoformat()
        }
        
    def restore_window_state(self, state):
        """Restore window state."""
        try:
            if 'geometry' in state:
                self.setGeometry(state['geometry'])
            logger.info("Window state restored")
        except Exception as e:
            logger.warning(f"Could not restore window state: {e}")


def create_results_window(results, config, job_id=None, parent=None):
    """
    Factory function to create a results window.
    
    Args:
        results: Simulation results dictionary
        config: Configuration dictionary
        job_id: Optional job identifier
        parent: Parent widget
        
    Returns:
        ResultsWindow instance
    """
    try:
        window = ResultsWindow(results, config, job_id, parent)
        logger.info(f"Created results window for job {job_id}")
        return window
    except Exception as e:
        logger.error(f"Failed to create results window: {e}")
        return None