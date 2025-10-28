#!/usr/bin/env python3
"""
plot_panel.py - Plot Panel Widget

Provides matplotlib-based plotting for simulation results.
Part of Phase 4 implementation.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QComboBox, QLabel, QFrame, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import logging
import numpy as np

# Matplotlib imports
try:
    import matplotlib
    matplotlib.use('Qt5Agg')  # Use Qt backend
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')  # Use seaborn style
except ImportError:
    # Fallback if seaborn style is not available
    try:
        plt.style.use('ggplot')
    except:
        pass  # Use default style
except ImportError as e:
    logger.error(f"Failed to import matplotlib: {e}")
    FigureCanvas = None
    NavigationToolbar = None

logger = logging.getLogger(__name__)


class PlotCanvas(FigureCanvas):
    """Custom matplotlib canvas for Qt integration."""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)
        
        # Set tight layout
        self.figure.tight_layout(pad=3.0)
        
        # Clear initial plot
        self.clear_plot()
        
    def clear_plot(self):
        """Clear the current plot."""
        self.figure.clear()
        self.draw()
        
    def plot_centroid_histogram(self, errors, title="Centroid Error Distribution"):
        """Plot centroid error histogram."""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            if len(errors) > 0:
                # Create histogram
                n, bins, patches = ax.hist(errors, bins=30, alpha=0.7, color='skyblue', 
                                         edgecolor='black', linewidth=0.5)
                
                # Add statistics
                mean_err = np.mean(errors)
                std_err = np.std(errors)
                
                # Add vertical lines for statistics
                ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_err:.3f} px')
                ax.axvline(mean_err + std_err, color='orange', linestyle='--', linewidth=1, 
                          label=f'+1σ: {mean_err + std_err:.3f} px')
                ax.axvline(mean_err - std_err, color='orange', linestyle='--', linewidth=1, 
                          label=f'-1σ: {mean_err - std_err:.3f} px')
                
                ax.set_xlabel('Centroid Error (pixels)')
                ax.set_ylabel('Frequency')
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(title)
                
            self.figure.tight_layout()
            self.draw()
            
        except Exception as e:
            logger.error(f"Error plotting centroid histogram: {e}")
            
    def plot_attitude_histogram(self, errors, title="Attitude Error Distribution"):
        """Plot attitude error histogram."""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            if len(errors) > 0:
                # Create histogram
                n, bins, patches = ax.hist(errors, bins=30, alpha=0.7, color='lightgreen', 
                                         edgecolor='black', linewidth=0.5)
                
                # Add statistics
                mean_err = np.mean(errors)
                std_err = np.std(errors)
                
                # Add vertical lines for statistics
                ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_err:.2f} arcsec')
                ax.axvline(mean_err + 3*std_err, color='orange', linestyle=':', linewidth=2, 
                          label=f'3σ bound: {mean_err + 3*std_err:.2f} arcsec')
                
                ax.set_xlabel('Attitude Error (arcseconds)')
                ax.set_ylabel('Frequency')
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(title)
                
            self.figure.tight_layout()
            self.draw()
            
        except Exception as e:
            logger.error(f"Error plotting attitude histogram: {e}")
            
    def plot_star_field(self, stars_x, stars_y, detected_x=None, detected_y=None, 
                       title="Star Field Visualization"):
        """Plot star field with detections."""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot all stars
            if len(stars_x) > 0 and len(stars_y) > 0:
                ax.scatter(stars_x, stars_y, c='blue', s=20, alpha=0.6, 
                          label=f'Catalog Stars ({len(stars_x)})')
                
                # Plot detected stars if provided
                if detected_x is not None and detected_y is not None:
                    ax.scatter(detected_x, detected_y, c='red', s=30, marker='+', 
                              linewidth=2, label=f'Detected ({len(detected_x)})')
                
                ax.set_xlabel('X Position (pixels)')
                ax.set_ylabel('Y Position (pixels)')
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
            else:
                ax.text(0.5, 0.5, 'No star field data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(title)
                
            self.figure.tight_layout()
            self.draw()
            
        except Exception as e:
            logger.error(f"Error plotting star field: {e}")
            
    def plot_residuals(self, x_residuals, y_residuals, title="Centroiding Residuals"):
        """Plot centroiding residuals."""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            if len(x_residuals) > 0 and len(y_residuals) > 0:
                # Create scatter plot
                ax.scatter(x_residuals, y_residuals, alpha=0.6, s=20)
                
                # Add circle for RMS error
                rms_error = np.sqrt(np.mean(x_residuals**2 + y_residuals**2))
                circle = plt.Circle((0, 0), rms_error, fill=False, color='red', 
                                  linestyle='--', linewidth=2, 
                                  label=f'RMS: {rms_error:.3f} px')
                ax.add_patch(circle)
                
                # Set equal aspect ratio and center on origin
                max_range = max(np.max(np.abs(x_residuals)), np.max(np.abs(y_residuals)))
                ax.set_xlim(-max_range*1.1, max_range*1.1)
                ax.set_ylim(-max_range*1.1, max_range*1.1)
                
                ax.set_xlabel('X Residual (pixels)')
                ax.set_ylabel('Y Residual (pixels)')
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                
                # Add crosshairs at origin
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No residual data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(title)
                
            self.figure.tight_layout()
            self.draw()
            
        except Exception as e:
            logger.error(f"Error plotting residuals: {e}")


class PlotPanel(QWidget):
    """
    Plot panel for displaying simulation results with matplotlib.
    
    Provides:
    - Tabbed interface for different plot types
    - Interactive matplotlib plots
    - Plot customization controls
    - Export functionality
    """
    
    # Signal for plot export requests
    export_plot_requested = pyqtSignal(str, str)  # (plot_type, filename)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_data = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Title and controls
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Results Visualization")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2196F3;")
        
        # Plot controls
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(10, 100)
        self.bins_spinbox.setValue(30)
        self.bins_spinbox.setPrefix("Bins: ")
        self.bins_spinbox.valueChanged.connect(self._on_bins_changed)
        
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(self._on_grid_changed)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_plots)
        refresh_btn.setMaximumWidth(80)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Plot Options:"))
        header_layout.addWidget(self.bins_spinbox)
        header_layout.addWidget(self.grid_checkbox)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Create tabbed plot interface
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 120px;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid white;
            }
        """)
        
        # Create plot canvases and tabs
        self._create_plot_tabs()
        
        layout.addWidget(self.plot_tabs)
        
    def _create_plot_tabs(self):
        """Create the plot tabs with canvases."""
        if FigureCanvas is None:
            # Matplotlib not available
            error_label = QLabel("Matplotlib not available.\nPlease install matplotlib to view plots.")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setStyleSheet("color: red; font-size: 14px;")
            self.plot_tabs.addTab(error_label, "Error")
            return
            
        # Centroid Error tab
        centroid_widget = QWidget()
        centroid_layout = QVBoxLayout(centroid_widget)
        
        self.centroid_canvas = PlotCanvas(parent=centroid_widget)
        centroid_toolbar = NavigationToolbar(self.centroid_canvas, centroid_widget)
        
        centroid_layout.addWidget(centroid_toolbar)
        centroid_layout.addWidget(self.centroid_canvas)
        
        self.plot_tabs.addTab(centroid_widget, "Centroid Error")
        
        # Attitude Error tab
        attitude_widget = QWidget()
        attitude_layout = QVBoxLayout(attitude_widget)
        
        self.attitude_canvas = PlotCanvas(parent=attitude_widget)
        attitude_toolbar = NavigationToolbar(self.attitude_canvas, attitude_widget)
        
        attitude_layout.addWidget(attitude_toolbar)
        attitude_layout.addWidget(self.attitude_canvas)
        
        self.plot_tabs.addTab(attitude_widget, "Attitude Error")
        
        # Star Field tab
        starfield_widget = QWidget()
        starfield_layout = QVBoxLayout(starfield_widget)
        
        self.starfield_canvas = PlotCanvas(parent=starfield_widget)
        starfield_toolbar = NavigationToolbar(self.starfield_canvas, starfield_widget)
        
        starfield_layout.addWidget(starfield_toolbar)
        starfield_layout.addWidget(self.starfield_canvas)
        
        self.plot_tabs.addTab(starfield_widget, "Star Field")
        
        # Residuals tab
        residuals_widget = QWidget()
        residuals_layout = QVBoxLayout(residuals_widget)
        
        self.residuals_canvas = PlotCanvas(parent=residuals_widget)
        residuals_toolbar = NavigationToolbar(self.residuals_canvas, residuals_widget)
        
        residuals_layout.addWidget(residuals_toolbar)
        residuals_layout.addWidget(self.residuals_canvas)
        
        self.plot_tabs.addTab(residuals_widget, "Residuals")
        
    def _on_bins_changed(self):
        """Handle bins spinbox change."""
        self.refresh_plots()
        
    def _on_grid_changed(self):
        """Handle grid checkbox change."""
        self.refresh_plots()
        
    def update_plots(self, results):
        """Update all plots with new results data."""
        try:
            logger.info("Updating plot panel with results")
            self.results_data = results
            
            if FigureCanvas is None:
                logger.warning("Matplotlib not available, cannot update plots")
                return
                
            # Update each plot based on available data
            self._update_centroid_plot()
            self._update_attitude_plot()
            self._update_starfield_plot()
            self._update_residuals_plot()
            
        except Exception as e:
            logger.error(f"Error updating plots: {e}")
            
    def _update_centroid_plot(self):
        """Update centroid error plot."""
        try:
            # Check for multi-star detection data
            if self.results_data and 'simulation_type' in self.results_data:
                sim_type = self.results_data['simulation_type']

                if sim_type == 'multi_star_catalog':
                    # For multi-star, show detection statistics as bar chart
                    self._plot_detection_stats()
                    return

            # Check for single-star centroid errors
            if self.results_data and 'centroid_errors' in self.results_data:
                errors = self.results_data['centroid_errors']
                if errors and len(errors) > 0:
                    self.centroid_canvas.plot_centroid_histogram(
                        np.array(errors),
                        title="Centroid Error Distribution"
                    )
                    return

            # If no data, show message (don't clear)
            self._show_no_data_message(self.centroid_canvas, "Centroid Error")
            
        except Exception as e:
            logger.error(f"Error updating centroid plot: {e}")
            
    def _plot_detection_stats(self):
        """Plot detection statistics for multi-star results."""
        try:
            self.centroid_canvas.figure.clear()
            ax = self.centroid_canvas.figure.add_subplot(111)
            
            # Get detection stats
            stars_in_catalog = self.results_data.get('stars_in_catalog', 0)
            stars_on_detector = self.results_data.get('stars_on_detector', 0)
            stars_detected = self.results_data.get('stars_detected', 0)
            stars_matched = self.results_data.get('stars_matched', 0)
            
            # Create bar chart
            categories = ['In Catalog', 'On Detector', 'Detected', 'Matched']
            counts = [stars_in_catalog, stars_on_detector, stars_detected, stars_matched]
            colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
            
            bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylabel('Number of Stars', fontsize=11)
            ax.set_title('Star Detection Statistics', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add detection rate text
            detection_rate = self.results_data.get('detection_rate', 0)
            matching_rate = self.results_data.get('matching_rate', 0)
            
            info_text = f"Detection Rate: {detection_rate:.1f}%\nMatching Rate: {matching_rate:.1f}%"
            ax.text(0.98, 0.97, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            self.centroid_canvas.figure.tight_layout()
            self.centroid_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error plotting detection stats: {e}")
            
    def _show_no_data_message(self, canvas, plot_type):
        """Show a 'no data' message on canvas without clearing."""
        try:
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'No {plot_type} data available\nfor this simulation type',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='#666')
            ax.set_title(f'{plot_type} Visualization')
            ax.axis('off')
            canvas.draw()
        except Exception as e:
            logger.error(f"Error showing no data message: {e}")
            
    def _update_attitude_plot(self):
        """Update attitude error plot."""
        try:
            # Check for multi-star QUEST angular errors
            if self.results_data and 'angular_errors' in self.results_data:
                errors = self.results_data['angular_errors']
                if errors and len(errors) > 0:
                    self._plot_quest_errors(errors)
                    return
            
            # Check for angular errors (multi-star QUEST or attitude validation)
            if self.results_data and 'angular_errors' in self.results_data:
                errors = self.results_data['angular_errors']
                if errors and len(errors) > 0:
                    self._plot_quest_errors(errors)
                    return

            # Check for single-star attitude errors
            if self.results_data and 'attitude_errors' in self.results_data:
                errors = self.results_data['attitude_errors']
                if errors and len(errors) > 0:
                    self.attitude_canvas.plot_attitude_histogram(
                        np.array(errors),
                        title="Attitude Error Distribution"
                    )
                    return

            # Show message (don't clear)
            self._show_no_data_message(self.attitude_canvas, "Attitude Error")
            
        except Exception as e:
            logger.error(f"Error updating attitude plot: {e}")
            
    def _plot_quest_errors(self, angular_errors):
        """Plot QUEST angular error distribution."""
        try:
            self.attitude_canvas.figure.clear()
            ax = self.attitude_canvas.figure.add_subplot(111)
            
            errors_array = np.array(angular_errors)
            
            # Create histogram
            n, bins, patches = ax.hist(errors_array, bins=30, alpha=0.7, color='lightgreen',
                                     edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_err = np.mean(errors_array)
            std_err = np.std(errors_array)
            quest_uncertainty = self.results_data.get('quest_uncertainty_arcsec', mean_err)
            
            # Add vertical lines
            ax.axvline(mean_err, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_err:.2f}"')
            ax.axvline(quest_uncertainty, color='blue', linestyle=':', linewidth=2,
                      label=f'QUEST σ: {quest_uncertainty:.2f}"')
            
            ax.set_xlabel('Angular Error (arcseconds)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Monte Carlo QUEST Angular Error Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add info box
            num_trials = len(errors_array)
            info_text = f"Trials: {num_trials}\nMean: {mean_err:.2f}\"\nStd: {std_err:.2f}\""
            ax.text(0.98, 0.97, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            self.attitude_canvas.figure.tight_layout()
            self.attitude_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error plotting QUEST errors: {e}")
            
    def _update_starfield_plot(self):
        """Update star field plot."""
        try:
            # Check if we have actual star field data
            if self.results_data and 'star_positions' in self.results_data:
                positions = self.results_data['star_positions']
                detected_positions = self.results_data.get('detected_positions', None)

                if positions and len(positions) > 0:
                    stars_x = [p[0] for p in positions]
                    stars_y = [p[1] for p in positions]

                    detected_x = None
                    detected_y = None
                    if detected_positions and len(detected_positions) > 0:
                        detected_x = [p[0] for p in detected_positions]
                        detected_y = [p[1] for p in detected_positions]

                    self.starfield_canvas.plot_star_field(
                        stars_x, stars_y, detected_x, detected_y,
                        title="Star Field Visualization"
                    )
                    return
            
            # Generate sample data for multi-star simulations
            if self.results_data and 'simulation_type' in self.results_data:
                if self.results_data['simulation_type'] == 'multi_star_catalog':
                    self._plot_sample_starfield()
                    return
            
            # Show message
            self._show_no_data_message(self.starfield_canvas, "Star Field")
            
        except Exception as e:
            logger.error(f"Error updating star field plot: {e}")
            
    def _plot_sample_starfield(self):
        """Plot a sample star field based on detection statistics."""
        try:
            np.random.seed(42)
            
            # Use actual statistics from results
            n_detected = self.results_data.get('stars_detected', 10)
            n_on_detector = self.results_data.get('stars_on_detector', 15)
            
            # Generate positions
            stars_x = np.random.uniform(0, 2048, n_on_detector)
            stars_y = np.random.uniform(0, 2048, n_on_detector)
            
            # Detected stars with small noise
            detected_x = stars_x[:n_detected] + np.random.normal(0, 0.5, n_detected)
            detected_y = stars_y[:n_detected] + np.random.normal(0, 0.5, n_detected)
            
            self.starfield_canvas.plot_star_field(
                stars_x, stars_y, detected_x, detected_y,
                title=f"Star Field (Simulated - {n_detected}/{n_on_detector} detected)"
            )
            
        except Exception as e:
            logger.error(f"Error plotting sample star field: {e}")
            
    def _update_residuals_plot(self):
        """Update residuals plot."""
        try:
            # For multi-star, show matching performance
            if self.results_data and 'simulation_type' in self.results_data:
                if self.results_data['simulation_type'] == 'multi_star_catalog':
                    self._plot_matching_performance()
                    return
            
            # For single-star, show residuals
            if self.results_data and 'centroid_errors' in self.results_data:
                errors = self.results_data['centroid_errors']
                if errors and len(errors) > 0:
                    # Generate synthetic residuals from errors
                    n_points = len(errors)
                    angles = np.random.uniform(0, 2*np.pi, n_points)
                    x_residuals = np.array(errors) * np.cos(angles)
                    y_residuals = np.array(errors) * np.sin(angles)
                    
                    self.residuals_canvas.plot_residuals(
                        x_residuals, y_residuals,
                        title="Centroiding Residuals"
                    )
                    return
                    
            # Show message
            self._show_no_data_message(self.residuals_canvas, "Residuals")
            
        except Exception as e:
            logger.error(f"Error updating residuals plot: {e}")
            
    def _plot_matching_performance(self):
        """Plot matching performance metrics for multi-star results."""
        try:
            self.residuals_canvas.figure.clear()
            ax = self.residuals_canvas.figure.add_subplot(111)
            
            # Get metrics
            detection_rate = self.results_data.get('detection_rate', 0)
            matching_rate = self.results_data.get('matching_rate', 0)
            quest_uncertainty = self.results_data.get('quest_uncertainty_arcsec', 0)
            exec_time = self.results_data.get('execution_time', 0)
            
            # Normalize QUEST uncertainty to 0-100 scale (assume 0-20 arcsec range)
            quest_score = max(0, 100 - (quest_uncertainty / 20.0) * 100)
            
            # Create metrics display
            metrics = ['Detection\nRate', 'Matching\nRate', 'QUEST\nQuality', 'Speed\nScore']
            values = [
                detection_rate,
                matching_rate,
                quest_score,
                min(100, 100 - min(exec_time * 10, 100))  # Speed score (faster = better)
            ]
            colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, value, metric in zip(bars, values, metrics):
                height = bar.get_height()
                if 'QUEST' in metric:
                    label = f'{quest_uncertainty:.1f}"'
                elif 'Speed' in metric:
                    label = f'{exec_time:.2f}s'
                else:
                    label = f'{value:.1f}%'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Performance Score (%)', fontsize=11)
            ax.set_ylim(0, 110)
            ax.set_title('Multi-Star Simulation Performance', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add summary text
            stars_matched = self.results_data.get('stars_matched', 0)
            stars_detected = self.results_data.get('stars_detected', 0)
            info_text = f"Matched: {stars_matched}/{stars_detected} stars\nQUEST: {quest_uncertainty:.1f}\"\nTime: {exec_time:.2f}s"
            ax.text(0.98, 0.97, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            self.residuals_canvas.figure.tight_layout()
            self.residuals_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error plotting matching performance: {e}")
            
    def refresh_plots(self):
        """Refresh all plots with current data."""
        if self.results_data:
            self.update_plots(self.results_data)
            
    def clear_plots(self):
        """Clear all plots."""
        if FigureCanvas is not None:
            self.centroid_canvas.clear_plot()
            self.attitude_canvas.clear_plot()
            self.starfield_canvas.clear_plot()
            self.residuals_canvas.clear_plot()
        self.results_data = None
        
    def export_current_plot(self, filename):
        """Export the currently visible plot."""
        try:
            current_index = self.plot_tabs.currentIndex()
            if FigureCanvas is None:
                logger.warning("Cannot export plot - matplotlib not available")
                return False
                
            canvas = None
            plot_type = ""
            
            if current_index == 0:
                canvas = self.centroid_canvas
                plot_type = "centroid"
            elif current_index == 1:
                canvas = self.attitude_canvas
                plot_type = "attitude"
            elif current_index == 2:
                canvas = self.starfield_canvas
                plot_type = "starfield"
            elif current_index == 3:
                canvas = self.residuals_canvas
                plot_type = "residuals"
                
            if canvas:
                canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Exported {plot_type} plot to {filename}")
                return True
                
        except Exception as e:
            logger.error(f"Error exporting plot: {e}")
            
        return False