#!/usr/bin/env python3
"""
stats_panel.py - Statistics Panel Widget

Provides detailed statistics display for simulation results.
Part of Phase 4 implementation.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QGroupBox, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import logging
import numpy as np

logger = logging.getLogger(__name__)


class StatsPanel(QWidget):
    """
    Statistics panel for displaying simulation results.
    
    Provides:
    - Centroiding performance metrics
    - Attitude determination statistics
    - Star matching success rates
    - Performance indicators with pass/fail
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Main content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Title
        title_label = QLabel("Simulation Results")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2196F3; margin-bottom: 15px;")
        content_layout.addWidget(title_label)
        
        # Create statistics groups
        self._create_centroiding_group(content_layout)
        self._create_attitude_group(content_layout)
        self._create_matching_group(content_layout)
        self._create_performance_group(content_layout)
        
        # Add stretch
        content_layout.addStretch()
        
        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        
    def _create_centroiding_group(self, parent_layout):
        """Create centroiding performance group."""
        group = QGroupBox("Centroiding Performance")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 8px;
                background-color: white;
                color: #4CAF50;
            }
        """)
        
        layout = QVBoxLayout(group)
        
        # Centroid error metrics
        self.centroid_mean_label = self._create_metric_row("Mean Error:", "--", "pixels")
        self.centroid_std_label = self._create_metric_row("Std Deviation:", "--", "pixels")
        self.centroid_95th_label = self._create_metric_row("95th Percentile:", "--", "pixels")
        self.centroid_max_label = self._create_metric_row("Maximum Error:", "--", "pixels")
        
        layout.addWidget(self.centroid_mean_label)
        layout.addWidget(self.centroid_std_label)
        layout.addWidget(self.centroid_95th_label)
        layout.addWidget(self.centroid_max_label)
        
        # Performance indicator
        self.centroid_indicator = self._create_performance_indicator()
        layout.addWidget(self.centroid_indicator)
        
        parent_layout.addWidget(group)
        
    def _create_attitude_group(self, parent_layout):
        """Create attitude determination group."""
        group = QGroupBox("Attitude Performance")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 8px;
                background-color: white;
                color: #2196F3;
            }
        """)
        
        layout = QVBoxLayout(group)
        
        # Attitude error metrics
        self.attitude_mean_label = self._create_metric_row("Mean Error:", "--", "arcsec")
        self.attitude_std_label = self._create_metric_row("Std Deviation:", "--", "arcsec")
        self.attitude_3sigma_label = self._create_metric_row("3σ Bound:", "--", "arcsec")
        self.attitude_max_label = self._create_metric_row("Maximum Error:", "--", "arcsec")
        
        layout.addWidget(self.attitude_mean_label)
        layout.addWidget(self.attitude_std_label)
        layout.addWidget(self.attitude_3sigma_label)
        layout.addWidget(self.attitude_max_label)
        
        # Performance indicator
        self.attitude_indicator = self._create_performance_indicator()
        layout.addWidget(self.attitude_indicator)
        
        parent_layout.addWidget(group)
        
    def _create_matching_group(self, parent_layout):
        """Create star matching group."""
        group = QGroupBox("Star Matching")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #FF9800;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 8px;
                background-color: white;
                color: #FF9800;
            }
        """)
        
        layout = QVBoxLayout(group)
        
        # Matching metrics
        self.success_rate_label = self._create_metric_row("Success Rate:", "--", "%")
        self.stars_detected_label = self._create_metric_row("Avg Stars Detected:", "--", "stars")
        self.stars_matched_label = self._create_metric_row("Avg Stars Matched:", "--", "stars")
        self.match_time_label = self._create_metric_row("Avg Match Time:", "--", "ms")
        
        layout.addWidget(self.success_rate_label)
        layout.addWidget(self.stars_detected_label)
        layout.addWidget(self.stars_matched_label)
        layout.addWidget(self.match_time_label)
        
        # Performance indicator
        self.matching_indicator = self._create_performance_indicator()
        layout.addWidget(self.matching_indicator)
        
        parent_layout.addWidget(group)
        
    def _create_performance_group(self, parent_layout):
        """Create overall performance group."""
        group = QGroupBox("Overall Performance")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #9C27B0;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 8px;
                background-color: white;
                color: #9C27B0;
            }
        """)
        
        layout = QVBoxLayout(group)
        
        # Performance metrics
        self.total_trials_label = self._create_metric_row("Total Trials:", "--", "")
        self.successful_trials_label = self._create_metric_row("Successful Trials:", "--", "")
        self.execution_time_label = self._create_metric_row("Execution Time:", "--", "seconds")
        self.avg_trial_time_label = self._create_metric_row("Avg Time/Trial:", "--", "ms")
        
        layout.addWidget(self.total_trials_label)
        layout.addWidget(self.successful_trials_label)
        layout.addWidget(self.execution_time_label)
        layout.addWidget(self.avg_trial_time_label)
        
        # Overall score
        self.overall_score = QProgressBar()
        self.overall_score.setRange(0, 100)
        self.overall_score.setValue(0)
        self.overall_score.setTextVisible(True)
        self.overall_score.setFormat("Overall Score: %p%")
        self.overall_score.setStyleSheet("""
            QProgressBar {
                border: 2px solid #9C27B0;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #E91E63, stop: 0.5 #9C27B0, stop: 1 #673AB7);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.overall_score)
        
        parent_layout.addWidget(group)
        
    def _create_metric_row(self, label_text, value_text, unit_text):
        """Create a metric display row."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(10, 5, 10, 5)
        
        # Label
        label = QLabel(label_text)
        label.setMinimumWidth(120)
        label.setStyleSheet("font-weight: normal; color: #333;")
        
        # Value
        value = QLabel(value_text)
        value.setObjectName(f"value_{label_text.lower().replace(' ', '_').replace(':', '')}")
        value.setStyleSheet("font-family: monospace; font-weight: bold; color: #000;")
        value.setMinimumWidth(80)
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Unit
        unit = QLabel(unit_text)
        unit.setMinimumWidth(60)
        unit.setStyleSheet("font-style: italic; color: #666;")
        
        row_layout.addWidget(label)
        row_layout.addStretch()
        row_layout.addWidget(value)
        row_layout.addWidget(unit)
        
        return row_widget
        
    def _create_performance_indicator(self):
        """Create a performance pass/fail indicator."""
        indicator = QLabel("⚪ Not evaluated")
        indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        indicator.setStyleSheet("""
            QLabel {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f9f9f9;
                font-weight: bold;
            }
        """)
        return indicator
        
    def update_results(self, results):
        """Update statistics display with simulation results."""
        try:
            logger.info("Updating statistics panel with results")
            
            # Extract results data
            if isinstance(results, dict):
                # Check simulation type
                sim_type = results.get('simulation_type', 'single_star')
                
                if sim_type == 'multi_star':
                    self._update_multi_star_stats(results)
                else:
                    self._update_centroiding_stats(results)
                    self._update_attitude_stats(results)
                    self._update_matching_stats(results)
                    self._update_performance_stats(results)
                    self._calculate_overall_score(results)
            else:
                logger.warning(f"Results not in expected format: {type(results)}")
                
        except Exception as e:
            logger.error(f"Error updating statistics panel: {e}")
            
    def _update_centroiding_stats(self, results):
        """Update centroiding statistics."""
        # Look for centroiding data in various possible keys
        centroid_errors = None
        
        if 'centroid_errors' in results:
            centroid_errors = results['centroid_errors']
        elif 'centroid_error_mean' in results:
            # Single values
            mean_val = results.get('centroid_error_mean', 0)
            std_val = results.get('centroid_error_std', 0)
            self._set_metric_value("mean_error", f"{mean_val:.3f}")
            self._set_metric_value("std_deviation", f"{std_val:.3f}")
            
        if centroid_errors is not None and len(centroid_errors) > 0:
            errors = np.array(centroid_errors)
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            p95_err = np.percentile(errors, 95)
            max_err = np.max(errors)
            
            self._set_metric_value("mean_error", f"{mean_err:.3f}")
            self._set_metric_value("std_deviation", f"{std_err:.3f}")
            self._set_metric_value("95th_percentile", f"{p95_err:.3f}")
            self._set_metric_value("maximum_error", f"{max_err:.3f}")
            
            # Update performance indicator
            if mean_err < 0.3:  # Good performance threshold
                self._set_performance_indicator(self.centroid_indicator, "✅ Within ±0.3 px criteria", "green")
            elif mean_err < 0.5:
                self._set_performance_indicator(self.centroid_indicator, "⚠️ Acceptable performance", "orange")
            else:
                self._set_performance_indicator(self.centroid_indicator, "❌ Exceeds error budget", "red")
                
    def _update_attitude_stats(self, results):
        """Update attitude determination statistics."""
        # Look for attitude data
        attitude_errors = None
        
        if 'attitude_errors' in results:
            attitude_errors = results['attitude_errors']
        elif 'attitude_error_rms' in results:
            # Single values
            rms_val = results.get('attitude_error_rms', 0)
            self._set_metric_value("mean_error", f"{rms_val:.2f}")
            
        if attitude_errors is not None and len(attitude_errors) > 0:
            errors = np.array(attitude_errors)
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            bound_3sigma = mean_err + 3 * std_err
            max_err = np.max(errors)
            
            self._set_metric_value("mean_error", f"{mean_err:.2f}")
            self._set_metric_value("std_deviation", f"{std_err:.2f}")
            self._set_metric_value("3σ_bound", f"{bound_3sigma:.2f}")
            self._set_metric_value("maximum_error", f"{max_err:.2f}")
            
            # Update performance indicator
            if bound_3sigma < 10.0:  # Good performance threshold
                self._set_performance_indicator(self.attitude_indicator, "✅ Meets 10 arcsec 3σ requirement", "green")
            elif bound_3sigma < 20.0:
                self._set_performance_indicator(self.attitude_indicator, "⚠️ Marginal performance", "orange")
            else:
                self._set_performance_indicator(self.attitude_indicator, "❌ Exceeds error budget", "red")
                
    def _update_multi_star_stats(self, results):
        """Update statistics for multi-star catalog simulations."""
        # Centroiding section - show detection info
        stars_catalog = results.get('stars_in_catalog', 0)
        stars_detector = results.get('stars_on_detector', 0)
        stars_detected = results.get('stars_detected', 0)
        
        self._set_metric_value("mean_error", f"{stars_detected}")
        self._set_metric_value("std_deviation", f"{stars_detector}")
        self._set_metric_value("95th_percentile", f"{stars_catalog}")
        self._set_metric_value("maximum_error", "N/A")
        
        # Update centroid labels to match multi-star context
        detection_rate = results.get('detection_rate', 0)
        if detection_rate > 0.9:
            self._set_performance_indicator(self.centroid_indicator, f"✅ {detection_rate*100:.1f}% detection rate", "green")
        elif detection_rate > 0.7:
            self._set_performance_indicator(self.centroid_indicator, f"⚠️ {detection_rate*100:.1f}% detection rate", "orange")
        else:
            self._set_performance_indicator(self.centroid_indicator, f"❌ {detection_rate*100:.1f}% detection rate", "red")
        
        # Attitude section - show QUEST results
        quest_uncertainty = results.get('quest_uncertainty_arcsec', 0)
        mean_angular_error = results.get('mean_angular_error_deg', 0) * 3600  # Convert to arcsec
        max_angular_error = results.get('max_angular_error_deg', 0) * 3600
        
        self._set_metric_value("mean_error", f"{mean_angular_error:.2f}" if mean_angular_error > 0 else "N/A")
        self._set_metric_value("std_deviation", f"{quest_uncertainty:.2f}" if quest_uncertainty > 0 else "N/A")
        self._set_metric_value("3σ_bound", f"{quest_uncertainty * 3:.2f}" if quest_uncertainty > 0 else "N/A")
        self._set_metric_value("maximum_error", f"{max_angular_error:.2f}" if max_angular_error > 0 else "N/A")
        
        if quest_uncertainty > 0:
            if quest_uncertainty < 10.0:
                self._set_performance_indicator(self.attitude_indicator, f"✅ QUEST: {quest_uncertainty:.1f} arcsec", "green")
            elif quest_uncertainty < 20.0:
                self._set_performance_indicator(self.attitude_indicator, f"⚠️ QUEST: {quest_uncertainty:.1f} arcsec", "orange")
            else:
                self._set_performance_indicator(self.attitude_indicator, f"❌ QUEST: {quest_uncertainty:.1f} arcsec", "red")
        else:
            self._set_performance_indicator(self.attitude_indicator, "⚪ QUEST not performed", "gray")
        
        # Matching section - show BAST results
        stars_matched = results.get('stars_matched', 0)
        matching_rate = results.get('matching_rate', 0)
        
        self._set_metric_value("success_rate", f"{matching_rate*100:.1f}" if matching_rate > 0 else "0.0")
        self._set_metric_value("avg_stars_detected", f"{stars_detected:.0f}")
        self._set_metric_value("avg_stars_matched", f"{stars_matched:.0f}")
        self._set_metric_value("avg_match_time", "N/A")
        
        if stars_matched >= 3:
            self._set_performance_indicator(self.matching_indicator, f"✅ {stars_matched} stars matched", "green")
        elif stars_matched >= 2:
            self._set_performance_indicator(self.matching_indicator, f"⚠️ {stars_matched} stars matched", "orange")
        else:
            self._set_performance_indicator(self.matching_indicator, f"❌ {stars_matched} stars matched", "red")
        
        # Performance section
        execution_time = results.get('execution_time', 0)
        quest_trials = results.get('quest_trials', 0)
        
        self._set_metric_value("total_trials", f"{quest_trials}" if quest_trials > 0 else "N/A")
        self._set_metric_value("successful_trials", f"{stars_matched}/{stars_detected}")
        self._set_metric_value("execution_time", f"{execution_time:.1f}")
        self._set_metric_value("avg_timetrialspan", "N/A")
        
        # Calculate overall score for multi-star
        score = 0
        if detection_rate > 0.8:
            score += 30
        if matching_rate > 0.5:
            score += 40
        if quest_uncertainty > 0 and quest_uncertainty < 10:
            score += 30
        
        self.overall_score.setValue(int(score))
                
    def _update_matching_stats(self, results):
        """Update star matching statistics."""
        # Extract matching statistics
        num_trials = results.get('num_trials', results.get('num_total', 1))
        num_successful = results.get('num_successful', num_trials)
        success_rate = (num_successful / num_trials) * 100 if num_trials > 0 else 0
        
        self._set_metric_value("success_rate", f"{success_rate:.1f}")
        
        # Look for star detection data
        if 'stars_detected' in results:
            if isinstance(results['stars_detected'], (list, np.ndarray)):
                avg_detected = np.mean(results['stars_detected']) if len(results['stars_detected']) > 0 else 0
            else:
                avg_detected = results['stars_detected']
            self._set_metric_value("avg_stars_detected", f"{avg_detected:.1f}")
            
        if 'stars_matched' in results:
            if isinstance(results['stars_matched'], (list, np.ndarray)):
                avg_matched = np.mean(results['stars_matched']) if len(results['stars_matched']) > 0 else 0
            else:
                avg_matched = results['stars_matched']
            self._set_metric_value("avg_stars_matched", f"{avg_matched:.1f}")
            
        # Update performance indicator
        if success_rate >= 95.0:
            self._set_performance_indicator(self.matching_indicator, "✅ Excellent match rate", "green")
        elif success_rate >= 85.0:
            self._set_performance_indicator(self.matching_indicator, "⚠️ Good match rate", "orange")
        else:
            self._set_performance_indicator(self.matching_indicator, "❌ Poor match rate", "red")
            
    def _update_performance_stats(self, results):
        """Update overall performance statistics."""
        num_trials = results.get('num_trials', results.get('num_total', 0))
        num_successful = results.get('num_successful', num_trials)
        execution_time = results.get('execution_time', 0)
        
        self._set_metric_value("total_trials", str(num_trials))
        self._set_metric_value("successful_trials", f"{num_successful}/{num_trials}")
        self._set_metric_value("execution_time", f"{execution_time:.1f}")
        
        if execution_time > 0 and num_trials > 0:
            avg_trial_time = (execution_time / num_trials) * 1000  # Convert to ms
            self._set_metric_value("avg_timetrialspan", f"{avg_trial_time:.1f}")
            
    def _calculate_overall_score(self, results):
        """Calculate and display overall performance score."""
        try:
            score = 0
            weight_sum = 0
            
            # Centroiding score (30% weight)
            if 'centroid_error_mean' in results:
                centroid_error = results['centroid_error_mean']
                if centroid_error < 0.2:
                    centroid_score = 100
                elif centroid_error < 0.5:
                    centroid_score = 80 - (centroid_error - 0.2) * 100
                else:
                    centroid_score = max(0, 50 - (centroid_error - 0.5) * 50)
                score += centroid_score * 0.3
                weight_sum += 0.3
                
            # Attitude score (40% weight)
            if 'attitude_error_rms' in results:
                attitude_error = results['attitude_error_rms']
                if attitude_error < 5.0:
                    attitude_score = 100
                elif attitude_error < 15.0:
                    attitude_score = 80 - (attitude_error - 5.0) * 4
                else:
                    attitude_score = max(0, 40 - (attitude_error - 15.0) * 2)
                score += attitude_score * 0.4
                weight_sum += 0.4
                
            # Matching score (30% weight)
            num_trials = results.get('num_trials', results.get('num_total', 1))
            num_successful = results.get('num_successful', num_trials)
            if num_trials > 0:
                success_rate = (num_successful / num_trials) * 100
                if success_rate >= 95:
                    matching_score = 100
                elif success_rate >= 80:
                    matching_score = 60 + (success_rate - 80) * 2.67
                else:
                    matching_score = success_rate * 0.75
                score += matching_score * 0.3
                weight_sum += 0.3
                
            # Normalize score
            if weight_sum > 0:
                final_score = int(score / weight_sum)
            else:
                final_score = 0
                
            self.overall_score.setValue(final_score)
            
        except Exception as e:
            logger.warning(f"Error calculating overall score: {e}")
            self.overall_score.setValue(0)
            
    def _set_metric_value(self, metric_name, value):
        """Set value for a specific metric."""
        try:
            value_label = self.findChild(QLabel, f"value_{metric_name}")
            if value_label:
                value_label.setText(str(value))
        except Exception as e:
            logger.warning(f"Could not update metric {metric_name}: {e}")
            
    def _set_performance_indicator(self, indicator, text, color):
        """Set performance indicator status."""
        color_styles = {
            'green': "background-color: #E8F5E8; color: #2E7D32; border-color: #4CAF50;",
            'orange': "background-color: #FFF3E0; color: #F57C00; border-color: #FF9800;",
            'red': "background-color: #FFEBEE; color: #C62828; border-color: #F44336;"
        }
        
        style = color_styles.get(color, "")
        indicator.setText(text)
        indicator.setStyleSheet(f"""
            QLabel {{
                padding: 8px;
                border: 2px solid;
                border-radius: 4px;
                font-weight: bold;
                {style}
            }}
        """)
        
    def clear_results(self):
        """Clear all displayed results."""
        # Reset all metric values to default
        for child in self.findChildren(QLabel):
            if child.objectName().startswith("value_"):
                child.setText("--")
                
        # Reset performance indicators
        for indicator in [self.centroid_indicator, self.attitude_indicator, self.matching_indicator]:
            self._set_performance_indicator(indicator, "⚪ Not evaluated", "gray")
            
        # Reset overall score
        self.overall_score.setValue(0)