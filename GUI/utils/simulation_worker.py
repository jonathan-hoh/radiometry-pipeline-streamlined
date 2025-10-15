"""
simulation_worker.py - Background simulation execution thread

Provides QThread-based worker for running star tracker simulations
without blocking the GUI interface.
"""

import logging
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
import time

logger = logging.getLogger(__name__)


class SimulationWorker(QThread):
    """Worker thread for running star tracker simulations in background."""
    
    # Signals for communication with main thread
    progress_updated = pyqtSignal(int, str)  # (percentage, status_message)
    simulation_finished = pyqtSignal(dict)   # simulation results
    simulation_failed = pyqtSignal(str)      # error message
    metrics_updated = pyqtSignal(dict)       # live metrics during simulation
    
    def __init__(self, pipeline, config):
        """
        Initialize simulation worker.
        
        Args:
            pipeline: StarTrackerPipeline instance
            config: Configuration dictionary with simulation parameters
        """
        super().__init__()
        self.pipeline = pipeline
        self.config = config
        self.is_cancelled = False
        
    def cancel(self):
        """Cancel the running simulation."""
        self.is_cancelled = True
        logger.info("Simulation cancellation requested")
        
    def run(self):
        """Main simulation execution method (runs in background thread)."""
        try:
            logger.info("Starting simulation worker thread")
            self.progress_updated.emit(0, "Initializing simulation...")
            
            # Allow GUI to update
            time.sleep(0.1)
            
            if self.is_cancelled:
                return
                
            # Update pipeline configuration
            self.progress_updated.emit(10, "Configuring pipeline...")
            self._configure_pipeline()
            
            if self.is_cancelled:
                return
                
            # Run the simulation with progress tracking
            self.progress_updated.emit(20, "Running simulations...")
            results = self._run_simulation_with_progress()
            
            if self.is_cancelled:
                return
                
            # Process and format results
            self.progress_updated.emit(90, "Processing results...")
            formatted_results = self._format_results(results)
            
            if self.is_cancelled:
                return
                
            self.progress_updated.emit(100, "Simulation complete!")
            self.simulation_finished.emit(formatted_results)
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}", exc_info=True)
            self.simulation_failed.emit(str(e))
            
    def _configure_pipeline(self):
        """Configure the pipeline with user settings."""
        # Update pipeline parameters based on config
        if hasattr(self.pipeline, 'magnitude'):
            self.pipeline.magnitude = self.config['magnitude']
        if hasattr(self.pipeline, 'num_simulations'):
            self.pipeline.num_simulations = self.config['num_trials']
            
        # Update camera model parameters if available
        if hasattr(self.pipeline, 'camera'):
            if hasattr(self.pipeline.camera, 'focal_length'):
                self.pipeline.camera.focal_length = self.config['focal_length']
            if hasattr(self.pipeline.camera, 'pixel_pitch'):
                self.pipeline.camera.pixel_pitch = self.config['pixel_pitch']
            if hasattr(self.pipeline.camera, 'resolution'):
                self.pipeline.camera.resolution = self.config['resolution']
                
    def _run_simulation_with_progress(self):
        """Run simulation with progress updates."""
        num_trials = self.config['num_trials']
        progress_step = max(1, num_trials // 10)  # Update every 10% of trials
        
        # Check if pipeline has batch processing capability
        if hasattr(self.pipeline, 'process_psf_batch'):
            return self._run_batch_simulation(num_trials, progress_step)
        else:
            return self._run_single_simulation(num_trials, progress_step)
            
    def _run_batch_simulation(self, num_trials, progress_step):
        """Run simulation using batch processing if available."""
        # For batch processing, we can't easily track individual trials
        # but we can provide intermediate updates
        self.metrics_updated.emit({
            'trials_completed': 0,
            'total_trials': num_trials,
            'estimated_time_remaining': 'Calculating...'
        })
        
        start_time = time.time()
        results = self.pipeline.process_psf_batch(num_trials)
        
        # Emit final metrics
        elapsed_time = time.time() - start_time
        self.metrics_updated.emit({
            'trials_completed': num_trials,
            'total_trials': num_trials,
            'elapsed_time': f"{elapsed_time:.1f}s"
        })
        
        return results
        
    def _run_single_simulation(self, num_trials, progress_step):
        """Run simulation trial by trial with detailed progress tracking."""
        results = []
        start_time = time.time()
        
        for trial in range(num_trials):
            if self.is_cancelled:
                return None
                
            # Run single trial
            trial_result = self.pipeline.process_psf()
            results.append(trial_result)
            
            # Update progress periodically
            if trial % progress_step == 0 or trial == num_trials - 1:
                progress_percent = 20 + int((trial + 1) / num_trials * 70)  # 20-90%
                self.progress_updated.emit(
                    progress_percent, 
                    f"Running trial {trial + 1}/{num_trials}"
                )
                
                # Calculate and emit metrics
                elapsed_time = time.time() - start_time
                if trial > 0:
                    avg_time_per_trial = elapsed_time / (trial + 1)
                    remaining_trials = num_trials - (trial + 1)
                    est_remaining = avg_time_per_trial * remaining_trials
                    
                    self.metrics_updated.emit({
                        'trials_completed': trial + 1,
                        'total_trials': num_trials,
                        'elapsed_time': f"{elapsed_time:.1f}s",
                        'estimated_time_remaining': f"{est_remaining:.1f}s",
                        'avg_time_per_trial': f"{avg_time_per_trial:.2f}s"
                    })
                
                # Allow GUI to update
                QApplication.processEvents()
                
        return results
        
    def _format_results(self, results):
        """Format simulation results for display."""
        if not results:
            return {"error": "No results generated"}
            
        # Handle both single result and list of results
        if isinstance(results, list):
            # Multiple trial results
            if len(results) == 0:
                return {"error": "No valid results"}
                
            # Calculate summary statistics
            formatted = {
                "num_trials": len(results),
                "results": results,
                "summary": self._calculate_summary_stats(results)
            }
        else:
            # Single result
            formatted = {
                "num_trials": 1,
                "results": [results],
                "summary": results
            }
            
        return formatted
        
    def _calculate_summary_stats(self, results):
        """Calculate summary statistics from multiple trial results."""
        try:
            # Extract key metrics if available
            if isinstance(results[0], dict):
                # Try to extract common metrics
                metrics = {}
                
                # Look for common result fields
                for key in ['centroid_error', 'detection_accuracy', 'processing_time']:
                    values = [r.get(key) for r in results if r.get(key) is not None]
                    if values:
                        metrics[key] = {
                            'mean': sum(values) / len(values),
                            'min': min(values),
                            'max': max(values),
                            'count': len(values)
                        }
                        
                return metrics
            else:
                # Simple numeric results
                return {
                    'mean': sum(results) / len(results),
                    'min': min(results),
                    'max': max(results),
                    'count': len(results)
                }
                
        except Exception as e:
            logger.warning(f"Could not calculate summary stats: {e}")
            return {"raw_results": str(results)[:200] + "..."}