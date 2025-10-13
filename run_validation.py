#!/usr/bin/env python3
"""
run_validation.py - Main Validation Script

Top-level script for executing comprehensive star tracker validation campaigns.
Orchestrates all validation modules, manages configuration, and generates reports.

Usage:
    PYTHONPATH=. python run_validation.py --help
    PYTHONPATH=. python run_validation.py --module all
    PYTHONPATH=. python run_validation.py --module attitude --config custom_config.yaml
    PYTHONPATH=. python run_validation.py --quick
"""

import argparse
import sys
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import validation modules
from validation.attitude_validation import AttitudeValidator, AttitudeValidationConfig
from validation.identification_validation import IdentificationValidator, IdentificationConfig
from validation.astrometric_validation import AstrometricValidator, AstrometricConfig
from validation.photometric_validation import PhotometricValidator, PhotometricConfig
from validation.noise_validation import NoiseValidator, NoiseConfig
from validation.reporting import ValidationReporter, ReportConfig
from validation.monte_carlo import MonteCarloValidator

# Import core pipeline components  
from src.core.star_tracker_pipeline import StarTrackerPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('validation/validation.log')
    ]
)
logger = logging.getLogger(__name__)

class ValidationCampaign:
    """
    Main validation campaign orchestrator.
    
    Manages execution of validation modules, configuration loading,
    progress tracking, and results compilation.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize validation campaign.
        
        Parameters
        ----------
        config_path : Path, optional
            Path to validation configuration file
        """
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # Initialize core components
        self.pipeline = None
        self.validators = {}
        self.results = {}
        self.start_time = None
        
        # Create output directories
        self._setup_output_directories()
        
        logger.info("ValidationCampaign initialized")
        
    def _load_configuration(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load validation configuration."""
        if config_path is None:
            config_path = Path("validation/config/validation_config.yaml")
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'monte_carlo': {'n_samples': 100, 'parallel_workers': 4},
            'validation_thresholds': {
                'attitude_error_arcsec': 5.0,
                'identification_rate_min': 0.90,
                'astrometric_rms_pixels': 0.5
            },
            'output': {'results_dir': 'validation/results'}
        }
    
    def _setup_output_directories(self):
        """Create output directory structure."""
        base_dir = Path(self.config.get('output', {}).get('results_dir', 'validation/results'))
        
        directories = [
            base_dir,
            base_dir / 'attitude',
            base_dir / 'identification', 
            base_dir / 'astrometric',
            base_dir / 'photometric',
            base_dir / 'noise',
            base_dir / 'reports'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directories created under {base_dir}")
    
    def initialize_pipeline(self):
        """Initialize star tracker pipeline."""
        try:
            self.pipeline = StarTrackerPipeline()
            logger.info("StarTrackerPipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def initialize_validators(self, modules: List[str]):
        """Initialize validation modules."""
        logger.info(f"Initializing validators for modules: {modules}")
        
        if not self.pipeline:
            self.initialize_pipeline()
        
        try:
            if 'attitude' in modules:
                config = AttitudeValidationConfig(
                    n_monte_carlo=self.config.get('monte_carlo', {}).get('n_samples', 100),
                    output_dir=Path(self.config['output']['results_dir']) / 'attitude'
                )
                self.validators['attitude'] = AttitudeValidator(self.pipeline, config)
                
            if 'identification' in modules:
                config = IdentificationConfig(
                    n_trials_per_config=50,
                    output_dir=Path(self.config['output']['results_dir']) / 'identification'
                )
                # Note: Would need actual BAST instance and catalog interface
                # self.validators['identification'] = IdentificationValidator(bast_instance, catalog, config)
                logger.warning("Identification validator requires BAST instance - skipping")
                
            if 'astrometric' in modules:
                config = AstrometricConfig(
                    field_grid_points=11,  # Reduced for faster execution
                    output_dir=Path(self.config['output']['results_dir']) / 'astrometric'
                )
                # Note: Would need actual camera model and catalog interface
                # self.validators['astrometric'] = AstrometricValidator(camera_model, catalog, config)
                logger.warning("Astrometric validator requires camera model - skipping")
                
            if 'photometric' in modules:
                config = PhotometricConfig(
                    n_trials_per_magnitude=25,
                    output_dir=Path(self.config['output']['results_dir']) / 'photometric'
                )
                # Note: Would need actual camera model and PSF simulator
                # self.validators['photometric'] = PhotometricValidator(camera_model, psf_sim, config)
                logger.warning("Photometric validator requires camera model - skipping")
                
            if 'noise' in modules:
                config = NoiseConfig(
                    n_trials_per_snr=50,
                    output_dir=Path(self.config['output']['results_dir']) / 'noise'
                )
                # Note: Would need detector model
                # self.validators['noise'] = NoiseValidator(self.pipeline, detector_model, config)
                logger.warning("Noise validator requires detector model - skipping")
                
            logger.info(f"Initialized {len(self.validators)} validators")
            
        except Exception as e:
            logger.error(f"Failed to initialize validators: {e}")
            raise
    
    def run_validation_module(self, module_name: str) -> Dict[str, Any]:
        """Run individual validation module."""
        if module_name not in self.validators:
            logger.error(f"Validator not initialized: {module_name}")
            return {'error': f'Validator {module_name} not available'}
        
        logger.info(f"Running {module_name} validation...")
        start_time = time.time()
        
        try:
            validator = self.validators[module_name]
            
            if module_name == 'attitude':
                results = validator.run_validation_campaign()
            elif module_name == 'identification':
                results = validator.run_identification_sweep()
            elif module_name == 'astrometric':
                results = validator.run_astrometric_validation()
            elif module_name == 'photometric':
                results = validator.run_photometric_validation()
            elif module_name == 'noise':
                results = validator.run_noise_characterization()
            else:
                raise ValueError(f"Unknown validation module: {module_name}")
            
            execution_time = time.time() - start_time
            results['execution_metadata'] = {
                'module': module_name,
                'execution_time_seconds': execution_time,
                'completion_time': datetime.utcnow().isoformat()
            }
            
            logger.info(f"{module_name} validation completed in {execution_time:.1f}s")
            return results
            
        except Exception as e:
            logger.error(f"Validation module {module_name} failed: {e}")
            return {
                'error': str(e),
                'module': module_name,
                'execution_time_seconds': time.time() - start_time
            }
    
    def run_campaign(self, modules: List[str]) -> Dict[str, Any]:
        """Run complete validation campaign."""
        logger.info(f"Starting validation campaign for modules: {modules}")
        self.start_time = time.time()
        
        # Initialize validators
        self.initialize_validators(modules)
        
        # Run validation modules
        campaign_results = {
            'campaign_metadata': {
                'start_time': datetime.utcnow().isoformat(),
                'modules_requested': modules,
                'configuration': self.config
            },
            'module_results': {},
            'campaign_summary': {}
        }
        
        successful_modules = []
        failed_modules = []
        
        for module in modules:
            if module in self.validators:
                result = self.run_validation_module(module)
                campaign_results['module_results'][module] = result
                
                if 'error' in result:
                    failed_modules.append(module)
                else:
                    successful_modules.append(module)
            else:
                logger.warning(f"Skipping unavailable module: {module}")
                failed_modules.append(module)
        
        # Generate campaign summary
        total_time = time.time() - self.start_time
        campaign_results['campaign_summary'] = {
            'total_execution_time_seconds': total_time,
            'modules_successful': successful_modules,
            'modules_failed': failed_modules,
            'success_rate': len(successful_modules) / len(modules) if modules else 0.0,
            'completion_time': datetime.utcnow().isoformat()
        }
        
        # Check validation thresholds
        threshold_results = self._check_validation_thresholds(campaign_results)
        campaign_results['threshold_analysis'] = threshold_results
        
        # Generate reports
        if successful_modules:
            reports = self._generate_reports(campaign_results)
            campaign_results['reports'] = reports
        
        logger.info(f"Validation campaign completed in {total_time:.1f}s")
        logger.info(f"Success rate: {len(successful_modules)}/{len(modules)} modules")
        
        self.results = campaign_results
        return campaign_results
    
    def _check_validation_thresholds(self, campaign_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check results against validation thresholds."""
        thresholds = self.config.get('validation_thresholds', {})
        threshold_results = {
            'thresholds_defined': thresholds,
            'threshold_checks': {},
            'overall_pass': True
        }
        
        module_results = campaign_results.get('module_results', {})
        
        # Check attitude error threshold
        if 'attitude_error_arcsec' in thresholds and 'attitude' in module_results:
            attitude_results = module_results['attitude']
            if 'statistics' in attitude_results:
                stats = attitude_results['statistics']
                if 'attitude_error_arcsec' in stats:
                    mean_error = stats['attitude_error_arcsec'].get('mean', float('inf'))
                    threshold = thresholds['attitude_error_arcsec']
                    passed = mean_error <= threshold
                    
                    threshold_results['threshold_checks']['attitude_error'] = {
                        'threshold': threshold,
                        'measured_value': mean_error,
                        'passed': passed,
                        'margin': threshold - mean_error
                    }
                    
                    if not passed:
                        threshold_results['overall_pass'] = False
        
        # Check identification rate threshold
        if 'identification_rate_min' in thresholds and 'identification' in module_results:
            id_results = module_results['identification']
            if 'statistics' in id_results:
                stats = id_results['statistics']
                if 'identification_rate' in stats:
                    mean_rate = stats['identification_rate'].get('mean', 0.0)
                    threshold = thresholds['identification_rate_min']
                    passed = mean_rate >= threshold
                    
                    threshold_results['threshold_checks']['identification_rate'] = {
                        'threshold': threshold,
                        'measured_value': mean_rate,
                        'passed': passed,
                        'margin': mean_rate - threshold
                    }
                    
                    if not passed:
                        threshold_results['overall_pass'] = False
        
        return threshold_results
    
    def _generate_reports(self, campaign_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation reports."""
        logger.info("Generating validation reports...")
        
        try:
            # Initialize reporter
            report_config = ReportConfig(
                output_dir=Path(self.config['output']['results_dir']) / 'reports',
                template_style='professional'
            )
            reporter = ValidationReporter(report_config)
            
            # Generate summary report
            summary_report = reporter.generate_summary_report(campaign_results['module_results'])
            
            # Generate presentation slides
            presentation = reporter.create_presentation_slides(
                campaign_results['module_results'], 
                template='saas_pitch'
            )
            
            # Save JSON summary
            json_summary_path = reporter.save_json_summary(campaign_results['module_results'])
            
            reports = {
                'summary_report': summary_report,
                'presentation': presentation,
                'json_summary_path': str(json_summary_path)
            }
            
            logger.info("Reports generated successfully")
            return reports
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'error': f'Report generation failed: {str(e)}'}
    
    def save_campaign_results(self, output_path: Optional[Path] = None) -> Path:
        """Save complete campaign results."""
        if output_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.config['output']['results_dir']) / f"campaign_results_{timestamp}.json"
        
        try:
            import json
            with open(output_path, 'w') as f:
                json.dump(self._make_json_serializable(self.results), f, indent=2)
            
            logger.info(f"Campaign results saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save campaign results: {e}")
            raise
    
    def _make_json_serializable(self, obj):
        """Convert objects for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Star Tracker Validation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --module all                    # Run all validation modules
  %(prog)s --module attitude               # Run attitude validation only
  %(prog)s --module attitude,astrometric   # Run specific modules
  %(prog)s --quick                         # Quick validation (reduced trials)
  %(prog)s --config my_config.yaml        # Use custom configuration
  %(prog)s --output-dir ./my_results       # Custom output directory
        """
    )
    
    parser.add_argument(
        '--module', '-m',
        type=str,
        default='all',
        help='Validation modules to run: all, attitude, identification, astrometric, photometric, noise (comma-separated)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to validation configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick validation with reduced trial counts'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running validation'
    )
    
    return parser.parse_args()

def main():
    """Main validation script entry point."""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse modules
    if args.module.lower() == 'all':
        modules = ['attitude', 'identification', 'astrometric', 'photometric', 'noise']
    else:
        modules = [m.strip() for m in args.module.split(',')]
    
    # Validate module names
    valid_modules = ['attitude', 'identification', 'astrometric', 'photometric', 'noise']
    invalid_modules = [m for m in modules if m not in valid_modules]
    if invalid_modules:
        logger.error(f"Invalid modules: {invalid_modules}")
        logger.error(f"Valid modules: {valid_modules}")
        return 1
    
    logger.info("=" * 60)
    logger.info("STAR TRACKER VALIDATION FRAMEWORK")
    logger.info("=" * 60)
    logger.info(f"Modules to run: {modules}")
    logger.info(f"Configuration: {args.config or 'default'}")
    logger.info(f"Output directory: {args.output_dir or 'default'}")
    logger.info(f"Quick mode: {args.quick}")
    
    if args.dry_run:
        logger.info("DRY RUN - No validation will be executed")
        return 0
    
    try:
        # Initialize validation campaign
        campaign = ValidationCampaign(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            campaign.config['output']['results_dir'] = str(args.output_dir)
            campaign._setup_output_directories()
        
        # Quick mode configuration
        if args.quick:
            logger.info("Quick mode: reducing trial counts for faster execution")
            campaign.config['monte_carlo']['n_samples'] = 50
            campaign.config.setdefault('quick_mode', True)
        
        # Run validation campaign
        results = campaign.run_campaign(modules)
        
        # Save results
        results_file = campaign.save_campaign_results()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION CAMPAIGN SUMMARY")
        logger.info("=" * 60)
        
        summary = results['campaign_summary']
        logger.info(f"Execution time: {summary['total_execution_time_seconds']:.1f}s")
        logger.info(f"Successful modules: {summary['modules_successful']}")
        logger.info(f"Failed modules: {summary['modules_failed']}")
        logger.info(f"Success rate: {summary['success_rate']*100:.1f}%")
        logger.info(f"Results saved: {results_file}")
        
        # Check validation thresholds
        if 'threshold_analysis' in results:
            threshold_results = results['threshold_analysis']
            overall_pass = threshold_results['overall_pass']
            logger.info(f"Validation thresholds: {'PASS' if overall_pass else 'FAIL'}")
            
            for check_name, check_result in threshold_results.get('threshold_checks', {}).items():
                status = 'PASS' if check_result['passed'] else 'FAIL'
                logger.info(f"  {check_name}: {status} ({check_result['measured_value']:.3f})")
        
        # Return appropriate exit code
        if summary['success_rate'] == 1.0 and results.get('threshold_analysis', {}).get('overall_pass', True):
            logger.info("✅ All validation tests passed!")
            return 0
        else:
            logger.warning("⚠️  Some validation tests failed or did not meet thresholds")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())