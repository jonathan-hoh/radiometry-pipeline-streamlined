# BAST Algorithm Optimizer - Codebase Creation Instructions

## Overview
Create a new standalone codebase for optimizing the BAST (Basic Autonomous Star Tracker) algorithm. This codebase will focus exclusively on performance testing and optimization of the three core BAST stages: **centroiding**, **matching**, and **QUEST attitude determination**.

## Key Differences from Current Codebase
- **Input**: PNG images of night sky (from external synthetic sky generator) instead of radiometric simulation
- **Scope**: Algorithm optimization only - no radiometry, sensor modeling, or complex physics
- **Purpose**: Measure performance, identify bottlenecks, and improve processing efficiency
- **Catalog**: Full Hipparcos catalog for comprehensive matching tests

---

## Directory Structure

Create the following structure in a new directory called `bast-optimizer`:

```
bast-optimizer/
├── README.md
├── requirements.txt
├── setup.py (optional)
├── config/
│   ├── default_config.yaml
│   └── catalog_settings.yaml
├── src/
│   ├── __init__.py
│   ├── bast/
│   │   ├── __init__.py
│   │   ├── catalog.py          [COPY from current]
│   │   ├── identify.py         [COPY from current]
│   │   ├── match.py            [COPY from current]
│   │   └── resolve.py          [COPY from current]
│   ├── io/
│   │   ├── __init__.py
│   │   ├── image_loader.py     [NEW]
│   │   └── catalog_loader.py   [NEW]
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── profiler.py         [NEW]
│   │   ├── stage_timer.py      [NEW]
│   │   └── memory_monitor.py   [NEW]
│   └── pipeline/
│       ├── __init__.py
│       ├── optimizer_pipeline.py [NEW]
│       └── batch_processor.py    [NEW]
├── tests/
│   ├── __init__.py
│   ├── test_centroiding.py     [NEW]
│   ├── test_matching.py        [NEW]
│   └── test_quest.py           [NEW]
├── benchmarks/
│   ├── __init__.py
│   ├── centroid_benchmark.py   [NEW]
│   ├── match_benchmark.py      [NEW]
│   └── quest_benchmark.py      [NEW]
├── data/
│   ├── catalogs/
│   │   └── .gitkeep
│   ├── test_images/
│   │   └── .gitkeep
│   └── ground_truth/
│       └── .gitkeep
├── results/
│   ├── profiles/
│   │   └── .gitkeep
│   └── reports/
│       └── .gitkeep
└── scripts/
    ├── download_hipparcos.py   [NEW]
    ├── run_benchmark_suite.py  [NEW]
    └── generate_report.py      [NEW]
```

---

## Step-by-Step Creation Instructions

### Step 1: Create Base Directory and Structure
```bash
# Create main directory
mkdir bast-optimizer
cd bast-optimizer

# Create all subdirectories
mkdir -p src/bast src/io src/benchmark src/pipeline
mkdir -p tests benchmarks data/catalogs data/test_images data/ground_truth
mkdir -p results/profiles results/reports scripts config
```

### Step 2: Copy BAST Core Files

Copy these files from `src/BAST/` in the current codebase to `src/bast/`:
- [`catalog.py`](src/BAST/catalog.py) - Star catalog management and triplet calculation
- [`identify.py`](src/BAST/identify.py) - Centroiding and bearing vector calculation
- [`match.py`](src/BAST/match.py) - Star pattern matching using triplet algorithm
- [`resolve.py`](src/BAST/resolve.py) - QUEST attitude determination

**Important**: After copying, modify imports in these files:
- Remove any dependencies on `preprocess.py` (not needed)
- Remove any dependencies on `calibrate.py` (simplified calibration)
- Ensure all imports use relative imports within the `bast` package

### Step 3: Create New Image Loading Module

**File**: `src/io/image_loader.py`

Purpose: Load PNG sky images and convert them to the format expected by BAST identify stage.

Key functions needed:
```python
def load_sky_image(filepath: str) -> np.ndarray:
    """Load PNG image and convert to numpy array suitable for BAST processing."""
    
def preprocess_for_identify(image: np.ndarray, threshold_sigma: float = 4.0) -> np.ndarray:
    """Apply thresholding to highlight stars (replaces preprocess.py functionality)."""
    
def batch_load_images(directory: str, pattern: str = "*.png") -> List[np.ndarray]:
    """Load multiple test images for batch processing."""
```

Implementation notes:
- Use OpenCV or PIL to load PNG images
- Convert to grayscale if needed
- Apply simple adaptive thresholding to extract star regions
- Return numpy arrays compatible with [`identify.identify()`](src/BAST/identify.py:148)

### Step 4: Create Catalog Loader Module

**File**: `src/io/catalog_loader.py`

Purpose: Manage Hipparcos catalog loading and filtering.

Key functions needed:
```python
def download_hipparcos(output_path: str):
    """Download Hipparcos catalog from online source."""
    
def load_catalog(catalog_path: str, magnitude_threshold: float, fov_degrees: float):
    """Load and prepare catalog using existing Catalog class."""
    
def filter_catalog_by_region(catalog, ra_center: float, dec_center: float, radius: float):
    """Filter catalog to specific sky region for targeted testing."""
```

### Step 5: Create Performance Profiling Module

**File**: `src/benchmark/profiler.py`

Purpose: Profile execution time and resource usage of BAST stages.

Key classes needed:
```python
class StageProfiler:
    """Profile individual BAST algorithm stages."""
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
    
    def profile_centroiding(self, image: np.ndarray, iterations: int = 100):
        """Profile centroiding stage performance."""
        
    def profile_matching(self, bearing_vectors, catalog, iterations: int = 100):
        """Profile matching stage performance."""
        
    def profile_quest(self, observed_vectors, matched_indices, catalog_vectors, iterations: int = 100):
        """Profile QUEST stage performance."""
        
    def generate_report(self, output_path: str):
        """Generate performance report with timing breakdowns."""
```

### Step 6: Create Stage Timer Module

**File**: `src/benchmark/stage_timer.py`

Purpose: Detailed timing utilities for micro-benchmarking.

Key features:
```python
class TimingContext:
    """Context manager for timing code blocks."""
    def __enter__(self): ...
    def __exit__(self, *args): ...
    
def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time a single function call and return result + duration."""
    
def time_repeated(func, iterations: int, *args, **kwargs) -> Dict[str, float]:
    """Run function multiple times and return timing statistics (mean, std, min, max)."""
```

### Step 7: Create Optimizer Pipeline

**File**: `src/pipeline/optimizer_pipeline.py`

Purpose: Main pipeline that orchestrates the full BAST process with instrumentation.

Key class:
```python
class OptimizationPipeline:
    """
    Main pipeline for BAST optimization testing.
    Runs full algorithm chain: PNG -> centroids -> matches -> attitude
    """
    def __init__(self, catalog, config: dict):
        self.catalog = catalog
        self.config = config
        self.profiler = StageProfiler()
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process single image through full pipeline with timing.
        Returns dict with results and performance metrics.
        """
        
    def process_batch(self, image_directory: str) -> pd.DataFrame:
        """Process multiple images and collect performance statistics."""
        
    def identify_bottlenecks(self) -> Dict[str, float]:
        """Analyze which stages consume the most time/resources."""
```

### Step 8: Create Benchmark Scripts

Create three focused benchmark scripts in `benchmarks/`:

**1. `centroid_benchmark.py`**: Test centroiding performance
- Vary image sizes
- Vary number of stars
- Vary star brightness/SNR
- Test different centroiding algorithms (if multiple implementations)

**2. `match_benchmark.py`**: Test matching performance  
- Vary catalog size
- Vary number of observed stars
- Vary FOV radius
- Test triplet search efficiency
- Profile catalog query performance

**3. `quest_benchmark.py`**: Test QUEST algorithm performance
- Vary number of matched star pairs
- Test eigenvalue computation speed
- Profile matrix operations

Each benchmark should:
- Generate performance plots
- Save timing data to CSV
- Identify performance scaling characteristics (O(n), O(n²), etc.)

### Step 9: Create Test Suite

Create unit tests in `tests/` for each stage:

**`test_centroiding.py`**: 
- Test accuracy with known star positions
- Test edge cases (stars at image boundaries, overlapping stars)
- Validate bearing vector calculations

**`test_matching.py`**:
- Test with known star patterns
- Validate triplet matching logic
- Test false positive rate

**`test_quest.py`**:
- Test with known rotation matrices
- Validate quaternion normalization
- Test numerical stability

### Step 10: Create Utility Scripts

**`scripts/download_hipparcos.py`**:
```python
"""Download and prepare Hipparcos catalog for BAST optimization."""
# Download from VizieR or similar
# Convert to CSV format expected by catalog.py
# Filter to appropriate magnitude range
```

**`scripts/run_benchmark_suite.py`**:
```python
"""Run complete benchmark suite and generate comprehensive report."""
# Run all three benchmark scripts
# Collect results
# Generate HTML/PDF report with plots
```

**`scripts/generate_report.py`**:
```python
"""Generate optimization report from benchmark results."""
# Load benchmark data from results/
# Create comparison plots
# Identify optimization opportunities
# Export report
```

### Step 11: Create Configuration Files

**`config/default_config.yaml`**:
```yaml
# BAST Optimizer Configuration

image_loading:
  threshold_sigma: 4.0
  min_star_pixels: 3
  max_star_pixels: 100

centroiding:
  focal_length: 2048.0
  distortion_correction: false

matching:
  angle_tolerance: 0.01
  min_confidence: 0.8
  max_matches: 4

quest:
  min_eigenvalue: 0.5

benchmark:
  iterations: 100
  warmup_iterations: 10
  save_results: true
  output_directory: "results/profiles"
```

**`config/catalog_settings.yaml`**:
```yaml
# Catalog Configuration

catalog:
  source: "HIPPARCOS"
  magnitude_threshold: 6.0
  fov_degrees: 10.0
  cache_triplets: true
  triplet_cache_path: "data/catalogs/hipparcos_triplets.pickle"
```

### Step 12: Create Requirements File

**`requirements.txt`**:
```
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
matplotlib>=3.4.0
astropy>=4.3.0
pyyaml>=5.4.0
tqdm>=4.62.0
pytest>=6.2.0
memory-profiler>=0.60.0
line-profiler>=3.3.0
```

### Step 13: Create README

**`README.md`**:
```markdown
# BAST Algorithm Optimizer

Performance optimization environment for the Basic Autonomous Star Tracker (BAST) algorithm.

## Purpose
Test, profile, and optimize the three core stages of BAST:
1. **Centroiding**: Extract star positions from images
2. **Matching**: Match observed stars to catalog
3. **QUEST**: Determine spacecraft attitude

## Setup
\`\`\`bash
pip install -r requirements.txt
python scripts/download_hipparcos.py
\`\`\`

## Usage
\`\`\`bash
# Run single image through pipeline
python -m src.pipeline.optimizer_pipeline data/test_images/test1.png

# Run full benchmark suite
python scripts/run_benchmark_suite.py

# Generate optimization report
python scripts/generate_report.py
\`\`\`

## Input Format
- PNG images of night sky from synthetic generator
- Expected format: Grayscale, stars as bright points against dark background
- Recommended: 2048x2048 pixels matching typical star tracker sensors

## Output
- Performance profiles for each stage
- Bottleneck analysis
- Optimization recommendations
- Comparative benchmarks
```

---

## Key Optimizations to Explore

Guide Claude Code to create benchmarks that test these optimization opportunities:

### Centroiding Stage ([`identify.py`](src/BAST/identify.py))
1. **Pixel grouping**: Compare connected components vs. other clustering methods
2. **Centroid calculation**: Test different moment algorithms
3. **Vectorization**: Identify loops that can be parallelized
4. **Memory access patterns**: Optimize numpy array operations

### Matching Stage ([`match.py`](src/BAST/match.py))
1. **Catalog search**: Implement spatial indexing (KD-tree, R-tree) for triplet lookup
2. **Triplet comparison**: Optimize angular distance calculations
3. **Early termination**: Add pruning strategies to avoid exhaustive search
4. **Parallel matching**: Test multiprocessing for independent triplet comparisons

### QUEST Stage ([`resolve.py`](src/BAST/resolve.py))
1. **Matrix operations**: Use optimized BLAS/LAPACK implementations
2. **Eigenvalue computation**: Test different algorithms
3. **Numerical stability**: Profile edge cases and numerical precision
4. **Vector operations**: Ensure efficient numpy broadcasting

---

## Additional Recommendations

### Visualization Tools
Create visualization utilities for:
- Star detection overlays on input images
- Match quality heatmaps
- Attitude error plots
- Performance timeline visualizations

### Ground Truth Generation
Develop tools to:
- Generate synthetic test images with known star positions
- Create matched datasets with known attitudes
- Validate algorithm accuracy alongside performance

### Comparison Framework
Set up infrastructure to:
- Compare different algorithm variants
- Track performance over optimization iterations
- A/B test changes before integrating

### Documentation
Maintain documentation of:
- Performance baselines
- Optimization attempts and results
- Algorithm trade-offs
- Best practices discovered

---

## Integration with Synthetic Sky Generator

When the external PNG generator is ready:

1. **Input Interface**: Ensure [`image_loader.py`](src/io/image_loader.py) can read the PNG format
2. **Metadata**: If generator provides ground truth (star positions, attitude), create loader for validation
3. **Batch Processing**: Support processing full test sets generated by the tool
4. **Parametric Testing**: Vary generator parameters (star count, brightness, noise) to test algorithm robustness

---

## Success Criteria

The optimizer codebase is complete when it can:
1. ✅ Load PNG sky images and process them through BAST
2. ✅ Profile each stage with microsecond precision
3. ✅ Generate comprehensive performance reports
4. ✅ Identify algorithmic bottlenecks with data
5. ✅ Support rapid testing of optimization hypotheses
6. ✅ Validate accuracy with ground truth data
7. ✅ Compare performance across algorithm variants
8. ✅ Scale testing from single images to large batches

---

## Notes for Claude Code

- **Copy, don't move**: Keep original BAST files intact in the simulation codebase
- **Minimize dependencies**: Don't copy simulation-specific code ([`preprocess.py`](src/BAST/preprocess.py), radiometry modules)
- **Focus on instrumentation**: Add extensive timing and profiling throughout
- **Make it modular**: Each stage should be testable independently
- **Document thoroughly**: This is for optimization work, so clarity is critical
- **Think about scaling**: Design benchmarks that reveal algorithmic complexity

The goal is a clean, focused environment for making BAST faster and more efficient!