import numpy as np
import matplotlib.pyplot as plt
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.multi_star.synthetic_catalog import SyntheticCatalogBuilder
from src.multi_star.multi_star_pipeline import MultiStarPipeline

def run_demo():
    """Run an interactive demo of the 3-star triangle matching."""
    print("Running 3-Star Triangle Matching Demo...")

    # 1. Initialize pipelines and builders
    pipeline = StarTrackerPipeline()
    multi_star_pipeline = MultiStarPipeline(pipeline)
    catalog_builder = SyntheticCatalogBuilder()

    # 2. Create 3-star catalog
    catalog = catalog_builder.create_triangle_catalog()
    print(f"Created synthetic catalog with {len(catalog)} stars.")

    # 3. Load PSF data
    psf_data = pipeline.load_psf_data("PSF_sims/Gen_1", "*0_deg*")[0.0]
    print("Loaded 0-degree PSF data.")

    # 4. Run the multi-star analysis
    results = multi_star_pipeline.run_multi_star_analysis(catalog, psf_data)
    print("Multi-star analysis complete.")

    # 5. Display results
    print("\n--- Results ---")
    print(f"Detected Stars: {results['detected_stars']}")
    print(f"Star Matches: {len(results['star_matches'])}")
    print(f"Validation Status: {results['validation']['status']}")

    if results['validation']['status'] == 'passed':
        print(f"Confidence: {results['star_matches'][0].confidence}")

    # 6. Visualize the scene
    plt.figure(figsize=(10, 10))
    plt.imshow(results['scene_data']['detector_image'], cmap='gray')
    plt.title("3-Star Synthetic Scene")
    plt.colorbar(label="Intensity")
    plt.show()

if __name__ == '__main__':
    run_demo()