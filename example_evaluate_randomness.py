import numpy as np
from scripts.analyze_randomness import analyze_randomness


if __name__ == "__main__":
    np.random.seed(0)
    
    # Test all input patterns
    test_cases = [
        ("2D grayscale", np.random.randint(0, 256, (32,32)).astype(np.uint8)),
        ("3D RGB", np.random.randint(0, 256, (32, 32, 3)).astype(np.uint8)),
        ("3D RGBA", np.random.randint(0, 256, (32, 32, 4)).astype(np.uint8)),
        ("4D multiple grayscale", np.random.randint(0, 256, (200, 32, 32)).astype(np.uint8)),
        ("4D multiple RGB", np.random.randint(0, 256, (200, 32, 32, 3)).astype(np.uint8)),
    ]
    
    for test_name, img in test_cases:
        print("=" * 60)
        print(f"Test: {test_name}")
        print("=" * 60)
        print(f"Image shape: {img.shape}")
        print(f"Pixel value range: [{img.min()}, {img.max()}]")
        print("-" * 60)
        
        results = analyze_randomness(
            img,
            chi_square_n_bins=256,
            chi_square_max_value=256,
            use_adaptive_fft=True,
            fft_baseline_n_samples=100
        )
        
        print(f"Is Random (Overall): {results['is_random']}")
        print("\nTest Results:")
        print(f"  1) Chi-Square p-value: {results['chi_square']['p_value']:.4f} "
              f"(Random: {results['chi_square']['is_random']})")
        print(f"  2) Correlation: h={results['correlation']['corr_h']:.4f}, "
              f"v={results['correlation']['corr_v']:.4f} "
              f"(Random: {results['correlation']['is_random']})")
        print(f"  3) FFT variance: {results['fft']['variance']:.4f} "
              f"(Random: {results['fft']['is_random']})")
        print(f"  4) Moran's I: {results['morans_i']['value']:.4f} "
              f"(Random: {results['morans_i']['is_random']})")
        print(f"  5) Runs z-score: {results['runs']['z_score']:.4f} "
              f"(Random: {results['runs']['is_random']})")
        print()