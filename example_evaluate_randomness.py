import numpy as np
from scripts.analyze_randomness import analyze_randomness
from scripts.utils import yaml_load

if __name__ == "__main__":
      np.random.seed(0)
      config = yaml_load("config.yaml")
      r_config = config["randomness_evaluation"]

      # Test all input patterns
      quantum_random_data = np.load("samples/quantum_random_32x32x4_fp32.npy")
      # Convert float32 to uint8 (0-255 range) if needed
      if quantum_random_data.dtype != np.uint8:
            # Normalize to 0-255 range
            quantum_random_data = (quantum_random_data - quantum_random_data.min()) / (quantum_random_data.max() - quantum_random_data.min() + 1e-10)
            quantum_random_data = (quantum_random_data * 255).astype(np.uint8)
      
      test_cases = [
            ("2D grayscale", np.random.randint(0, 256, (32,32)).astype(np.uint8)),
            ("3D RGB", np.random.randint(0, 256, (32, 32, 3)).astype(np.uint8)),
            ("3D RGBA", np.random.randint(0, 256, (32, 32, 4)).astype(np.uint8)),
            ("4D multiple grayscale", np.random.randint(0, 256, (200, 32, 32)).astype(np.uint8)),
            ("4D multiple RGB", np.random.randint(0, 256, (200, 32, 32, 3)).astype(np.uint8)),
            ("Quantum random 32x32x4", quantum_random_data),
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
                  chi_square_p_threshold=r_config["chi_square_p_threshold"],
                  corr_threshold=r_config["corr_threshold"],
                  moran_i_threshold=r_config["moran_i_threshold"],
                  runs_z_threshold=r_config["runs_z_threshold"],
                  fft_variance_threshold=r_config["fft_variance_threshold"],
                  chi_square_n_bins=r_config["chi_square_n_bins"],
                  chi_square_max_value=r_config["chi_square_max_value"],
                  use_adaptive_fft=r_config["use_adaptive_fft"],
                  fft_baseline_n_samples=r_config["fft_baseline_n_samples"],
                  fft_baseline_random_seed=r_config["fft_baseline_random_seed"]
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

      """expected output
      ============================================================
      Test: 2D grayscale
      ============================================================
      Image shape: (32, 32)
      Pixel value range: [0, 255]
      ------------------------------------------------------------
      Is Random (Overall): False

      Test Results:
      1) Chi-Square p-value: 0.9002 (Random: True)
      2) Correlation: h=-0.0167, v=-0.0302 (Random: True)
      3) FFT variance: 156232.7953 (Random: True)
      4) Moran's I: -0.0235 (Random: False)
      5) Runs z-score: -0.0386 (Random: True)

      ============================================================
      Test: 3D RGB
      ============================================================
      Image shape: (32, 32, 3)
      Pixel value range: [0, 255]
      ------------------------------------------------------------
      Is Random (Overall): True

      Test Results:
      1) Chi-Square p-value: 0.4274 (Random: True)
      2) Correlation: h=-0.0139, v=-0.0094 (Random: True)
      3) FFT variance: 183800.9546 (Random: True)
      4) Moran's I: -0.0116 (Random: True)
      5) Runs z-score: 0.0331 (Random: True)

      ============================================================
      Test: 3D RGBA
      ============================================================
      Image shape: (32, 32, 4)
      Pixel value range: [0, 255]
      ------------------------------------------------------------
      Is Random (Overall): True

      Test Results:
      1) Chi-Square p-value: 0.3518 (Random: True)
      2) Correlation: h=-0.0035, v=-0.0162 (Random: True)
      3) FFT variance: 182223.5988 (Random: True)
      4) Moran's I: -0.0098 (Random: True)
      5) Runs z-score: -0.0987 (Random: True)

      ============================================================
      Test: 4D multiple grayscale
      ============================================================
      Image shape: (200, 32, 32)
      Pixel value range: [0, 255]
      ------------------------------------------------------------
      Is Random (Overall): True

      Test Results:
      1) Chi-Square p-value: 0.4314 (Random: True)
      2) Correlation: h=0.0028, v=-0.0010 (Random: True)
      3) FFT variance: 1151131.0090 (Random: True)
      4) Moran's I: 0.0009 (Random: True)
      5) Runs z-score: -0.2296 (Random: True)

      ============================================================
      Test: 4D multiple RGB
      ============================================================
      Image shape: (200, 32, 32, 3)
      Pixel value range: [0, 255]
      ------------------------------------------------------------
      Is Random (Overall): True

      Test Results:
      1) Chi-Square p-value: 1.0000 (Random: True)
      2) Correlation: h=-0.0008, v=-0.0011 (Random: True)
      3) FFT variance: 177275.8147 (Random: True)
      4) Moran's I: -0.0009 (Random: True)
      5) Runs z-score: 0.0379 (Random: True)

      ============================================================
      Test: Quantum random 32x32x4
      ============================================================
      Image shape: (32, 32, 4)
      Pixel value range: [0, 255]
      ------------------------------------------------------------
      Is Random (Overall): True

      Test Results:
      1) Chi-Square p-value: 0.5877 (Random: True)
      2) Correlation: h=0.0181, v=-0.0007 (Random: True)
      3) FFT variance: 168653.7726 (Random: True)
      4) Moran's I: 0.0087 (Random: True)
      5) Runs z-score: -0.3136 (Random: True)
      """