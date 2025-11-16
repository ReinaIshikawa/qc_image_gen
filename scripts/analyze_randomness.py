import numpy as np
from scipy.stats import chisquare
from scipy.fft import fft2


def adaptive_baseline(image_shape=(16, 16), n_samples=100, random_seed=None):
    """
    Compute FFT variance threshold from random noise baseline.
    
    Returns:
        float: Threshold as mean + 3*std of baseline FFT variances
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate baseline from random noise images
    baseline = [
        fft_spectral_flatness(np.random.randint(0, 256, image_shape).astype(np.uint8))
        for _ in range(n_samples)
    ]
    
    fft_thr = np.mean(baseline) + 3 * np.std(baseline)
    return fft_thr


def chi_square_goodness_of_fit_test(image, n_bins=256, max_value=256):
    """
    Test if pixel intensities are uniformly distributed (chi-square goodness-of-fit).
    
    Returns:
        float: p-value from chi-square distribution (n_bins-1 df)
    """
    image_flat = image.flatten()
    N = image_flat.size
    
    observed_counts, _ = np.histogram(image_flat, bins=n_bins, range=(0, max_value))
    expected_count = N / n_bins
    _, p_value = chisquare(observed_counts, f_exp=expected_count)
    
    return p_value


def adjacent_pixel_correlation(image):
    """
    Compute correlation between adjacent pixels (horizontal and vertical).
    
    Returns:
        tuple: (corr_h, corr_v) - horizontal and vertical correlation coefficients
    """
    image = image.astype(float)
    
    x_h1, x_h2 = image[:, :-1].flatten(), image[:, 1:].flatten()
    corr_h = np.corrcoef(x_h1, x_h2)[0, 1] if len(x_h1) > 1 else np.nan
    
    x_v1, x_v2 = image[:-1, :].flatten(), image[1:, :].flatten()
    corr_v = np.corrcoef(x_v1, x_v2)[0, 1] if len(x_v1) > 1 else np.nan
    
    return corr_h, corr_v


def fft_spectral_flatness(image, exclude_dc=True, window=True):
    """
    Compute FFT spectral flatness (variance of frequency spectrum).
    
    Returns:
        float: Variance of FFT magnitude spectrum
    """
    image = image.astype(float)
    
    if window:
        image = image - np.mean(image)
        rows, cols = image.shape
        hamming_2d = np.outer(np.hamming(rows), np.hamming(cols))
        image = image * hamming_2d
    
    fft_result = fft2(image)
    fft_magnitude = np.abs(fft_result)
    
    if exclude_dc:
        fft_magnitude[0, 0] = 0
    
    return np.var(fft_magnitude)


def morans_i(image, neighborhood='4-nearest'):
    """
    Compute Moran's I statistic for spatial autocorrelation.
    
    Formula: I = (N / Σ_ij w_ij) * [ Σ_ij w_ij (x_i - x̄)(x_j - x̄) ] / Σ_i (x_i - x̄)²
    
    Args:
        neighborhood: '4-nearest' or '8-nearest'
    
    Returns:
        float: Moran's I statistic (≈0 for spatial randomness)
    """
    x = image.astype(float)
    n = x.size
    x_mean = np.mean(x)
    
    if neighborhood == '4-nearest':
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif neighborhood == '8-nearest':
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        raise ValueError("neighborhood must be '4-nearest' or '8-nearest'")
    
    rows, cols = x.shape
    w_sum = 0
    numerator_sum = 0
    x_vec = x.flatten()
    x_centered = x_vec - x_mean
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    nidx = ni * cols + nj
                    w_sum += 1
                    numerator_sum += x_centered[idx] * x_centered[nidx]
    
    denominator = np.sum(x_centered ** 2)
    I = (n / w_sum) * (numerator_sum / denominator) if w_sum > 0 and denominator > 0 else 0.0
    
    return I


def runs_test(image, threshold=128):
    """
    Wald-Wolfowitz runs test on binarized image.
    
    Returns:
        float: z-score (≈0 for random sequence)
    """
    binary = (image >= threshold).astype(int)
    sequence = binary.flatten()
    
    num_runs = 1 + np.sum(np.diff(sequence) != 0)
    n0 = np.count_nonzero(sequence == 0)
    n1 = np.count_nonzero(sequence == 1)
    n = n0 + n1
    
    expected_runs = 1 + (2 * n0 * n1) / n if n > 0 else 0.0
    variance_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1)) if n > 1 else 0.0
    z_score = (num_runs - expected_runs) / np.sqrt(variance_runs) if variance_runs > 0 else 0.0
    
    return z_score


def analyze_single_image(image_2d,
                        chi_square_n_bins=256,
                        chi_square_max_value=256):
    """
    Analyze randomness for a single 2D image.
    
    Returns:
        dict: Results with test-specific values
    """
    chi_p_value = chi_square_goodness_of_fit_test(
        image_2d, n_bins=chi_square_n_bins, max_value=chi_square_max_value)
    corr_h, corr_v = adjacent_pixel_correlation(image_2d)
    fft_var = fft_spectral_flatness(image_2d)
    moran_I = morans_i(image_2d)
    runs_z = runs_test(image_2d)
    
    return {
        "chi_square": {
            "p_value": chi_p_value
        },
        "correlation": {
            "corr_h": corr_h,
            "corr_v": corr_v
        },
        "fft": {
            "variance": fft_var
        },
        "morans_i": {
            "value": moran_I
        },
        "runs": {
            "z_score": runs_z
        }
    }


def analyze_single_image_with_channels(image_3d,
                                      chi_square_n_bins=256,
                                      chi_square_max_value=256):
    """
    Analyze randomness for a 3D image (e.g., RGB) by evaluating each channel separately.
    
    Returns:
        dict: Results with test-specific values (averaged across channels)
    """
    n_channels = image_3d.shape[2]
    
    # Evaluate each channel separately
    channel_results = []
    for c in range(n_channels):
        channel_2d = image_3d[:, :, c]
        result = analyze_single_image(
            channel_2d,
            chi_square_n_bins=chi_square_n_bins,
            chi_square_max_value=chi_square_max_value
        )
        channel_results.append(result)
    
    # Average results across channels
    chi_p_values = [r["chi_square"]["p_value"] for r in channel_results]
    valid_chi_p = [p for p in chi_p_values if not np.isnan(p)]
    chi_p_value = np.mean(valid_chi_p) if len(valid_chi_p) > 0 else np.nan
    
    corr_h_values = [r["correlation"]["corr_h"] for r in channel_results]
    corr_v_values = [r["correlation"]["corr_v"] for r in channel_results]
    corr_h = np.nanmean(corr_h_values)
    corr_v = np.nanmean(corr_v_values)
    
    fft_vars = [r["fft"]["variance"] for r in channel_results]
    fft_var = np.mean(fft_vars)
    
    moran_I_values = [r["morans_i"]["value"] for r in channel_results]
    moran_I = np.mean(moran_I_values)
    
    runs_z_values = [r["runs"]["z_score"] for r in channel_results]
    runs_z = np.mean(runs_z_values)
    
    return {
        "chi_square": {
            "p_value": chi_p_value
        },
        "correlation": {
            "corr_h": corr_h,
            "corr_v": corr_v
        },
        "fft": {
            "variance": fft_var
        },
        "morans_i": {
            "value": moran_I
        },
        "runs": {
            "z_score": runs_z
        }
    }


def analyze_randomness(image, 
                     chi_square_p_threshold=0.05,
                     corr_threshold=0.10,
                     moran_i_threshold=0.02,
                     runs_z_threshold=1.96,
                     fft_variance_threshold=None,
                     chi_square_n_bins=256,
                     chi_square_max_value=256,
                     use_adaptive_fft=True,
                     fft_baseline_n_samples=100,
                     fft_baseline_random_seed=None):
    """
    Evaluate image randomness using 5 statistical tests.
    
    Args:
        image: 2D (H, W), 3D (H, W, C), or 4D (N, H, W) / (N, H, W, C) array.
               For 4D arrays, each image is evaluated individually and the proportion
               of random images is computed. Other metrics are averaged.
    
    Returns:
        dict: Results with "is_random" (bool) and test-specific dicts containing
              values and "is_random" flags. For 4D arrays, chi_square["p_value"] is
              the proportion of random images (0.0-1.0), while other metrics are averaged.
    """
    # Handle multiple images (4D array)
    if image.ndim == 4:
        n_images = image.shape[0]
        
        # Analyze each image independently
        # This avoids the problem where large sample sizes make even small deviations statistically significant
        results_list = []
        image_shape = None
        
        for i in range(n_images):
            img = image[i]
            if img.ndim == 3:
                result = analyze_single_image_with_channels(
                    img,
                    chi_square_n_bins=chi_square_n_bins,
                    chi_square_max_value=chi_square_max_value
                )
                if image_shape is None:
                    image_shape = img.shape[:2]
            else:
                result = analyze_single_image(
                    img,
                    chi_square_n_bins=chi_square_n_bins,
                    chi_square_max_value=chi_square_max_value
                )
                if image_shape is None:
                    image_shape = img.shape
            results_list.append(result)
        
        # For multiple images, compute proportion of random images (statistically more appropriate than combining p-values)
        chi_p_values = [r["chi_square"]["p_value"] for r in results_list]
        valid_p_values = [p for p in chi_p_values if not np.isnan(p)]
        
        if len(valid_p_values) > 0:
            n_random = sum(1 for p in valid_p_values if p > chi_square_p_threshold)
            chi_p_value = n_random / len(valid_p_values)
        else:
            chi_p_value = np.nan
        
        # Average other metrics across images
        corr_h = np.nanmean([r["correlation"]["corr_h"] for r in results_list])
        corr_v = np.nanmean([r["correlation"]["corr_v"] for r in results_list])
        fft_var = np.mean([r["fft"]["variance"] for r in results_list])
        moran_I = np.mean([r["morans_i"]["value"] for r in results_list])
        runs_z = np.mean([r["runs"]["z_score"] for r in results_list])
    elif image.ndim == 3:
        image_shape = image.shape[:2]
        result = analyze_single_image_with_channels(
            image,
            chi_square_n_bins=chi_square_n_bins,
            chi_square_max_value=chi_square_max_value
        )
        chi_p_value = result["chi_square"]["p_value"]
        corr_h = result["correlation"]["corr_h"]
        corr_v = result["correlation"]["corr_v"]
        fft_var = result["fft"]["variance"]
        moran_I = result["morans_i"]["value"]
        runs_z = result["runs"]["z_score"]
    elif image.ndim == 2:
        image_shape = image.shape
        result = analyze_single_image(
            image,
            chi_square_n_bins=chi_square_n_bins,
            chi_square_max_value=chi_square_max_value
        )
        chi_p_value = result["chi_square"]["p_value"]
        corr_h = result["correlation"]["corr_h"]
        corr_v = result["correlation"]["corr_v"]
        fft_var = result["fft"]["variance"]
        moran_I = result["morans_i"]["value"]
        runs_z = result["runs"]["z_score"]
    else:
        raise ValueError(f"Image must be 2D, 3D, or 4D array, got {image.ndim}D array with shape {image.shape}")
    
    # Calculate FFT threshold if needed
    if fft_variance_threshold is None:
        if use_adaptive_fft and image_shape is not None:
            fft_variance_threshold = adaptive_baseline(
                image_shape=image_shape,
                n_samples=fft_baseline_n_samples,
                random_seed=fft_baseline_random_seed
            )
        else:
            fft_variance_threshold = 10000.0
    
    # Evaluate randomness based on thresholds
    is_chi_random = (
        (chi_p_value >= 0.5 if image.ndim == 4 else chi_p_value > chi_square_p_threshold)
        if not np.isnan(chi_p_value) else False
    )
    is_corr_random = abs(corr_h) < corr_threshold and abs(corr_v) < corr_threshold
    is_fft_random = fft_var < fft_variance_threshold
    is_moran_random = abs(moran_I) < moran_i_threshold
    is_runs_random = abs(runs_z) < runs_z_threshold
    
    is_random = (is_chi_random and is_corr_random and is_fft_random and 
                 is_moran_random and is_runs_random)
    
    return {
        "is_random": is_random,
        "chi_square": {
            "p_value": chi_p_value,
            "is_random": is_chi_random
        },
        "correlation": {
            "corr_h": corr_h,
            "corr_v": corr_v,
            "is_random": is_corr_random
        },
        "fft": {
            "variance": fft_var,
            "is_random": is_fft_random
        },
        "morans_i": {
            "value": moran_I,
            "is_random": is_moran_random
        },
        "runs": {
            "z_score": runs_z,
            "is_random": is_runs_random
        }
    }