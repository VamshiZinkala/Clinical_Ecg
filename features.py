"""
features.py - Feature Extraction Module
========================================
Extracts numerical features from heartbeat segments for ML classification.

FEATURES EXTRACTED:
  Basic (6 features):
    1. Mean amplitude       - Average voltage level
    2. Standard deviation   - Signal variability
    3. Max amplitude        - Peak voltage (R-wave height)
    4. Min amplitude        - Deepest valley (S-wave depth)
    5. Energy               - Sum of squared values (signal power)
    6. Peak-to-peak         - Range (max - min)

  Advanced (6 features):
    7. Skewness             - Asymmetry of the distribution
    8. Kurtosis             - Tailedness (peakedness) of distribution
    9. Zero crossing rate   - How often signal crosses zero
    10. RR interval (prev)  - Time since previous beat (if available)
    11. RR interval (next)  - Time to next beat (if available)
    12. RR ratio            - Ratio of prev/next RR intervals

Total: Up to 12 features per heartbeat
"""

import numpy as np
from scipy.stats import skew, kurtosis


def extract_features_single(beat):
    """
    Extract features from a single heartbeat segment.

    This function computes both basic statistical features and
    advanced morphological features from one heartbeat window.

    Parameters
    ----------
    beat : np.ndarray
        1D array of voltage values for one heartbeat (e.g., 200 samples).

    Returns
    -------
    features : np.ndarray
        1D array of extracted feature values.
    """
    features = []

    # --- BASIC FEATURES ---

    # 1. Mean: average amplitude (DC level of the beat)
    features.append(np.mean(beat))

    # 2. Standard deviation: measure of signal variability
    #    Higher std = more complex waveform morphology
    features.append(np.std(beat))

    # 3. Maximum amplitude: height of the R-wave
    #    Abnormal beats often have different R-wave heights
    features.append(np.max(beat))

    # 4. Minimum amplitude: depth of the S-wave or Q-wave
    features.append(np.min(beat))

    # 5. Energy: sum of squared values
    #    Represents the total "power" of the heartbeat
    #    Higher energy may indicate ventricular beats (wider QRS)
    features.append(np.sum(beat ** 2))

    # 6. Peak-to-peak amplitude: max - min
    #    Represents the full voltage range of the beat
    features.append(np.max(beat) - np.min(beat))

    # --- ADVANCED FEATURES ---

    # 7. Skewness: measures asymmetry of the waveform
    #    Normal beats tend to have a specific skewness pattern
    #    Positive skew = tail on the right; Negative = tail on the left
    features.append(float(skew(beat)))

    # 8. Kurtosis: measures "tailedness" or peakedness
    #    High kurtosis = sharp peaks (normal QRS)
    #    Low kurtosis = flatter peaks (may indicate abnormality)
    features.append(float(kurtosis(beat)))

    # 9. Zero crossing rate: how often the signal crosses zero
    #    This captures the frequency content of the waveform
    #    More crossings = higher frequency content
    zero_crossings = np.sum(np.diff(np.sign(beat)) != 0)
    features.append(zero_crossings)

    # --- ADD THIS: QRS Width Approximation ---
    # We find the samples where the signal is significantly high (above 40% of peak)
    # This represents the "width" of the main spike.
    threshold = np.max(beat) * 0.4
    width_samples = np.sum(beat > threshold)
    features.append(float(width_samples))

    return np.array(features, dtype=np.float64)


def get_feature_names():
    """
    Return the names of all features in the same order as extract_features_single().

    Returns
    -------
    list of str
        Feature names for labeling and interpretation.
    """
    return [
            'Mean', 'Std_Dev', 'Max_Amplitude', 'Min_Amplitude',
            'Energy', 'Peak_to_Peak', 'Skewness', 'Kurtosis',
            'Zero_Crossing_Rate', 'QRS_Width_Approx' # Added this
        ]


def extract_features_all(beats, r_peaks=None, fs=None):
    """
    Extract features from all heartbeat segments.

    Optionally includes RR interval features if r_peaks and fs are provided.

    Parameters
    ----------
    beats : np.ndarray
        2D array of shape (num_beats, window_size).
    r_peaks : np.ndarray or None
        R-peak indices (for RR interval features).
    fs : int or None
        Sampling frequency (for RR interval features).

    Returns
    -------
    X : np.ndarray
        2D feature matrix of shape (num_beats, num_features).
    feature_names : list of str
        Names of the features.
    """
    all_features = []
    feature_names = get_feature_names()

    # Add RR interval feature names if available
    if r_peaks is not None and fs is not None and len(r_peaks) > 2:
        feature_names.extend(['RR_prev_ms', 'RR_next_ms', 'RR_ratio'])

    for i in range(len(beats)):
        # Extract basic features for this beat
        feat = extract_features_single(beats[i])
        feat_list = list(feat)

        # Add RR interval features if available
        if r_peaks is not None and fs is not None and len(r_peaks) > 2:
            # RR interval to previous beat (in ms)
            if i > 0:
                rr_prev = (r_peaks[i] - r_peaks[i - 1]) / fs * 1000
            else:
                rr_prev = (r_peaks[1] - r_peaks[0]) / fs * 1000  # Use first interval

            # RR interval to next beat (in ms)
            if i < len(r_peaks) - 1:
                rr_next = (r_peaks[i + 1] - r_peaks[i]) / fs * 1000
            else:
                rr_next = (r_peaks[-1] - r_peaks[-2]) / fs * 1000  # Use last interval

            # RR ratio: prev / next (indicates rhythm regularity)
            rr_ratio = rr_prev / rr_next if rr_next > 0 else 1.0

            feat_list.extend([rr_prev, rr_next, rr_ratio])

        all_features.append(feat_list)

    X = np.array(all_features, dtype=np.float64)

    print(f"  Extracted {X.shape[1]} features from {X.shape[0]} heartbeats")
    print(f"  Features: {', '.join(feature_names)}")

    return X, feature_names
