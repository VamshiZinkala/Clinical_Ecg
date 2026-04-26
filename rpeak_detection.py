"""
rpeak_detection.py - R-Peak Detection Module
=============================================
Detects R-peaks (the tall spikes in the QRS complex) from ECG signals.

R-peaks are the most prominent feature of an ECG and serve as:
  - Anchor points for heartbeat segmentation
  - Basis for heart rate calculation (HR = 60 / RR_interval)
  - Indicators of rhythm regularity

Methods:
  - Primary: scipy.signal.find_peaks with adaptive thresholds
  - The height threshold is set to a fraction of the signal's max amplitude
  - The distance threshold ensures peaks are at least 250ms apart
    (corresponding to a maximum heart rate of 240 bpm)
"""

import numpy as np
from scipy.signal import find_peaks


def detect_r_peaks(signal, fs, height_factor=0.4, min_distance_ms=200):
    """
    Improved R-peak detection with high sensitivity for 1D signals 
    extracted from images.
    """
    # 1. Calculate a dynamic threshold based on the signal's range
    sig_max = np.max(signal)
    sig_min = np.min(signal)
    sig_range = sig_max - sig_min
    
    # R-peaks are usually in the top 40-50% of the signal range
    threshold = sig_min + (sig_range * height_factor)
    
    # 2. Set distance (min 200ms between beats to allow for fast HR)
    distance = int((min_distance_ms / 1000.0) * fs)

    # 3. Progressive search: if we find 0-1 beats, we lower the threshold automatically
    for sensitivity in [height_factor, height_factor * 0.75, height_factor * 0.5]:
        current_thresh = sig_min + (sig_range * sensitivity)
        
        r_peaks, _ = find_peaks(
            signal, 
            height=current_thresh, 
            distance=distance,
            prominence=sig_range * 0.15 # Ensures it's a sharp spike, not a slow wave    
        )
        
        if len(r_peaks) >= 2:
            break
            
    if len(r_peaks) < 1:
        # Final fallback: just take the highest points
        r_peaks, _ = find_peaks(signal, distance=distance)

    return r_peaks


def compute_rr_intervals(r_peaks, fs):
    """
    Compute RR intervals (time between consecutive R-peaks).

    RR intervals are fundamental for:
      - Heart rate calculation
      - Heart rate variability (HRV) analysis
      - Arrhythmia detection (irregular RR = irregular rhythm)

    Parameters
    ----------
    r_peaks : np.ndarray
        Array of R-peak sample indices.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    dict with keys:
        rr_samples  : np.ndarray - RR intervals in samples
        rr_seconds  : np.ndarray - RR intervals in seconds
        rr_ms       : np.ndarray - RR intervals in milliseconds
        heart_rates : np.ndarray - Instantaneous heart rate in bpm
        mean_hr     : float      - Mean heart rate in bpm
    """
    # RR intervals in samples
    rr_samples = np.diff(r_peaks)

    # Convert to seconds: divide by sampling frequency
    rr_seconds = rr_samples / fs

    # Convert to milliseconds
    rr_ms = rr_seconds * 1000.0

    # Heart rate = 60 / RR_interval_in_seconds
    heart_rates = 60.0 / rr_seconds

    return {
        'rr_samples': rr_samples,
        'rr_seconds': rr_seconds,
        'rr_ms': rr_ms,
        'heart_rates': heart_rates,
        'mean_hr': float(np.mean(heart_rates)) if len(heart_rates) > 0 else 0.0,
    }


def print_rpeak_summary(r_peaks, fs, signal):
    """
    Print a formatted summary of R-peak detection results.

    Parameters
    ----------
    r_peaks : np.ndarray
        Detected R-peak indices.
    fs : int
        Sampling frequency.
    signal : np.ndarray
        ECG signal (for amplitude values).
    """
    rr_data = compute_rr_intervals(r_peaks, fs)

    print(f"\n{'=' * 60}")
    print(f"  R-PEAK DETECTION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total R-peaks detected : {len(r_peaks)}")
    print(f"  Mean heart rate        : {rr_data['mean_hr']:.1f} bpm")
    if len(rr_data['rr_ms']) > 0:
        print(f"  Mean RR interval       : {np.mean(rr_data['rr_ms']):.1f} ms")
        print(f"  RR std deviation       : {np.std(rr_data['rr_ms']):.1f} ms")
        print(f"  Min RR interval        : {np.min(rr_data['rr_ms']):.1f} ms")
        print(f"  Max RR interval        : {np.max(rr_data['rr_ms']):.1f} ms")
    print(f"{'=' * 60}")
