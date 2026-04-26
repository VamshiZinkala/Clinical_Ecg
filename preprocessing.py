"""
preprocessing.py - Signal & Image Preprocessing Module
=======================================================
Contains:
  1. Butterworth bandpass filter for ECG signals (0.5-40 Hz)
  2. Image preprocessing (CLAHE + adaptive threshold)

KEY CONCEPTS:
  - Nyquist Frequency: Half the sampling rate. For MIT-BIH at 360 Hz,
    the Nyquist frequency is 180 Hz. We can only reliably filter
    frequencies below 180 Hz.
  - Bandpass Filter: Keeps frequencies between lowcut and highcut,
    removes baseline wander (< 0.5 Hz) and high-frequency noise (> 40 Hz).
  - filtfilt: Applies the filter forward and backward, producing
    zero phase distortion (no time delay in the output).
"""

import numpy as np
import cv2
from scipy.signal import butter, filtfilt


# ============================================================
# MODULE 3: SIGNAL PREPROCESSING (Butterworth Bandpass)
# ============================================================

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter.

    The Butterworth filter has a maximally flat frequency response
    in the passband, making it ideal for ECG signal processing.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz. For ECG, typically 0.5 Hz
        to remove baseline wander caused by breathing/movement.
    highcut : float
        Upper cutoff frequency in Hz. For ECG, typically 40 Hz
        to remove high-frequency noise (muscle artifacts, power line).
    fs : float
        Sampling frequency of the signal in Hz.
    order : int
        Filter order. Higher order = sharper cutoff but more
        computational cost. 4 is a good balance for ECG.

    Returns
    -------
    b, a : np.ndarray
        Numerator (b) and denominator (a) polynomials of the filter.
    """
    # Nyquist frequency = half the sampling rate
    # This is the maximum frequency we can represent without aliasing
    nyquist = 0.5 * fs

    # Normalize cutoff frequencies by Nyquist frequency
    # butter() expects frequencies in the range [0, 1] where 1 = Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design the Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    return b, a


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """
    Apply a Butterworth bandpass filter to an ECG signal.

    This removes:
      - Baseline wander (frequencies below 0.5 Hz) caused by
        patient breathing and body movement
      - High-frequency noise (above 40 Hz) from muscle artifacts
        and power line interference (50/60 Hz)

    Parameters
    ----------
    signal : np.ndarray
        Raw 1D ECG signal.
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Lower cutoff frequency (default: 0.5 Hz).
    highcut : float
        Upper cutoff frequency (default: 40.0 Hz).
    order : int
        Filter order (default: 4).

    Returns
    -------
    np.ndarray
        Filtered ECG signal with same length as input.
    """
    # Get filter coefficients
    b, a = butter_bandpass(lowcut, highcut, fs, order)

    # Apply the filter using filtfilt (zero-phase filtering)
    # filtfilt applies the filter twice: once forward, once backward
    # This eliminates any phase shift introduced by the filter
    filtered = filtfilt(b, a, signal)

    return filtered


def preprocess_signal(signal, fs):
    # 1. Bandpass Filter (0.5 - 40Hz)
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 40.0 / nyq
    b, a = butter(3, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    
    # 2. Smoothing Filter (Removes 'pixel jitters' from images)
    window = 11
    filtered = np.convolve(filtered, np.ones(window)/window, mode='same')
    
    return filtered - np.mean(filtered) 


# ============================================================
# IMAGE PREPROCESSING (for ECG image input)
# ============================================================

def preprocess_image(img):
    # Grayscale for grid calibration
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Red Channel masking to remove the red grid
    r_channel = img[:, :, 2]
    
    # Hard threshold to isolate the black trace
    _, binary = cv2.threshold(r_channel, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Noise cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary, gray