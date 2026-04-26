"""
utils.py - Utility Functions for ECG Analysis System
=====================================================
Contains:
  - ECG basics explanation (educational)
  - Input validation helpers
  - Error handling utilities
  - Console formatting helpers
"""

import os
import numpy as np


# ============================================================
# SECTION 1: ECG BASICS (Educational)
# ============================================================

def print_ecg_basics():
    """
    Print a beginner-friendly explanation of ECG waveforms.
    Useful for educational and documentation purposes.
    """
    info = """
============================================================
  ECG WAVEFORM BASICS
============================================================

  An ECG (Electrocardiogram) records the electrical activity
  of the heart over time. Each heartbeat produces a distinct
  waveform with the following components:

  P Wave   : Atrial depolarization (atria contract)
  Q Wave   : Start of ventricular depolarization (small dip)
  R Wave   : Peak of ventricular depolarization (tall spike)
  S Wave   : End of ventricular depolarization (small dip)
  T Wave   : Ventricular repolarization (recovery)

  The QRS Complex (Q + R + S) represents ventricular
  contraction and is the most prominent feature.

  WHY R-PEAKS MATTER:
  - R-peaks are the tallest, most detectable points
  - The interval between R-peaks (RR interval) gives heart rate
  - Heartbeats are segmented by centering a window on each R-peak
  - Irregular RR intervals can indicate arrhythmia

  HEARTBEAT SEGMENTATION:
  - We extract a fixed-size window (e.g., 200 samples) centered
    on each R-peak
  - This gives us one "heartbeat" per window
  - These segments are used for feature extraction and ML

============================================================
"""
    print(info)


# ============================================================
# SECTION 2: INPUT VALIDATION
# ============================================================

def validate_file_exists(path):
    """
    Check if a file exists at the given path.

    Parameters
    ----------
    path : str
        File path to validate.

    Returns
    -------
    bool
        True if file exists, False otherwise.
    """
    if not os.path.isfile(path):
        print(f"  [ERROR] File not found: {path}")
        return False
    return True


def validate_file_extension(path, allowed_extensions):
    """
    Check if a file has one of the allowed extensions.

    Parameters
    ----------
    path : str
        File path to validate.
    allowed_extensions : list of str
        List of allowed extensions (e.g., ['.png', '.jpg', '.csv']).

    Returns
    -------
    bool
        True if extension is valid, False otherwise.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in allowed_extensions:
        print(f"  [ERROR] Unsupported file type: {ext}")
        print(f"  Allowed: {', '.join(allowed_extensions)}")
        return False
    return True


def validate_signal_length(signal, expected_length=200):
    """
    Check if a signal has the expected length for classification.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array.
    expected_length : int
        Expected number of samples.

    Returns
    -------
    bool
        True if length matches, False otherwise.
    """
    if len(signal) != expected_length:
        print(f"  [ERROR] Signal length is {len(signal)}, expected {expected_length}")
        return False
    return True


def validate_input(path):
    """
    Comprehensive input validation: checks existence and extension.

    Parameters
    ----------
    path : str
        File path to validate.

    Returns
    -------
    str or None
        Input type ('image', 'csv', 'signal') or None if invalid.
    """
    if not validate_file_exists(path):
        return None

    ext = os.path.splitext(path)[1].lower()

    # Image files
    if ext in ['.png', '.jpg', '.jpeg', '.pdf']:
        return 'image'

    # CSV files (heartbeat data)
    elif ext == '.csv':
        return 'csv'

    # WFDB signal files
    elif ext in ['.dat', '.hea']:
        return 'signal'

    else:
        print(f"  [ERROR] Unsupported file type: {ext}")
        print("  Supported: .png, .jpg, .jpeg, .pdf, .csv, .dat, .hea")
        return None


# ============================================================
# SECTION 3: CONSOLE FORMATTING
# ============================================================

def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_subheader(title):
    """Print a formatted sub-section header."""
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def print_step(step_num, total, description):
    """Print a formatted pipeline step indicator."""
    print(f"[{step_num}/{total}] {description}")


def print_success(message):
    """Print a success message."""
    print(f"  [OK] {message}")


def print_error(message):
    """Print an error message."""
    print(f"  [ERROR] {message}")


def print_warning(message):
    """Print a warning message."""
    print(f"  [WARN] {message}")
