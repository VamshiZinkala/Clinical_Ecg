"""
loader.py - Data Loading Module for ECG Analysis System
========================================================
Handles loading ECG data from multiple sources:
  1. MIT-BIH Arrhythmia Database (via wfdb library)
  2. ECG images (PNG, JPG, PDF via OpenCV)
  3. CSV files (pre-extracted heartbeat signals)

MIT-BIH Database:
  - Contains 48 half-hour recordings of two-channel ambulatory ECG
  - Sampled at 360 Hz
  - Each record has signal data (.dat) and annotations (.atr)
  - We use Lead MLII (column 0) for analysis
"""

import os
import cv2
import numpy as np
import pandas as pd

# wfdb is used for loading MIT-BIH PhysioNet data
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("  [WARN] wfdb not installed. MIT-BIH loading disabled.")
    print("         Install with: pip install wfdb")

# pdf2image is optional, only needed for PDF ECG images
try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ============================================================
# MODULE 1: MIT-BIH DATA LOADING
# ============================================================

def load_mitbih_record(record_number=100, lead=0, data_dir=None):
    """
    Load a record from the MIT-BIH Arrhythmia Database.

    The function downloads the record from PhysioNet if not found locally.
    It returns the raw signal, annotations, and sampling frequency.

    Parameters
    ----------
    record_number : int
        MIT-BIH record number (e.g., 100, 101, ..., 234).
        Default is 100 (a commonly used normal sinus rhythm record).
    lead : int
        Which ECG lead to use. 0 = MLII (default), 1 = V1/V5.
    data_dir : str or None
        Local directory containing .dat/.hea/.atr files.
        If None, downloads from PhysioNet automatically.

    Returns
    -------
    dict with keys:
        signal     : np.ndarray  - 1D ECG signal (selected lead)
        annotation : wfdb.Annotation - Beat annotations
        fs         : int         - Sampling frequency (360 Hz)
        record_name: str         - Record identifier
    """
    if not WFDB_AVAILABLE:
        raise ImportError(
            "wfdb library is required for MIT-BIH loading. "
            "Install with: pip install wfdb"
        )

    record_name = str(record_number)

    try:
        if data_dir and os.path.isfile(os.path.join(data_dir, f"{record_name}.dat")):
            # Load from local directory
            record = wfdb.rdrecord(os.path.join(data_dir, record_name))
            annotation = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')
        else:
            # Download from PhysioNet (MIT-BIH Arrhythmia Database)
            record = wfdb.rdrecord(record_name, pn_dir='mitdb')
            annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')

        # Extract the signal from the selected lead
        # p_signal is a 2D array: (num_samples, num_leads)
        signal = record.p_signal[:, lead]

        # Get sampling frequency (should be 360 Hz for MIT-BIH)
        fs = record.fs

        print(f"  Loaded MIT-BIH record {record_name}")
        print(f"  Signal length: {len(signal)} samples ({len(signal)/fs:.1f} seconds)")
        print(f"  Sampling rate: {fs} Hz")
        print(f"  Lead: {record.sig_name[lead]}")
        print(f"  Annotations: {len(annotation.sample)} beats")

        return {
            'signal': signal,
            'annotation': annotation,
            'fs': fs,
            'record_name': record_name,
        }

    except Exception as e:
        raise ValueError(f"Failed to load MIT-BIH record {record_name}: {e}")


# ============================================================
# MODULE 2: ECG IMAGE LOADING
# ============================================================

def load_ecg_image(path):
    """
    Load an ECG image from file (PNG, JPG, JPEG, or PDF).

    For PDF files, the first page is converted to an image at 300 DPI.
    For image files, OpenCV is used to read the file.

    Parameters
    ----------
    path : str
        Path to the ECG image file.

    Returns
    -------
    np.ndarray
        BGR image array (OpenCV format).

    Raises
    ------
    ValueError
        If file format is unsupported or image cannot be loaded.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        if not PDF_AVAILABLE:
            raise ImportError(
                "pdf2image is required for PDF files. "
                "Install with: pip install pdf2image"
            )
        # Convert first page of PDF to image at 300 DPI
        pages = convert_from_path(path, dpi=300)
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)

    elif ext in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(path)

    else:
        raise ValueError(f"Unsupported image format: {ext}")

    if img is None:
        raise ValueError(f"Image could not be loaded from: {path}")

    print(f"  Loaded ECG image: {path}")
    print(f"  Image dimensions: {img.shape[1]}x{img.shape[0]} pixels")

    return img


# ============================================================
# MODULE 3: CSV SIGNAL LOADING
# ============================================================

def load_csv_signal(path):
    """
    Load an ECG heartbeat signal from a CSV file.

    Expected format: Single column of voltage values (200 samples)
    or two columns (Time, Voltage).

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    np.ndarray
        1D signal array.
    """
    try:
        df = pd.read_csv(path)

        if df.shape[1] == 1:
            # Single column: just voltage values
            signal = df.iloc[:, 0].values
        elif df.shape[1] >= 2:
            # Two+ columns: assume second column is voltage
            signal = df.iloc[:, 1].values
        else:
            raise ValueError("CSV file must have at least 1 column")

        signal = signal.astype(np.float64)

        print(f"  Loaded CSV signal: {path}")
        print(f"  Signal length: {len(signal)} samples")

        return signal

    except Exception as e:
        raise ValueError(f"Failed to load CSV signal from {path}: {e}")