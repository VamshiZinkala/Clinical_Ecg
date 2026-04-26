"""
labels.py - Label Processing Module for MIT-BIH Dataset
========================================================
Maps MIT-BIH annotation symbols to binary classification labels.

MIT-BIH ANNOTATION SYMBOLS:
  The MIT-BIH database uses single-character codes for beat types:

  NORMAL BEATS (Label = 0):
    'N' - Normal beat
    'L' - Left bundle branch block beat
    'R' - Right bundle branch block beat
    'e' - Atrial escape beat
    'j' - Nodal (junctional) escape beat

  ABNORMAL BEATS (Label = 1):
    'V' - Premature ventricular contraction (PVC)
    'A' - Atrial premature beat
    'a' - Aberrated atrial premature beat
    'S' - Supraventricular premature beat
    'F' - Fusion of ventricular and normal beat
    'J' - Nodal (junctional) premature beat
    'E' - Ventricular escape beat
    'f' - Fusion of paced and normal beat

  We focus on the 5 most common types: N, L, R, V, A
  This covers >95% of all beats in the MIT-BIH database.
"""

import numpy as np


# Define which annotation symbols are Normal vs Abnormal
NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}
ABNORMAL_SYMBOLS = {'V', 'A', 'a', 'S', 'F', 'J', 'E', 'f'}

# Combined set of all valid symbols we process
VALID_SYMBOLS = NORMAL_SYMBOLS | ABNORMAL_SYMBOLS


def map_label(symbol):
    """
    Map a single MIT-BIH annotation symbol to a binary label.

    Parameters
    ----------
    symbol : str
        MIT-BIH annotation symbol (e.g., 'N', 'V', 'A').

    Returns
    -------
    int or None
        0 for Normal, 1 for Abnormal, None for unknown/non-beat symbols.
    """
    if symbol in NORMAL_SYMBOLS:
        return 0  # Normal
    elif symbol in ABNORMAL_SYMBOLS:
        return 1  # Abnormal
    else:
        return None  # Non-beat annotation (e.g., '+', '~', '|')


def process_labels(annotation, r_peaks, valid_indices=None):
    """
    Process MIT-BIH annotations and align them with detected R-peaks.

    The MIT-BIH annotations provide beat labels at specific sample positions.
    This function:
      1. Finds the annotation closest to each R-peak
      2. Maps the annotation symbol to binary (Normal=0, Abnormal=1)
      3. Filters out non-beat annotations

    Parameters
    ----------
    annotation : wfdb.Annotation
        WFDB annotation object with .sample and .symbol attributes.
    r_peaks : np.ndarray
        Detected R-peak sample indices.
    valid_indices : list of int or None
        If provided, only process R-peaks at these indices
        (from segmentation step where boundary beats were skipped).

    Returns
    -------
    labels : np.ndarray
        Binary labels array (0=Normal, 1=Abnormal).
    label_mask : np.ndarray of bool
        Boolean mask: True for beats with valid labels.
        Use this to filter the corresponding feature matrix.
    """
    # Get annotation positions and symbols
    ann_samples = annotation.sample   # Sample indices of annotations
    ann_symbols = annotation.symbol   # Beat type symbols

    # If valid_indices provided, only process those R-peaks
    if valid_indices is not None:
        peaks_to_process = r_peaks[valid_indices]
    else:
        peaks_to_process = r_peaks

    labels = []
    label_mask = []

    for peak in peaks_to_process:
        # Find the annotation closest to this R-peak
        # The annotation should be within a small window of the R-peak
        distances = np.abs(ann_samples - peak)
        closest_idx = np.argmin(distances)

        # Only use annotation if it's within 50 samples of the R-peak
        # (annotations and detected peaks may not be at exact same position)
        if distances[closest_idx] <= 50:
            symbol = ann_symbols[closest_idx]
            label = map_label(symbol)

            if label is not None:
                labels.append(label)
                label_mask.append(True)
            else:
                labels.append(-1)  # Placeholder for unknown
                label_mask.append(False)
        else:
            labels.append(-1)
            label_mask.append(False)

    labels = np.array(labels)
    label_mask = np.array(label_mask)

    # Count label distribution
    valid_labels = labels[label_mask]
    n_normal = np.sum(valid_labels == 0)
    n_abnormal = np.sum(valid_labels == 1)
    n_skipped = np.sum(~label_mask)

    print(f"  Label distribution:")
    print(f"    Normal (0)   : {n_normal}")
    print(f"    Abnormal (1) : {n_abnormal}")
    print(f"    Skipped      : {n_skipped}")

    return labels, label_mask
