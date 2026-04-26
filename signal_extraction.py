"""
signal_extraction.py - Converts a 2D binary image into a 1D digital signal array
"""
import numpy as np
from scipy.interpolate import interp1d

def extract_signal_from_image(binary, px_mm_x, px_mm_y):
    """
    Extracts the 1D signal from the binary image trace.
    Returns raw time and voltage arrays.
    """
    # Find all white pixels (the ECG trace)
    coords = np.column_stack(np.where(binary == 255))
    
    if len(coords) == 0:
        return np.array([]), np.array([])

    # Group y-coordinates by their x-coordinates
    # (Since the image might have thick lines, we average the y-values at each x)
    unique_x = np.unique(coords[:, 1])
    y_values = []
    
    for x in unique_x:
        y_at_x = coords[coords[:, 1] == x, 0]
        y_values.append(np.mean(y_at_x))
        
    y_values = np.array(y_values)
    
    # Invert Y so the signal goes up instead of down (standard image coordinates)
    max_y = binary.shape[0]
    y_values = max_y - y_values

    # Convert pixels to physical units (optional, but good for scaling)
    # Assuming standard ECG paper speed of 25 mm/s
    time_raw = (unique_x / px_mm_x) / 25.0  
    voltage_raw = y_values / px_mm_y
    
    return time_raw, voltage_raw

import numpy as np
from scipy.interpolate import interp1d

def interpolate_signal(time_raw, voltage_raw, target_fs=360):
    """
    Resamples the raw signal to 360 Hz. Includes a robust fallback mechanism
    if grid calibration fails on cropped images.
    """
    if len(time_raw) < 2:
        return time_raw, voltage_raw
        
    total_time = time_raw[-1] - time_raw[0]
    
    # --- THE FALLBACK ---
    # If the grid calibration failed and thinks the image is ridiculously short (< 0.5s)
    # or ridiculously long (> 20s), we override it.
    # We assume the image represents a standard 5.0 second ECG strip.
    if total_time < 0.5 or total_time > 20:
        total_time = 5.0
    # --------------------

    num_samples = int(total_time * target_fs)
    
    # Interpolate
    f = interp1d(time_raw, voltage_raw, kind='cubic', fill_value="extrapolate")
    time_uniform = np.linspace(time_raw[0], time_raw[-1], num_samples)
    voltage_uniform = f(time_uniform)
    
    return time_uniform, voltage_uniform

def normalize_signal(signal):
    """
    Standardizes a heartbeat to have zero mean and unit variance.
    This prevents the ML model from getting confused by different image darkness/thickness.
    """
    if len(signal) == 0:
        return signal
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    
    if std_val == 0:
        return signal - mean_val
        
    return (signal - mean_val) / std_val