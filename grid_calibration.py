"""
grid_calibration.py - Detects the physical spacing of the ECG paper grid
"""
import numpy as np
import cv2

def calibrate_grid(gray):
    clahe = cv2.createCLAHE(2.0, (8,8))
    enhanced = clahe.apply(gray)

    vertical_profile = np.mean(enhanced, axis=0)
    horizontal_profile = np.mean(enhanced, axis=1)

    v_fft = np.abs(np.fft.rfft(vertical_profile - np.mean(vertical_profile)))
    h_fft = np.abs(np.fft.rfft(horizontal_profile - np.mean(horizontal_profile)))

    # Ignore the zero-frequency (DC) component
    v_fft[0] = 0
    h_fft[0] = 0

    v_peaks = np.argsort(v_fft)[-3:]  # take top 3 peaks
    h_peaks = np.argsort(h_fft)[-3:]

    v_spacing = np.mean(len(vertical_profile) / v_peaks)
    h_spacing = np.mean(len(horizontal_profile) / h_peaks)

    if v_spacing <= 0 or h_spacing <= 0:
        return None, None

    return v_spacing, h_spacing