import cv2
import numpy as np

import matplotlib.pyplot as plt


def reject_if_blurry(gray, thresh=80):
    return cv2.Laplacian(gray, cv2.CV_64F).var() < thresh

def reject_if_low_contrast(gray, thresh=25):  # was 30
    return gray.std() < thresh

def reject_if_noisy(binary, thresh=0.65):
    # Remove grid-like structures using morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    noise_ratio = cleaned.sum() / cleaned.size
    return noise_ratio > 1.0  # Temporarily disabled to allow tightly cropped small images

def check_image_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    if reject_if_blurry(gray):
        return False, "Rejected: blurry image"
    if reject_if_low_contrast(gray):
        return False, "Rejected: low contrast"
    if reject_if_noisy(binary):
        return False, "Rejected: noisy or obstructed"

    return True, "Accepted"
