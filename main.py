"""
main.py - Training Pipeline for Hybrid ECG Classification System
=================================================================
This script downloads standard MIT-BIH records, extracts digital
heartbeat signals, and trains the Hybrid 1D CNN + Random Forest model.
"""

import numpy as np
import os
import sys

# Import the existing modules you copied over
from loader import load_mitbih_record
from preprocessing import preprocess_signal
from rpeak_detection import detect_r_peaks
from segmentation import segment_heartbeats
from labels import process_labels
from model import train_hybrid_system

def prepare_training_data(record_numbers=[100, 105, 119, 200, 201, 213]):
    """
    Downloads and processes a mix of Normal and Abnormal MIT-BIH records.
    Extracts 200-sample heartbeat windows for training.
    """
    print("\n" + "="*50)
    print(" STEP 1: DOWNLOADING & PROCESSING MIT-BIH DATA")
    print("="*50)
    
    all_beats = []
    all_labels = []

    for rec in record_numbers:
        print(f"-> Processing Record {rec}...")
        try:
            # 1. Load data via wfdb
            data = load_mitbih_record(rec)
            signal = data['signal']
            fs = data['fs']
            annotation = data['annotation']

            # 2. Preprocess (Bandpass filter)
            filtered_signal = preprocess_signal(signal, fs)

            # 3. Find R-peaks and Segment into 200-sample beats
            r_peaks = detect_r_peaks(filtered_signal, fs)
            beats, valid_indices = segment_heartbeats(filtered_signal, r_peaks, window_size=200)

            # 4. Extract labels (0 = Normal, 1 = Abnormal)
            labels_arr, label_mask = process_labels(annotation, r_peaks, valid_indices)

            # Keep only the beats that have valid labels
            valid_beats = beats[label_mask]
            valid_labels = labels_arr[label_mask]

            all_beats.append(valid_beats)
            all_labels.append(valid_labels)
            
            print(f"   [OK] Extracted {len(valid_beats)} valid beats.")
            
        except Exception as e:
            print(f"   [ERROR] Skipping record {rec}: {e}")

    # Combine everything into giant arrays for ML training
    if not all_beats:
        print("Failed to extract any data. Check internet connection for MIT-BIH download.")
        sys.exit(1)

    X = np.vstack(all_beats)
    y = np.concatenate(all_labels)
    
    print("\n" + "="*50)
    print(" DATASET READY")
    print(f" Total Heartbeats: {len(X)}")
    print(f" Normal Beats: {np.sum(y == 0)}")
    print(f" Abnormal Beats: {np.sum(y == 1)}")
    print("="*50 + "\n")
    
    return X, y

if __name__ == "__main__":
    # Check if models already exist
    if os.path.exists("cnn_extractor.h5") and os.path.exists("ml_pipeline.joblib"):
        print("\n✅ Models are already trained and saved in this directory!")
        print("To start the dashboard, run the following command in your terminal:")
        print("\n    streamlit run app.py\n")
    else:
        # Step 1: Get the data
        X_data, y_data = prepare_training_data()
        
        # Step 2: Train the Hybrid Model (1D CNN + Random Forest)
        # This function is imported from your new model.py
        train_hybrid_system(X_data, y_data)
        
        print("\n🎉 Setup Complete! Models have been successfully generated.")
        print("To launch your Clinical Dashboard, type the following into your terminal:")
        print("\n    streamlit run app.py\n")