import pandas as pd
import numpy as np
from loader import load_mitbih_record
from preprocessing import preprocess_signal
from rpeak_detection import detect_r_peaks
from segmentation import segment_heartbeats
from labels import process_labels

def export_sample(record_no, filename, beat_type="Normal"):
    print(f"--- Exporting {beat_type} sample from Record {record_no} ---")
    
    # 1. Load and process
    data = load_mitbih_record(record_no)
    signal = preprocess_signal(data['signal'], data['fs'])
    annotation = data['annotation']
    
    # 2. Segment
    r_peaks = detect_r_peaks(signal, data['fs'])
    beats, valid_indices = segment_heartbeats(signal, r_peaks, window_size=200)
    labels, label_mask = process_labels(annotation, r_peaks, valid_indices)
    
    # 3. Filter for the specific type
    target_label = 0 if beat_type == "Normal" else 1
    type_indices = np.where(labels == target_label)[0]
    
    if len(type_indices) > 0:
        # Take the first available beat of that type
        sample_beat = beats[type_indices[0]]
        pd.DataFrame(sample_beat).to_csv(filename, index=False, header=False)
        print(f"✅ Success! Saved to {filename}")
    else:
        print(f"❌ No {beat_type} beats found in Record {record_no}")

# Generate your two test files
if __name__ == "__main__":
    # Record 100 is famous for clean Normal beats
    export_sample(100, "patient_normal.csv", "Normal")
    
    # Record 200 contains many Ventricular premature contractions (Abnormal)
    export_sample(200, "patient_abnormal.csv", "Abnormal")