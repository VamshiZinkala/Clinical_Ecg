import numpy as np

def segment_heartbeats(signal, r_peaks, window_size=200):
    half_window = window_size // 2
    beats = []
    valid_indices = []

    for i, peak in enumerate(r_peaks):
        start = peak - half_window
        end = peak + half_window
        
        # Salvage beats at the edges using Padding
        if start < 0:
            pad_len = abs(start)
            beat = np.pad(signal[0:end], (pad_len, 0), mode='edge')
        elif end > len(signal):
            pad_len = end - len(signal)
            beat = np.pad(signal[start:len(signal)], (0, pad_len), mode='edge')
        else:
            beat = signal[start:end]

        if len(beat) == window_size:
            # Normalize to 0 mean and 1 std dev
            beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
            beats.append(beat)
            valid_indices.append(i)

    return np.array(beats), valid_indices