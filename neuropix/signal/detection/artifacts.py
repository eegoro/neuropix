import numpy as np

def artefacts_by_max_in_window(window_size, overlap, threshold = 300):
    def detect_func(data):
        return np.apply_along_axis(_artefacts_by_max_in_window, 1, data, window_size, overlap, threshold)
    return detect_func


def _artefacts_by_max_in_window(data, window_size, overlap, threshold = 300):
    step = window_size - overlap
    processed_signal = data.copy()
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i+window_size]
        if np.max(np.abs(window)) > threshold:
            processed_signal[i:i+window_size] = np.nan
    return processed_signal


