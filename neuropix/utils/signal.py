import numpy as np

def extract_time_segment(signal, 
                time_array, 
                start_time, 
                end_time):
    
    start_idx = np.searchsorted(time_array, start_time, side='left')
    start_idx = min(start_idx, len(time_array) - 1)

    end_idx = np.searchsorted(time_array, end_time, side='right')
    end_idx = max(end_idx, 0)

    if len(signal.shape) == 1:
        segment = signal[start_idx:end_idx]
    else:
        segment = signal[:, start_idx:end_idx]

    time_segment = time_array[start_idx:end_idx]

    return segment, time_segment