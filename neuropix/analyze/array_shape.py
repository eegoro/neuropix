import numpy as np

def get_bins_array(data, fs, bin_size = 10, bin_overlap = 9):
    data_bins = np.lib.stride_tricks.sliding_window_view(data, round(bin_size*fs))[::round((bin_size-bin_overlap)*fs)] 
    return data_bins
