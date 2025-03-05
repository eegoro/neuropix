import numpy as np
from numpy.lib.stride_tricks import as_strided
import signal

def corr_pearson_vectorized(data, template):
    data_mean = np.nanmean(data, axis=1, keepdims=True)
    data_std = np.nanstd(data, axis=1, keepdims=True)
    template_mean = np.nanmean(template)
    template_std = np.nanstd(template)
    
    numerator = np.nanmean((data - data_mean) * (template - template_mean), axis=1)
    return numerator / (data_std.ravel() * template_std)

def cross_corr(data, template):
    half_size_temp = len(template) // 2
    padded_data = np.pad(data, (half_size_temp, half_size_temp), 
                        mode='constant', constant_values=np.nan)
    window_shape = len(template)
    # step = 1
    n_windows = len(padded_data) - window_shape + 1
    windows = as_strided(padded_data,
                        shape=(n_windows, window_shape),
                        strides=(padded_data.strides[0], padded_data.strides[0]))
    return corr_pearson_vectorized(windows, template)

def normalized_cross_correlation(signal_data, window, mode='same'):

    correlation = signal.correlate(signal_data, window, mode=mode)
    signal_energy = np.sqrt(signal.correlate(signal_data**2, np.ones(len(window)), mode=mode))
    window_energy = np.sqrt(np.sum(window**2))
    
    normalized_correlation = correlation / (signal_energy * window_energy)
    
    return normalized_correlation

def corr_matrix(data):
    return np.corrcoef(data)