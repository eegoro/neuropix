from scipy import signal
import numpy as np

def compute_power_spectrum(data, fs, nperseg = None, noverlap=None):
    frequencies, Pxx = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap)

    return frequencies, Pxx

def normalize_power_spectrum(data):
    avg_psd = np.mean(data, axis=0)
    std_per_f = np.std(data, axis=0)
    return (data - avg_psd) / std_per_f

