from scipy import signal
from scipy.ndimage import gaussian_filter

def notch_filter(freq=50.0, fs=30000.0, q=30.0, axis=1):
    w0 = freq / (fs / 2) 
    b, a = signal.iirnotch(w0, q)
    def filter_func(data):
        return signal.filtfilt(b, a, data, axis=axis)
    return filter_func

def bandpass_filter(lowcut=0.5, highcut=250.0, fs=30000.0, order = 2, axis=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        high = 0.95
    b, a = signal.butter(order, [low, high], btype='band')
    def filter_func(data):
        return signal.filtfilt(b, a, data, axis=axis)
    return filter_func

def lowpass_filter(lowcut=0.5, fs=30000.0, order = 2, axis=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    def filter_func(data):
        return signal.filtfilt(b, a, data, axis=axis)
    return filter_func

def highpass_filter(highcut=500, fs=30000.0, order = 2, axis=1):
    nyq = 0.5 * fs
    low = highcut / nyq
    b, a = signal.butter(order, low, btype='high')
    def filter_func(data):
        return signal.filtfilt(b, a, data, axis=axis)
    return filter_func

def gauss_filter(sigma = [20, 10]):
    def filter_func(data):
        return gaussian_filter(data, sigma = sigma)
    return filter_func



# def notch_filter(data, freq=50.0, fs=30000.0, q=30.0):
#         w0 = freq / (fs/2)
#         b, a = signal.iirnotch(w0, q)
#         filtered_data = signal.filtfilt(b, a, data)
            
#         return filtered_data

# def bandpass_filter(data, lowcut=0.5, highcut=250.0, fs=30000.0, order = 2):
#     filtered_data = np.zeros(shape = (data.shape[0], data.shape[1]))
#     # filtered_data = data.copy()

#     for idx, raw in enumerate(data):
#         nyq = 0.5 * fs
#         low = lowcut / nyq
#         high = highcut / nyq
#         if high >= 1.0:
#             high = 0.95 
        
#         b, a = signal.butter(order, [low, high], btype='band')
#         filtered_data[idx] = signal.filtfilt(b, a, raw)
    
#     return filtered_data

# def lowpass_filter(data, lowcut=0.5, fs=30000.0, order = 2):
#     filtered_data = np.zeros(shape = (data.shape[0], data.shape[1]))
#     # filtered_data = data.copy()

#     for idx, raw in enumerate(data):
#         nyq = 0.5 * fs
#         low = lowcut / nyq
        
#         b, a = signal.butter(order, low, btype='low')
#         filtered_data[idx] = signal.filtfilt(b, a, raw)
    
#     return filtered_data

# def highpass_filter(data, highcut=500, fs=30000.0, order = 2):
#     filtered_data = np.zeros(shape = (data.shape[0], data.shape[1]))
#     # filtered_data = data.copy()

#     for idx, raw in enumerate(data):
#         nyq = 0.5 * fs
#         low = highcut / nyq
        
#         b, a = signal.butter(order, low, btype='high')
#         filtered_data[idx] = signal.filtfilt(b, a, raw)
    
#     return filtered_data