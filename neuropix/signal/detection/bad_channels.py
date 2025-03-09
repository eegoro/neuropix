import numpy as np

def detect_bad_channels_by_mean(mean_threshold=500):
    def detect_func(data):
        bad_channels = np.abs(np.mean(data, axis=1)) > mean_threshold
        modified_data = data.copy()
        bad_channel_indices = np.where(bad_channels)[0]
        for bad_idx in bad_channel_indices:
            modified_data[bad_idx, :] = np.nan
        return modified_data
    return detect_func
   
def detect_bad_channels_by_std(std_threshold=3.0):
    def detect_func(data):
        channel_stds = np.std(data, axis=1)
        mean_std = np.mean(channel_stds)
        std_of_stds = np.std(channel_stds)
        bad_channels = np.abs(channel_stds - mean_std) > std_threshold * std_of_stds
        modified_data = data.copy()
        bad_channel_indices = np.where(bad_channels)[0]
        for bad_idx in bad_channel_indices:
            modified_data[bad_idx, :] = np.nan
        return modified_data
    return detect_func
