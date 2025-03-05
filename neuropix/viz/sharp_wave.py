import matplotlib.pyplot as plt
import numpy as np

def plot_sharp_wave(data, fs, peaks_tr, num_cols = 10):
    window = int(0.5 * fs)

    # Calculate number of rows and columns for subplots
    num_peaks = len(peaks_tr)  # Limiting to 11 plots (0-10)
    num_cols = num_cols  # You can adjust this number to have more plots in a row
    num_rows = (num_peaks + num_cols - 1) // num_cols

    # Create figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 2*num_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # Plot each peak in its own subplot
    for idx, peak in enumerate(peaks_tr):
            
        part = data[max(0,peak-window):peak+window]
        time_axis = (np.arange(len(part))/fs)*1000
        
        axes[idx].plot(time_axis, part)
        axes[idx].axvline(x=time_axis[len(part) - window], color='green', linestyle='--')
        axes[idx].set_title(f'Peak {idx+1}')
        axes[idx].set_xlabel('Time [ms]')
        axes[idx].set_ylabel('Amplitude')

    # Remove empty subplots if any
    for idx in range(num_peaks, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig, axes

def plot_appearance(peaks_time, fs):
    fig, ax = plt.subplots(1,1,figsize=(12, 1))
    ax.scatter(peaks_time, [0]*len(peaks_time))
    ax.set_title('Sharp Wave appearance')
    ax.set_xlabel('Time [s]')

    return fig, ax