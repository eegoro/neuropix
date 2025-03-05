import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import scipy.cluster.hierarchy as sch
import numpy as np

def plot_sorted_correlation_matrix(correlation_matrix, linkage_matrix):
        fig = plt.figure(figsize=(5, 2), dpi=150) 
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8]) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        sch.set_link_color_palette(['orange', 'lightblue'])
        with plt.rc_context({'lines.linewidth': 0.7}):
            dendrogram = sch.dendrogram(linkage_matrix, ax=ax0, no_labels=True, orientation='left', above_threshold_color='grey')

        ax0.tick_params(axis='x', which='major', labelsize=6)
        ax0.grid(False)
        ax0.set_xlabel('Distance', fontsize=10)
        ax0.spines['left'].set_edgecolor("white")

        sorted_indices = list(map(int, dendrogram['ivl'][::-1])) 
        sorted_correlation_matrix = correlation_matrix[sorted_indices][:, sorted_indices]

        sns.heatmap(sorted_correlation_matrix, cmap='viridis',  cbar_kws={"shrink": 0.5, 'label': 'correlation'}, vmin=-1, vmax=1, ax=ax1)
    
        ax1.tick_params(axis='x', which='major', labelsize=8)
        ax1.tick_params(axis='y', which='major', labelsize=8)

        plt.title('Sorted Correlation Matrix')
        ax1.set_xlabel('Sample Index', fontsize=10)

        plt.grid(False)

        return fig, ax0, ax1

def plot_autocorr(autocorr, avg_peak_index, peak_range = 300, bin_second = 10):
        start_index = max(avg_peak_index - (peak_range // bin_second), 0) 
        end_index = min(avg_peak_index + (peak_range // bin_second), len(autocorr))

        autocorr_plot = autocorr[start_index:end_index + 1]
        avg_autocorr_range = np.mean(autocorr_plot)
        lags = np.arange(-len(autocorr_plot)//2 + 1, len(autocorr_plot)//2 + 1)* bin_second 

        fig = plt.figure(figsize=(5, 2))
        plt.plot(lags, autocorr_plot, color='red', label='Autocorrelation')
        plt.axhline(y=avg_autocorr_range, color='black', linestyle='--', label='Average Autocorrelation')
        plt.xlabel('Lag [s]',fontsize=14)
        plt.ylabel('Autocorr.',fontsize=14)
        plt.title(f'Average Autocorrelation Function within {peak_range} seconds around the peak', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True)

        return fig

def plot_heat_map(cross_corr, lags):
        fig = plt.figure(figsize=(5, 2))
        
        plt.imshow(cross_corr, aspect='auto', cmap='viridis', interpolation='nearest', extent=[lags[0], lags[-1], cross_corr.shape[0], 0], vmax=1)
        plt.colorbar(label='Autocorr.')
        plt.ylabel('min')
        plt.xlabel('Lag[s]')

        return fig