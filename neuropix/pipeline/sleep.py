from .. import analyze
from .. import viz
import numpy as np


def apply_sleep_analysis(data, fs, 
                       bin_size = 10, 
                       bin_overlap = 9, 
                       nperseg=None, 
                       noverlap=None, 
                       low_freq_bound = 30,
                       num_clusters = 20,
                       delta_window=(0, 4), 
                       beta_window=(15,30),
                       confidence_level = 0.95, 
                       smooth = 500,
                       num_bins_x = 6,
                       peak_range = 300
                       ):
    
    data_bins = analyze.get_bins_array(data=data, fs=fs, bin_size = bin_size, bin_overlap = bin_overlap)
    if nperseg==None:
        nperseg=fs
    freq, PSD = analyze.compute_power_spectrum(data_bins, fs=fs,nperseg=nperseg, noverlap=noverlap)
    nPSD_low = analyze.normalize_power_spectrum(PSD[:,:low_freq_bound])
    correlation_matrix = analyze.corr_matrix(nPSD_low)
    linkage_matrix, cluster_labels, clusters, avg_psd_cluster = analyze.cluster_correlation_matrix(correlation_matrix, nPSD_low, num_clusters=num_clusters)
    transition_frequencies = analyze.calculate_transition_frequency(freq[:low_freq_bound], avg_psd_cluster[0], avg_psd_cluster[1])
    delta_beta_ratio = analyze.calculate_delta_beta_ratio(PSD, delta_window=delta_window, beta_window=beta_window)
    norm_autocorr = analyze.cross_corr(delta_beta_ratio, delta_beta_ratio)
    dict_sleep_cycles = analyze.analyze_sleep_cycles(norm_autocorr, bin_second = bin_size, confidence_level = confidence_level, smooth = smooth)

    fig_corr_matrix, _, _ = viz.plot_sorted_correlation_matrix(correlation_matrix, linkage_matrix)
    fig_freq = viz.plot_transition_frequency(freq[:low_freq_bound], avg_psd_cluster[0], avg_psd_cluster[1], transition_frequencies)
    fig_db_matrix = viz.plot_delta_beta_matrix(delta_beta_ratio, num_bins_x = num_bins_x)
    fig_autocorr = viz.plot_autocorr(norm_autocorr, np.argmax(norm_autocorr), peak_range = peak_range, bin_second = bin_size)
    fig_kde = viz.plot_kde_distribution(dict_sleep_cycles)

    return {
                'data_bins': data_bins,
                'freq': freq,
                'PSD': PSD,
                'nPSD_low': nPSD_low,
                'correlation_matrix': correlation_matrix,
                'linkage_matrix': linkage_matrix,
                'cluster_labels': cluster_labels,
                'clusters': clusters,
                'avg_psd_cluster': avg_psd_cluster,
                'transition_frequencies': transition_frequencies,
                'delta_beta_ratio': delta_beta_ratio,
                'norm_autocorr': norm_autocorr,
                'dict_sleep_cycles': dict_sleep_cycles,
                'fig_corr_matrix': fig_corr_matrix,
                'fig_freq': fig_freq,
                'fig_db_matrix': fig_db_matrix,
                'fig_autocorr': fig_autocorr,
                'fig_kde': fig_kde
    }

