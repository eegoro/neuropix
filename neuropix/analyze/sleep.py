from scipy import signal, stats
import numpy as np

def analyze_sleep_cycles(norm_autocorr, bin_second = 10, confidence_level = 0.95, smooth = 500):

    peaks, _ = signal.find_peaks(norm_autocorr)
    peak_distances = np.diff(peaks)

    avg_dist_p2p = np.mean(peak_distances) * bin_second
    
    variance = np.var(peak_distances) * bin_second
    std_dev = np.std(peak_distances) * bin_second

    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * std_dev / np.sqrt(len(peak_distances))
    lower_bound = avg_dist_p2p - margin_of_error
    upper_bound = avg_dist_p2p + margin_of_error
    z_scores = stats.zscore(peak_distances)

    time_values = peak_distances * bin_second 
    kde = stats.gaussian_kde(time_values)

    x_values = np.linspace(time_values.min(), time_values.max(), smooth)
    y_values = kde(x_values)

    return {'average distance between cycles': avg_dist_p2p,
            'standard deviation': std_dev,
            '95 percent confidence interval': (lower_bound, upper_bound),
            'maximal cycle time (sec)': np.max(time_values),
            'minimal cycle time (sec)': np.min(time_values),
            'kde': (x_values, y_values)
            }
    