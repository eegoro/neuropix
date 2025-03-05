import numpy as np

def calculate_transition_frequency(freq, power_1, power_2):
    diff = power_1 - power_2
    idx = np.where(np.diff(np.sign(diff)))[0]
    x_intersections = freq[idx] - diff[idx] * (freq[idx + 1] - freq[idx]) / (diff[idx + 1] - diff[idx])
    y_intersections = np.interp(x_intersections, freq, power_1)

    return np.stack([x_intersections,y_intersections], axis=1)

def calculate_delta_beta_ratio(power_spectrum, delta_window=(0, 4), beta_window=(15,30)):
    power_spectrum_delta = power_spectrum[:, delta_window[0]:delta_window[1]]
    power_spectrum_beta = power_spectrum[:, beta_window[0]:beta_window[1]]
    delta_mean_power = np.mean(power_spectrum_delta, axis=1)
    beta_mean_power = np.mean(power_spectrum_beta, axis=1)
    delta_beta_ratio = delta_mean_power / beta_mean_power

    return delta_beta_ratio