import matplotlib.pyplot as plt
import numpy as np

def plot_csd_result(data_filtered, data_smothed, data_csd, data_time, unique_y, data_x_line = [], data_y_line = []):
    fig = plt.figure(figsize=(16, 4))

    for plot_idx, data_for_plot in enumerate([data_filtered, data_smothed, data_csd]):
        # ax = (ax1, ax2, ax3)[plot_idx+1]
        plt.subplot(1, 3, plot_idx+1)
        max_a = np.max(np.abs(data_for_plot))
        levels = np.linspace(-max_a, max_a, 200)
        plt.contourf(data_time, unique_y, data_for_plot, levels=levels, cmap=['viridis', 'viridis', plt.cm.bwr ][plot_idx]) #mpl.cm.jet (import matplotlib as mpl)
        for x in data_x_line:
            plt.axvline(x=x, color='red', linestyle='--')
        for y in data_y_line:
            plt.axhline(y=y, color='red', linestyle='--')
        plt.colorbar(label=[ 'Voltage (uV)', 'Voltage (uV)', 'CSD (arbitrary units)'][plot_idx])
        plt.title(['Filtered', 'Smothed', 'Current Source Density (CSD)'][plot_idx])
        plt.xlabel('Time [ms]')
        plt.ylabel('Y coordinate')
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 

    return fig 