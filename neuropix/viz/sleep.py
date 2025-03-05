import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_kde_distribution(dict_sleep_cycles):
        x_values, y_values = dict_sleep_cycles['kde']
        avg_dist_p2p = dict_sleep_cycles['average distance between cycles']
        lower_bound, upper_bound = dict_sleep_cycles['95 percent confidence interval']
        std_dev = dict_sleep_cycles['standard deviation']
        
        # Plot the KDE Distribution
        fig = plt.figure(figsize=(5, 2))
        plt.plot(x_values, y_values, color='skyblue', label='Sleep Time Distribution (KDE)')

        # Add Average and Confidence Interval
        plt.axvline(avg_dist_p2p, color='red', linestyle='dashed',
                    linewidth=1, label=f'Average ({avg_dist_p2p:.2f} seconds)')
        plt.axvspan(lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')

        # Plot Labels and Title
        plt.xlabel('Cycle Length [seconds]', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Distribution of Sleep Cycle Times', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.5)  # Optional: Add gridlines for clarity
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)  # Larger tick labels for x-axis
        plt.yticks(fontsize=12)

        # Add Standard Deviation
        plt.text(0.75, 0.8, f'Std. Dev. = {std_dev:.2f} seconds', transform=plt.gca().transAxes, fontsize=10)

        return fig