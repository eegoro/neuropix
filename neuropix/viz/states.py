import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

def plot_transition_frequency(x, y1, y2, intersection_point):
    fig = plt.figure(figsize=(5, 2))
    plt.plot(x, y1, label='Cluster 1', color='blue')
    plt.plot(x, y2, label='Cluster 2', color='red')
    plt.xlabel('Frequency', fontsize=10)
    plt.ylabel('Power', fontsize=10)
    plt.title('Average Normalized PSD for Clusters') 
    plt.legend(fontsize=10)
    plt.tick_params(axis='x', which='major', labelsize=8)
    plt.tick_params(axis='y', which='major', labelsize=8)
    plt.grid(True)

    plt.xlim(0, 35)

    for (x_inter,y_iter) in intersection_point:
        plt.axvline(x=x_inter, color='green', linestyle='--')
        plt.text(x_inter + 0.5, y_iter + 0.1, f'{x_inter:.2f}', fontsize=8) 
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    return fig

def plot_delta_beta_matrix(data, num_bins_x = 6, clim_low = 0.1, clim_high=0.99):
    num_y = len(data) // num_bins_x
    delta_beta_matrix = data[:num_bins_x*num_y].reshape((num_y, num_bins_x))

    fig = plt.figure(figsize=(5, 2))
    plt.imshow(delta_beta_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.clim(np.quantile(delta_beta_matrix, clim_low),np.quantile(delta_beta_matrix, clim_high))
    plt.colorbar(label='Delta/Beta Power Ratio')
    plt.title('Delta/Beta Power Ratio for 30-Minute Intervals', fontsize=14)
    # plt.xlabel('Time [Minutes]', fontsize=16)
    # plt.ylabel('Half-Hour count', fontsize=16)
    # plt.xticks(tick_locations, tick_labels, fontsize=14)
    # plt.yticks(fontsize=14)
    plt.grid(False)

    return fig


def plot_colored_db_ratio_by_cluster(data, labels, label_names=None):
    points = np.array([range(len(data)), data]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = ['#2ecc71', '#e74c3c', '#3498db']
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(12, 6))
    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(labels[:-1]) 
    ax.add_collection(lc)
    ax.set_xlim(0, len(data) - 1)
    ax.set_ylim(np.min(data) - 0.1, np.max(data) + 0.1)
    if label_names is None:
        label_names = [f'Class {i}' for i in range(len(colors))]
    
    legend_elements = [plt.Line2D([0], [0], color=color, label=label)
                      for color, label in zip(colors, label_names)]
    ax.legend(handles=legend_elements)
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Delta Beta Ratio')
    
    return fig, ax