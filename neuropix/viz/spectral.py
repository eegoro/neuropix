import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_freq_power_matrix(data, freq):
    fig = plt.figure(figsize=(5, 2))
    plt.imshow(np.log(data), origin='lower', aspect='auto', cmap='viridis', extent=[freq[0],freq[-1],0,data.shape[0]])
    plt.colorbar(label='Log')
    plt.xlabel('Freq')
    # plt.title(f'Animal: {Chosen_Animal}. Insertion: {Chosen_Insertion}') 
    return fig
