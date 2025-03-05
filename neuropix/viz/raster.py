import matplotlib.pyplot as plt

def plot_raster(spike_times, spike_clusters, sampling_rate, start_trig, end_trig):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(spike_times/sampling_rate,
         spike_clusters,
         '.', color='gray', markersize = 0.5)

    for start, end in zip(start_trig, end_trig):
        ax.plot([start, end], [0, 0], 'r-', linewidth=3)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Unit ID')
    ax.set_title('Spike raster for all units')
    
    return fig, ax

def plot_one_unit(unitID, spike_times, spike_clusters, amplitudes, sampling_rate):
    fig, ax = plt.subplots(figsize=(10,2))
    ax.plot(spike_times[spike_clusters == unitID]/sampling_rate,
            amplitudes[spike_clusters == unitID],
            '.', color='orange',
            markersize=1,
            alpha=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Spike amplitudes for one unit')

    return fig, ax