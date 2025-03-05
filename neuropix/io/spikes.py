import numpy as np
import os 
import pandas as pd
# path_folder = "/media/sil1/Data/Large_scale_mapping_NP/Awake_exp/SA6/SA6_experiment_4_9_23/Insertion4/catgt_SA6_experiment_4_9_23_4_g0"

def get_spike_arrays(path_folder):
    amplitudes = np.load(os.path.join(path_folder, 'amplitudes.npy'))
    channel_positions = np.load(os.path.join(path_folder, 'channel_positions.npy'))
    spike_clusters = np.load(os.path.join(path_folder, 'spike_clusters.npy'))
    spike_times = np.load(os.path.join(path_folder, 'spike_times.npy'))
    cluster_groups = pd.read_csv(os.path.join(path_folder, 'cluster_group.tsv'), sep = '\t')

    return {'amplitudes': amplitudes, 
            "channel_positions": channel_positions, 
            "spike_clusters": spike_clusters, 
            "spike_times": spike_times, 
            "cluster_groups": cluster_groups}
