from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import OrderedDict
import numpy as np

def cluster_correlation_matrix(correlation_matrix, psd, num_clusters=20):
    linkage_matrix = linkage(correlation_matrix, method='ward') # hierarchical clustering
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')  # Cut the dendrogram to get clusters
    clusters = {}
    for i, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(i)

    avg_psd_cluster = []
    order_clusters = OrderedDict(sorted(clusters.items(), reverse=True, key = lambda x : len(x[1])))
    for indexes in order_clusters.values():
        avg_psd_cluster.append(np.mean(psd[indexes], axis=0))

    return linkage_matrix, cluster_labels, clusters, avg_psd_cluster