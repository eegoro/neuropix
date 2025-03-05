import matplotlib.pyplot as plt

def plot_tsne_results(features_df, depth, cluster_labels, estimated_depth, color_by='coord_x'):
    """
    Create two scatter plots of t-SNE results:
    1. Colored by specified continuous variable
    2. Colored by cluster assignments
    
    Parameters:
    -----------
    features_df : DataFrame
        Output from analyze_electrode_data
    color_by : str
        Column name to use for coloring points in the first plot
    clusters : array-like, optional
        Cluster assignments for each point
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 4))
    
    # First subplot - original coloring by specified variable
    scatter1 = ax1.scatter(
        features_df['tsne_1'],
        features_df['tsne_2'],
        c=features_df[color_by],
        cmap='viridis'
    )
    fig.colorbar(scatter1, ax=ax1, label=color_by)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title(f't-SNE Features (colored by {color_by})')
    
    clusters=features_df['coord_x']>depth
    clusters_inv=features_df['coord_x']>(max(features_df['coord_x'])-depth)
    # Second subplot - coloring by clusters

    scatter2 = ax2.scatter(
        features_df['tsne_1'],
        features_df['tsne_2'],
        c=clusters,
        cmap='tab20'  # Discrete colormap suitable for clusters
    )
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('t-SNE Features (colored by depth)')


    scatter3 = ax3.scatter(
            features_df['tsne_1'],
            features_df['tsne_2'],
            c=clusters_inv,
            cmap='tab20'  # Discrete colormap suitable for clusters
        )
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title('t-SNE Features (colored by max_coord - depth )')

    scatter4 = ax4.scatter(
            features_df['tsne_1'],
            features_df['tsne_2'],
            c=cluster_labels,
            cmap='tab20'  # Discrete colormap suitable for clusters
        )
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    ax4.set_title('t-SNE Features (colored by Kmeans clusters)')

    scatter5 = ax5.scatter(
            features_df['coord_x'],
            features_df['tsne_one_comp'],
            c=cluster_labels,
            cmap='cividis'  # Discrete colormap suitable for clusters
        )
    if estimated_depth!=None:
        ax5.axvline(x=estimated_depth, color='green', linestyle='--')

    ax5.set_xlabel('Y coordinate')
    ax5.set_ylabel('t-SNE one component')
    ax5.set_title('Kmeans clusters')
    
    plt.tight_layout()
    plt.show()

    return fig

def plot_depth_results(time_for_estimated_depth, estimated_depth_list):
    fig = plt.figure(figsize=(20, 4))
    plt.plot(time_for_estimated_depth, estimated_depth_list, color='black')
    plt.xlabel('Time [s]',fontsize=14)
    plt.ylabel('Estimated Depth',fontsize=14)
    plt.title(f'Depth', fontsize=14)
    plt.grid(True)
    plt.show()

    return fig