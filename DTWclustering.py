'''
more info on the theory behind DTW Clustering: https://rtavenar.github.io/blog/dtw.html
'''

import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from tslearn.metrics import dtw
import matplotlib.pyplot as plt

class DTWClustering:
    """
    A class for hierarchical clustering using Euclidean and DTW distances.

    Parameters:
        df (pd.DataFrame): Input dataframe containing time-series data.
    
    Methods:
        check_inputs(): Validates the input dataframe.
        euclidean_clustering(): Performs clustering using Euclidean distance.
        dtw_clustering(): Performs clustering using DTW distance (specify window size).
        plot_dendrogram(): Plots the dendrogram for clustering results.
        analyze_clusters(): Calculates silhouette scores and outputs analysis results.
        clustering(): Perform clustering using either KMeans or DBSCAN.
        plot_cluster_timeseries(): Plots the time series of each cluster.
    """
    def __init__(self, df):
        """
        Initialize the DTWClustering object.

        Args:
            df (pd.DataFrame): Input dataframe with time-series data.
        """
        self.df = df
        self.validate_dataframe()

    def validate_dataframe(self):
        """
        Validate the input dataframe to ensure no missing or infinite values.
        """
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        if self.df.isnull().any().any():
            raise ValueError("Data contains missing values.")
        if not np.isfinite(self.df.values).all():
            raise ValueError("Data contains infinite values.")
        if not np.issubdtype(self.df.dtypes.values[0], np.number):
            raise ValueError("Data contains non-numeric values.")


    def clustering(self, method: str, k: None, eps: None, min_samples: None, window: None, distance_metric: str):
        """
        Perform clustering using either KMeans or DBSCAN with Euclidean or DTW distances.

        Args:
            method (str): Clustering method ('kmeans' or 'dbscan').
            k (int): Number of clusters (only for KMeans).
            eps (float): Maximum distance between two samples to be considered as in the same neighborhood (only for DBSCAN).
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point (only for DBSCAN).
            window (int, optional): Window size for DTW. If specified, applies a Sakoe-Chiba constraint to limit warping to within the given radius. None means unrestricted DTW.
            distance_metric (str): The distance metric to use, either 'euclidean' or 'dtw'.

        Returns:
            dict: Clustering results, including cluster labels, silhouette score, and elapsed time.
        """
        start_time = time.time()

        # Choose the distance metric
        if distance_metric == "euclidean":
            pairwise_distances = pdist(self.df.values, metric="euclidean")
        elif distance_metric == "dtw":
            pairwise_distances = pdist(
                self.df.values,
                metric=lambda x, y: dtw(x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=window) if window else dtw(x, y)
            )
        else:
            raise ValueError("Unsupported distance metric. Use 'euclidean' or 'dtw'.")

        # Convert the pairwise distance from 1D to 2D matrix (this is the key fix)
        self.pairwise_distances_2d = squareform(pairwise_distances)

        # Choose clustering method
        if method == "kmeans":
            if k is None:
                raise ValueError("k must be specified for KMeans clustering.")
            # For KMeans, we don't need pairwise distance matrix, we need feature matrix
            model = KMeans(n_clusters=k)
            clusters = model.fit_predict(self.df.values)  # Pass original feature matrix for KMeans
             
             # Create a plot for each cluster
            plt.figure(figsize=(12, 8))
            
            # Plot each cluster
            for cluster in range(k):
                cluster_indices = [i for i, label in enumerate(clusters) if label == cluster]
                for idx in cluster_indices:
                    # Plot each time series in the cluster
                    plt.plot(self.df.index, self.df.iloc[:, idx], alpha=0.6, label=f"Cluster {cluster}" if idx == cluster_indices[0] else "")

            # Add titles and labels
            plt.title(f'KMeans Clustering with {k} clusters (Time Series Data)')
            plt.xlabel('Timestamp')
            plt.ylabel('Price')
            plt.legend(title="Cluster Labels", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.show()
            
            
        elif method == "dbscan":
            model = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
            clusters = model.fit_predict(self.pairwise_distances_2d)  # Use the pairwise distance matrix for DBSCAN

        else:
            raise ValueError("Unsupported clustering method. Use 'kmeans' or 'dbscan'.")

        # Calculate silhouette score if possible
        silhouette = -1  # Default for when silhouette is not possible
        if len(set(clusters)) > 1:  # More than one cluster
            silhouette = silhouette_score(self.df.values, clusters, metric="euclidean")

        elapsed_time = time.time() - start_time

        return {"clusters": clusters, "silhouette": silhouette, "time": elapsed_time}

    def plot_dendrogram(self, Z, title="Dendrogram"):
        """
        Plot a dendrogram from a linkage matrix.

        Args:
            Z (array): Linkage matrix.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(10, 6))
        dendrogram(Z)
        plt.title(title)
        plt.xlabel("Cluster Size")
        plt.ylabel("Distance")
        plt.show()

    def analyze_clusters(self, clustering_results, k, title_prefix=""):
        """
        Analyze clusters, calculate and display cluster-level statistics.

        Args:
            clustering_results (dict): Output from clustering method.
            k (int): Number of clusters.
            title_prefix (str): Prefix for titles in outputs.
        """
        clusters = clustering_results["clusters"]
        cluster_labels = np.unique(clusters)
        stats = []

        for label in cluster_labels:
            idx = np.where(clusters == label)
            cluster_data = self.df.iloc[idx]
            total_return = cluster_data.sum().sum()
            average = cluster_data.mean().mean()
            variance = cluster_data.var().mean()
            stats.append({"Cluster": label, "Total Return": total_return, "Average": average, "Variance": variance})

        stats_df = pd.DataFrame(stats)
        print(f"\n{title_prefix} Cluster Analysis")
        print(stats_df)
        
    def plot_cluster_timeseries(self, clustering_results):
        """
        Plot the time series of each cluster.

        Args:
            clustering_results (dict): The result of the clustering method.
        """
# Assuming 'clusters' and 'self.df' are available from your clustering results

        clusters = clustering_results["clusters"]
        unique_clusters = np.unique(clusters)

        # Loop over each cluster and create a plot for it
        for cluster in unique_clusters:
            # Find the indices of the time series belonging to this cluster
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_data = self.df.iloc[cluster_indices]

            # Create a new figure for each cluster
            plt.figure(figsize=(10, 6))
            for idx in cluster_indices:
                # Plot each time series in the cluster
                plt.plot(self.df.columns, self.df.iloc[idx], alpha=0.6, label=self.df.index[idx])
            
            # Customize the plot for each cluster
            plt.title(f"Cluster {cluster} - Time Series")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend(title="Tickers", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)  # Show legend with tickers
            plt.tight_layout()
            plt.show()  # Display the plot for this cluster
