import numpy as np
import matplotlib.pyplot as plt

from .kMeans import calculate_euclidean_distance
from sklearn.cluster import KMeans


class Divisive:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.data = None
        self.clusters = []

    def find_clusters(self, data, k):
        self.data = data

        # define all points in one cluster
        self.clusters = [data]

        while len(self.clusters) < k:
            # choose cluster to split
            current = self.choose_cluster()
            # split cluster
            new_clusters = self.split_cluster(self.clusters[current])
            del self.clusters[current]
            self.clusters.append(new_clusters[0])
            self.clusters.append(new_clusters[1])

    def split_cluster(self, cluster):
        fitter = KMeans(n_clusters=2).fit(cluster)
        clusters = [[], []]
        for cluster_idx in range(0, 2):
            clusters[cluster_idx] = np.array([cluster[i].tolist() for i in range(cluster.shape[0])
                                              if fitter.labels_[i] == cluster_idx])
        return clusters

    def choose_cluster(self):
        square_errors = []
        for cluster in self.clusters:
            temp_errors = []
            mean = np.transpose(cluster).mean(axis=1)
            for point in cluster:
                temp_errors.append(np.square(calculate_euclidean_distance(point, mean)))
            square_errors.append(sum(temp_errors))
        return square_errors.index(max(square_errors))

    def plot_clusters(self):
        for cluster in self.clusters:
            cluster = np.array(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 1])
        plt.title("Clusters according to own divisive clustering method")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
