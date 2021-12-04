import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .kMeans import calculate_euclidean_distance


def calculate_cluster_distance(cluster_a, cluster_b):
    distances = []
    for point in cluster_a:
        for second_point in cluster_b:
            distances.append(calculate_euclidean_distance(np.array(point), np.array(second_point)))
    return min(distances)


class Agglomerative:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.data = None
        self.clusters = None

    def calculate_distances_between_clusters(self):
        distances = []
        clusters = []
        second_clusters = []

        for cluster in self.clusters:
            for second_cluster in self.clusters:
                if cluster == second_cluster:
                    continue
                else:
                    clusters.append(self.clusters.index(cluster))
                    second_clusters.append(self.clusters.index(second_cluster))
                    distances.append(calculate_cluster_distance(cluster, second_cluster))
        return pd.DataFrame({'cluster1': clusters,
                             'cluster2': second_clusters,
                             'distances': distances},
                            columns=['cluster1', 'cluster2', 'distances'])

    def plot_clusters(self):
        for cluster in self.clusters:
            cluster = np.array(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 1])
        plt.title("Clusters according to own agglomerative clustering method")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def find_clusters(self, data, k, printClusters=False):
        self.data = data

        # initialize each observation as a cluster
        self.clusters = [[data[i].tolist()] for i in range(data.shape[0])]

        while len(self.clusters) > k:
            # get distances between all clusters
            dists = self.calculate_distances_between_clusters()

            # merge closest clusters into one
            closest = dists[dists.distances == dists.distances.min()].values.tolist()[0]
            new_cluster = self.clusters[int(closest[0])] + self.clusters[int(closest[1])]
            if self.verbose:
                print("Replacing ", self.clusters[int(closest[0])], " and ", self.clusters[int(closest[1])], " with ",
                      new_cluster, " as the distance was ", closest[2])
            del self.clusters[int(closest[0])]
            del self.clusters[int(closest[1]-1)]
            self.clusters.append(new_cluster)

        if printClusters:
            print(self.clusters)
