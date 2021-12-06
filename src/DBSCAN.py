import numpy as np
from matplotlib import pyplot as plt

from .kMeans import calculate_euclidean_distance


class DBSCAN:

    def __init__(self, verbose=False):
        self.data = None
        self.epsilon = 1
        self.minPts = 3
        self.labels = None
        self.verbose = verbose
        self.clusters = None

    def find_neighbours(self, point):
        neighbours = []
        if self.verbose:
            print("Looking for neighbours of point ", point)
        idx = 0
        for possible_neighbour in self.data:
            if np.array_equiv(point, possible_neighbour):
                idx += 1
                continue
            distance = calculate_euclidean_distance(point, possible_neighbour)
            if distance <= self.epsilon:
                neighbours.append(idx)
            idx += 1

        if self.verbose:
            print("Found neighbours: ", neighbours)

        return neighbours

    def grow_cluster(self, point_idx, label, neighbours):
        self.labels[point_idx] = label

        i = 0
        while i < len(neighbours):
            if self.labels[neighbours[i]] == 0 or self.labels[neighbours[i]] == -1:
                self.labels[neighbours[i]] = label

                current_neighbours = self.find_neighbours(self.data[neighbours[i]])
                if len(current_neighbours) >= self.minPts:
                    neighbours += current_neighbours

            i += 1

        if self.verbose:
            print("Cluster ", label, " created.")

    def find_clusters(self, data, epsilon, minPts):
        self.data = data
        self.epsilon = epsilon
        self.minPts = minPts
        self.labels = [0] * len(self.data)

        current_label = 0
        if self.verbose:
            print("Starting")
        for point in range(len(self.data)):
            if not (self.labels[point] == 0):
                continue

            current_neighbours = self.find_neighbours(self.data[point])
            if len(current_neighbours) < self.minPts:
                self.labels[point] = -1
            else:
                current_label += 1
                if self.verbose:
                    print("Creating cluster ", current_label, "...")
                self.grow_cluster(point, current_label, current_neighbours)

        if self.verbose:
            print("Labeled! Labels: ", self.labels)
        # parse into clusters
        self.clusters = [[] for _ in range(current_label)]
        for label in range(0, current_label):
            self.clusters[label] = [self.data[i] for i in range(len(self.data)) if self.labels[i] == label+1]
        if self.verbose:
            print("Finished! Clusters: ", self.clusters)

    def plot_clusters(self):
        for cluster in self.clusters:
            cluster = np.array(cluster)
            print(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 1])
        plt.title("Clusters according to own DBSCAN")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
