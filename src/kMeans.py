import numpy as np
import matplotlib.pyplot as plt


class KMeans:

    def __init__(self, max_iterations=50, verbose=False):
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.centroids = []
        self.k = 2
        self.data = None
        self.clusters = None

    def plot(self, include_centroids=False):
        plt.scatter(self.data[:, 0], self.data[:, 1])
        if include_centroids:
            for point in self.centroids:
                plt.scatter(*point, marker='x', color='black')
        plt.show()

    def create_random_centroids(self):
        centroid_min_coordinate = self.data.min().min()
        centroid_max_coordinate = self.data.max().max()
        for centroid in range(self.k):
            centroid = np.random.uniform(centroid_min_coordinate, centroid_max_coordinate, self.data.shape[1])
            self.centroids.append(centroid)

    def calculate_euclidean_distance(self):
        pass

    def find_clusters(self, k, data):
        self.k = k
        self.data = data
        self.clusters = [[] for _ in range(self.k)]

        # initialize random centroids
        self.create_random_centroids()
        self.plot(include_centroids=True)
