import numpy as np
import matplotlib.pyplot as plt


class KMeans:

    def __init__(self, max_iterations=50, verbose=False):
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.centroids = []
        self.k = None
        self.data = None
        self.clusters = None

    def plot(self, include_centroids=False):
        plt.scatter(self.data[:, 0], self.data[:, 1])
        if include_centroids:
            for point in self.centroids:
                plt.scatter(*point, marker='x', color='black')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def create_random_centroids(self):
        centroid_min_coordinate = self.data.min().min()  # find point with smallest coordinates
        centroid_max_coordinate = self.data.max().max()  # find point with biggest coordinates
        for centroid in range(self.k):  # repeat k-times
            # create random coordinates with previously found smallest and biggest as borders
            centroid = np.random.uniform(centroid_min_coordinate, centroid_max_coordinate, self.data.shape[1])
            self.centroids.append(centroid)  # add created centroid to the list

    def calculate_euclidean_distance(self):
        pass

    def find_clusters(self, k, data):
        self.k = k
        self.data = data
        self.clusters = [[] for _ in range(self.k)]

        # initialize random centroids
        self.create_random_centroids()
        self.plot(include_centroids=False)
