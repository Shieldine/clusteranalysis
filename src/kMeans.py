import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class KMeans:

    def __init__(self, max_iterations=50, verbose=False):
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.centroids = []
        self.k = None
        self.data = None
        self.clusters = None
        self.distances = None

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

    def calculate_euclidean_distance(self, point_a, point_b):
        return np.sqrt(np.sum(np.square(point_a - point_b)))

    def assign_points_to_centroids(self):
        distances = []  # for distances between the points and the centroids
        assigned = []  # for the assigned centroids
        points = self.data.shape[0]  # number of points (n)

        # for each point
        for point in range(points):
            point_distances = np.array([])
            # for each centroid
            for centroid in range(self.k):
                # calculate distances from the current point to all centroids
                current_distance = self.calculate_euclidean_distance(
                    self.centroids[centroid], self.data[point])
                point_distances = np.append(point_distances, current_distance)

            # choose closest centroid
            closest = np.where(point_distances == np.amin(point_distances))[0].tolist()[0]
            distance_to_centroid = np.amin(point_distances)

            # append to lists
            distances.append(distance_to_centroid)
            assigned.append(closest)

        return assigned, distances

    def plot_with_clusters(self, include_centroids=False):
        for cluster in range(self.k):
            cluster_data = np.array([self.data[i].tolist() for i in range(self.data.shape[0]) if self.distances[0][i] == cluster])
            print(cluster_data)
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1])
        if include_centroids:
            for point in self.centroids:
                plt.scatter(*point, marker='x', color='black')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def find_clusters(self, k, data):
        self.k = k
        self.data = data
        self.clusters = [[] for _ in range(self.k)]
        print(self.data)

        # initialize random centroids
        self.create_random_centroids()
        # self.plot(include_centroids=True)

        # assign points to centroids
        self.distances = self.assign_points_to_centroids()
        if self.verbose:
            print(self.distances)

        self.plot_with_clusters(include_centroids=True)

