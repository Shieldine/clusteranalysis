import numpy as np
import matplotlib.pyplot as plt


def calculate_euclidean_distance(point_a, point_b):
    return np.sqrt(np.sum(np.square(point_a - point_b)))


class KMeans:

    def __init__(self, max_iterations=50, verbose=False):
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.centroids = []
        self.k = None
        self.data = None
        self.clusters = None
        self.distances = None
        self.sum_of_distances = 0

    def plot(self, include_centroids=False):
        plt.scatter(self.data[:, 0], self.data[:, 1])
        if include_centroids:
            for point in self.centroids:
                plt.scatter(*point, marker='x', color='black')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def create_random_centroids(self):
        # randomly choose k points in dataset
        idx = np.random.choice(len(self.data), self.k, replace=False)
        # save to variable
        self.centroids = self.data[idx, :]

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
                current_distance = calculate_euclidean_distance(
                    self.centroids[centroid], self.data[point])
                point_distances = np.append(point_distances, current_distance)

            # choose closest centroid
            closest = np.where(point_distances == np.amin(point_distances))[0].tolist()[0]
            distance_to_centroid = np.amin(point_distances)
            # print(point_distances, " ", closest)

            # append to lists
            distances.append(distance_to_centroid)
            assigned.append(closest)

        return np.array([assigned, distances])

    def relocate_centroids(self):
        idx = 0
        for cluster in self.clusters:
            if not cluster.any():
                continue
            mean = np.transpose(cluster).mean(axis=1)
            self.centroids[idx] = mean
            idx += 1
            if self.verbose:
                print("Mean: ", mean)

    def plot_with_clusters(self, include_centroids=False):
        for cluster in self.clusters:
            plt.scatter(cluster[:, 0], cluster[:, 1])
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

        # initialize random centroids
        self.create_random_centroids()

        #  repeat till optimized or till max_iterations has been reached
        for iteration in range(self.max_iterations):
            old_sum = self.sum_of_distances

            #  assign points to centroids
            self.distances = self.assign_points_to_centroids()
            #  parse to clusters
            for cluster in range(self.k):
                self.clusters[cluster] = np.array([self.data[i].tolist() for i in range(self.data.shape[0])
                                                   if self.distances[0][i] == cluster])
            #  calculate sum of distances
            self.sum_of_distances = self.distances[1].sum()

            #  plot if verbose is true
            if self.verbose:
                print(self.sum_of_distances)
                self.plot_with_clusters(include_centroids=True)

            #  break, if clusters didn't change
            if self.sum_of_distances == old_sum:
                print(f"Finished after {iteration} iterations!")
                break

            #  relocate centroids if clusters changed
            self.relocate_centroids()
