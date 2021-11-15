import numpy as np
from src.kMeans import KMeans
import matplotlib.pyplot as plt

data = np.array([[1, 1], [1, 2], [2, 2], [1.5, 1], [2.1, 1.5], [1.4, 2.2],
                 [5, 1], [5, 2], [6, 2], [5.75, 0.5], [5.5, 1.6], [5.1, 1.9],
                 [4, 5], [4, 6], [3, 5], [3.25, 4.8], [3.5, 5.4], [3.3, 5.9]
                 ])
k = 3


def plot_error_graph():
    errors = []
    for n in range(data.shape[0]+1):
        print("Computing for n = ", n)
        if n == 0:
            continue
        solver = KMeans(verbose=False)
        solver.find_clusters(n, data)
        errors.append(solver.sum_of_distances)
    print(errors)
    plt.plot(errors)
    plt.show()


if __name__ == '__main__':
    plot_error_graph()
