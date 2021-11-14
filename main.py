import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1, 1], [1, 2], [2, 2], [1.5, 1], [2.1, 1.5], [1.4, 2.2],
                 [5, 1], [5, 2], [6, 2], [5.75, 0.5], [5.5, 1.6], [5.1, 1.9],
                 [4, 5], [4, 6], [3, 5], [3.25, 4.8], [3.5, 5.4], [3.3, 5.9]
                 ])
k = 3


def plot():
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


if __name__ == '__main__':
    plot()
