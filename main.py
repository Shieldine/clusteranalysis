import numpy as np
from src.kMeans import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

data = np.array([[1, 1], [1, 2], [2, 2], [1.5, 1], [2.1, 1.5], [1.4, 2.2],
                 [5, 1], [5, 2], [6, 2], [5.75, 0.5], [5.5, 1.6], [5.1, 1.9],
                 [4, 5], [4, 6], [3, 5], [3.25, 4.8], [3.5, 5.4], [3.3, 5.9]
                 ])
k = 3


def plot_error_graph():
    errors = []
    n = data.shape[0]
    for n in range(1, n+1):
        print("Computing for n = ", n)
        if n == 0:
            continue
        solver = KMeans(verbose=False)
        solver.find_clusters(n, data)
        errors.append(solver.sum_of_distances)

    plt.plot(range(1, n+1), errors)
    plt.xticks(list(range(1, n)))
    plt.xlabel('n')
    plt.ylabel('sum of errors')
    plt.show()


if __name__ == '__main__':
    # KMeans, own thingy
    # solver = KMeans(verbose=True)
    # solver.find_clusters(3, data)

    # agglomerative, using library
    Z = linkage(data, method='ward')

    # plotting dendrogram
    dendro = dendrogram(Z)
    plt.title('Dendrogram')
    plt.ylabel('Euclidean distance')
    plt.show()

    data_pd = pd.DataFrame(data, columns=['X', 'Y'])
    linkages = ['single', 'average', 'complete', 'ward']
    for linkage in linkages:
        hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage=linkage)
        plt.scatter(data_pd['X'], data_pd['Y'], c=hc.fit_predict(data), cmap='rainbow')
        plt.title("Clusters")
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.show()
