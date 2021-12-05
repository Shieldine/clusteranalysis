import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from src.kMeans import KMeans as ownK
from src.agglomerative import Agglomerative as ownAgglo
from src.divisive import Divisive as ownDiv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

data = np.array([[1, 1], [1, 2], [2, 2], [1.5, 1], [2.1, 1.5], [1.4, 2.2],
                 [5, 1], [5, 2], [6, 2], [5.75, 0.5], [5.5, 1.6], [5.1, 1.9],
                 [4, 5], [4, 6], [3, 5], [3.25, 4.8], [3.5, 5.4], [3.3, 5.9],
                 [3.5, 3]
                 ])
k = 3


def plot_distortion_graph():
    distortions = []
    K = range(1, data.shape[0])
    # calculate distortions
    for K in K:
        meanModel = KMeans(n_clusters=k).fit(data)
        meanModel.fit(data)
        distortions.append(sum(np.min(cdist(data, meanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / data.shape[0])
    # plot distortions
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method')
    plt.show()


def plot_sample_dendrogram():
    Z = linkage(data, method='ward')

    # plot dendrogram
    dendro = dendrogram(Z)
    plt.title('Dendrogram')
    plt.ylabel('Euclidean distance')
    plt.show()


def plot_all_linkages():
    data_pd = pd.DataFrame(data, columns=['X', 'Y'])
    linkages = ['single', 'average', 'complete', 'ward']
    for linkage in linkages:
        hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage=linkage)
        plt.scatter(data_pd['X'], data_pd['Y'], c=hc.fit_predict(data), cmap='rainbow')
        plt.title("Clusters")
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.show()


if __name__ == '__main__':
    # KMeans, own thingy
    solver = ownK(verbose=True)
    solver.find_clusters(4, data)

    # own agglomerative clustering
    agglo = ownAgglo(verbose=True)
    agglo.find_clusters(data, 4)
    agglo.plot_clusters()

    # own divisive clustering
    div = ownDiv(verbose=True)
    div.find_clusters(data, 4)
    div.plot_clusters()
