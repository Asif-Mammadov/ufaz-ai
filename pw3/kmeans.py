import numpy as np
import matplotlib.pyplot as plt

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


class KMeans():
    def __init__(self, data, k, max_iters=100):
        self.data = data
        self.k = k
        self.max_iters = max_iters
        self.dimension = self.data.shape[1]
        self.centroids = self.set_centroids()
        self.clusters = self.set_clusters()
        self.colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'darkturquoise',
                       4: 'rebeccapurple', 5: 'darkslategray', 6: 'chocolate'}

    def distance(self, x1, x2):
        # Euclidian distance between two points
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def assign(self, x):
        # Assign the point to the cluster
        return np.argmin([self.distance(x, c) for c in self.centroids])

    def update(self):
        # Updating the centroids and assignments
        for i in range(self.max_iters):
            # Updates centroids
            for cluster in range(self.k):
                self.centroids[cluster] = np.mean([self.data[x] for x in range(
                    len(self.data)) if self.clusters[x] == cluster], axis=0)
            # reset clusters
            self.clusters = self.set_clusters()

    def get_centroids(self):
        return self.centroids

    def set_centroids(self, seed=42):
        # Generate random initial clusters centroids
        np.random.seed(seed)
        self.centroids = np.zeros((self.k, self.dimension))
        # returns and array of random unique indices and the centroids are set equal to the data of those indices
        rand_ind = np.random.choice(self.data.shape[0], self.k, replace=False)
        for i in range(self.k):
            self.centroids[i] = self.data[rand_ind[i]]
        return self.centroids

    def get_clusters(self):
        return self.clusters

    def set_clusters(self):
        # Set the clusters for instances
        self.clusters = np.asarray([self.assign(x) for x in self.data])
        return self.clusters

    def plot_graphs(self):
        for j in range(self.dimension):
            for k in range(self.dimension):
                plt.figure(figsize=(12, 8))
                for i, p in enumerate(self.data):
                    plt.scatter(
                        p[j], p[k], color=self.colors[self.clusters[i]], alpha = 0.6, s=50)
                for i, centroid in enumerate(self.centroids):
                    plt.scatter(
                        centroid[j], centroid[k], color=self.colors[i], marker="x", s=100)
                plt.xlabel(feature_columns[j])
                plt.ylabel(feature_columns[k])
                plt.show()
