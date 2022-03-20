import math
import numpy as np
import matplotlib.pyplot as plt

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


class KMeanPlusPlus:
    def __init__(self, data, k):
        self.n_centroids = 0
        self.data = data
        self.k = k
        self.dimension = self.data.shape[1]
        self.centroids = []
        self.colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'darkturquoise',
            4: 'rebeccapurple', 5: 'darkslategray', 6: 'chocolate'}

    def distance(self, x1, x2):
        # Euclidian distance between two points
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def first_centroid(self, seed=42):
        np.random.seed(seed)
        self.rand_ind = np.random.choice(self.data.shape[0], self.data.shape[0], replace = False)
        self.centroids.append(self.data[self.pop(0)])
        self.n_centroids = 1

    def pop(self,ind):
        popped_ind = self.rand_ind[ind]
        self.rand_ind = np.delete(self.rand_ind, ind)
        return popped_ind

    def set_next_centroids(self):
        while (self.n_centroids != self.k):
            distances = self.get_closest_distances()
            new_centroid_ind = self.choose_weigthed(distances)
            self.centroids.append(self.data[self.pop(new_centroid_ind )])
            self.n_centroids += 1
        self.centroids = np.array(self.centroids)

    def get_closest_distances(self):
        distances = []
        for instance in self.data:
            distances.append(self.get_closest_centroid(instance))
        return distances

    def get_closest_centroid(self,x):
        min_dist = math.inf
        for centroid in self.centroids:
            dist = self.distance(x,centroid)
            min_dist = dist if (dist < min_dist) else min_dist
        return min_dist

    def choose_weigthed(self,distances):
        squared_dist = [x**2 for x in distances]
        weights = self.get_weights(squared_dist)
        indices = [i for i in range(len(distances))]
        return np.random.choice(indices, p = weights)

    def get_weights(self, distances):
        sum_dist = np.sum(distances)
        return [x/sum_dist for x in distances]

    def assign(self,x):
        # Assign the point to the cluster 
        return np.argmin([self.distance(x, c) for c in self.centroids])

    def set_clusters(self):
        # Set the clusters for instances
        self.clusters = np.asarray([self.assign(x) for x in self.data])

    def plot_graphs(self):
        for j in range(self.dimension):
            for k in range(self.dimension):
                plt.figure(figsize = (12,8))
                for i, p in enumerate(self.data):
                    plt.scatter(p[j], p[k], color=self.colors[self.clusters[i]], alpha = 0.6, s = 50)
                for i, centroid in enumerate(self.centroids):
                    plt.scatter(centroid[j], centroid[k], color = self.colors[i], marker="x", s = 100)
                plt.xlabel(feature_columns[j])
                plt.ylabel(feature_columns[k])
                plt.show()