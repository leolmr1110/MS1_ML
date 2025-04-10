import numpy as np
import itertools

from .. import utils


class KMeans(object):
    """
    kNN classifier object.
    """

    def __init__(self, max_iters=500):
        """
        Call set_arguments function of this class.
        """
        self.max_iters = max_iters
        self.centroids = None
        self.best_permutation = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        Hint:
            (1) Since Kmeans is unsupervised clustering, we don't need the labels for training. But you may want to use it to determine the number of clusters.
            (2) Kmeans is sensitive to initialization. You can try multiple random initializations when using this classifier.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        K = utils.get_n_classes(training_labels)
        
        self.centroids = self.init_centers_1(training_data, K)

        for i in range(self.max_iters):
            if ((i+1) % 10 == 0):
                print(f"Iteration {i+1}/{self.max_iters}...")
            old_centers = self.centroids.copy()

            distances = self.compute_distance(training_data, self.centroids)
            cluster_assignments = self.find_closest_cluster(distances)
            self.centroids = self.compute_centers(training_data, cluster_assignments, K)

            if np.all(old_centers == self.centroids):
                print(f"K-Means has converged after {i+1} iterations!")
                break
        
        distances = self.compute_distance(training_data, self.centroids)
        cluster_assignments = self.find_closest_cluster(distances)

        self.centroid_labels = self.assign_labels_to_centers(self.centroids, cluster_assignments, training_labels.astype(int))

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        distances = self.compute_distance(test_data, self.centroids)
        cluster_assignments = self.find_closest_cluster(distances)
        test_labels = self.centroid_labels[cluster_assignments]

        return test_labels
    
    def init_centers_1(self, data, K):
        """
        Randomly pick K data points from the data as initial cluster centers.
        
        Arguments: 
            data (np.array): data of shape (N,D)
            K (int): number of clusters
        Returns:
            centers (np.array): initial cluster centers of shape (K,D)
        """
        indices = np.random.permutation(data.shape[0])[:K]
        return data[indices]
    

    def compute_distance(self, data, centers):
        """
        Computes the euclidean distance between each datapoint and each center.
        
        Arguments:    
            data (np.array): data of shape (N,D)
            centers (np.array): centers of the K clusters of shape (K,D)
        Returns:
            distances: array of shape (N,K) with the distances to each cluster in K for every point in N
        """
        return np.linalg.norm(data[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    
    
    def find_closest_cluster(self, distances):
        """
        Assigns datapoints to the closest clusters.
        
        Arguments:
            distances (np.array): the distance of each data point to each cluster center of shape (N,K)
        Returns:
            cluster_assignments (np.array): cluster assignment of each datapoint of shape (N,)
        """
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments
    

    def compute_centers(self, data, cluster_assignments, K):
        """
        Computes the center of each cluster

        Arguments: 
            data (np.array): data of shape (N,D)
            cluster_assignments (np.array): the assigned cluster of each data sample of shape (N,)
            K: the number of clusters
        Returns:
            centers (np.array): the new centers of each cluster of shape is (K,D)
        """
        centers = np.zeros((K, data.shape[1]))
        for k in range(K):
            centers[k] = np.mean(data[cluster_assignments == k], axis=0)
        return centers
    

    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Uses voting to attribute a label to each cluster center.

        Arguments: 
            centers (np.array): cluster centers of shape (K,D)
            cluster_assignments (np.array): cluster assignment for each data point of shape (N,)
            true_labels (np.array): true labels of data of shape (N,)
        Returns: 
            cluster_center_label (np.array): the labels of the cluster centers of shape (K,)
        """
        K = centers.shape[0]
        cluster_center_label = np.zeros((K,))
        for k in range(K):
            labels_for_k = true_labels[cluster_assignments == k]
            cluster_center_label[k] = np.argmax(np.bincount(labels_for_k))

        return cluster_center_label
