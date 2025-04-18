import numpy as np
import itertools

from .. import utils


class KMeans(object):
    """
    kMeans classifier object.
    """

    def __init__(self, max_iters=500, centroids_init="random"):
        """
        Call set_arguments function of this class.
        """
        self.max_iters = max_iters
        self.centroids = None
        self.centroids_init = centroids_init


    def fit(self, training_data, training_labels, n_init=10):
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
        K = utils.get_n_classes(training_labels) #number of clusters
        best_inertia = np.inf
        best_inertia_run = 0
        generator = np.random.default_rng()

        # we do multiple random initializations and keep the one where the inertia is lowest 
        # (i.e. the best in terms of the sum of square distances)
        for run in range(1, n_init+1):
            centroids = (
                self.init_centers_kmeanspp(training_data, K, generator) if self.centroids_init == "kmeanspp" else 
                self.init_centers_random(training_data, K, generator)
            )

            for i in range(self.max_iters):
                old_centroids = centroids.copy()
                distances = self.compute_distance(training_data, centroids)
                cluster_assignments = self.find_closest_cluster(distances)
                centroids = self.compute_centers(training_data, cluster_assignments, K)

                if np.allclose(old_centroids, centroids, atol=1e-6):
                    print(f"Run {run}: K-Means has converged after {i+1} iterations")
                    break
                
                if (i == self.max_iters - 1):
                    print(f"Run {run}: Max iterations reached")

            inertia = np.sum((training_data - centroids[cluster_assignments]) ** 2)

            if inertia < best_inertia:
                best_inertia = inertia
                best_inertia_run = run
                self.centroids = centroids
                self.centroid_labels = self.assign_labels_to_centers(centroids, cluster_assignments, training_labels.astype(int))

        print(f"The run with lowest inertia was number {best_inertia_run}")        
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
    

    def init_centers_random(self, data, K, generator):
        """
        Randomly pick K data points from the data as initial cluster centers.
        
        Arguments: 
            data (np.array): data of shape (N,D)
            K (int): number of clusters
        Returns:
            centers (np.array): initial cluster centers of shape (K,D)
        """
        indices = generator.permutation(data.shape[0])[:K]
        return data[indices]
    

    def init_centers_kmeanspp(self, data, K, generator):
        """
        Pick K data points from the data as initial cluster centers by taking points that have 
        higher probability of being further away from each other.
        
        Arguments: 
            data (np.array): data of shape (N,D)
            K (int): number of clusters
        Returns:
            centers (np.array): initial cluster centers of shape (K,D)
        """
        centers = []
        first_idx = generator.integers(0, data.shape[0])
        centers.append(data[first_idx])

        for _ in range(1, K):
            distances = np.min(self.compute_distance(data, np.array(centers)), axis=1)
            probs = distances / np.sum(distances)
            next_center = data[np.random.choice(data.shape[0], p=probs)]
            centers.append(next_center)
        return np.array(centers)
    

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
