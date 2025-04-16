import numpy as np

from ..utils import get_n_classes, label_to_onehot, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        D = training_data.shape[1]  # number of features
        C = get_n_classes(training_labels)  # number of classes
        onehot_labels = label_to_onehot(training_labels, C)
        self.weights = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            gradient = self.gradient_logistic_multi(training_data, onehot_labels)
            self.weights = self.weights - self.lr * gradient

            predictions = self.predict(training_data)
            
            if accuracy_fn(predictions, training_labels) == 100:
                break
        
        return predictions

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        softmaxed_data = self.f_softmax(test_data)
        return np.argmax(softmaxed_data, 1)
    
    def f_softmax(self, data):
        """
        Softmax function for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and 
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        exp_distance = np.exp(data @ self.weights)
        sum_terms = np.sum(exp_distance, 1)
        return exp_distance / sum_terms[:, np.newaxis]
    
    def gradient_logistic_multi(self, data, labels):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        return data.T @ (self.f_softmax(data) - labels)
