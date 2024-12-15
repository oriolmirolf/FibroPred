# cbr_model.py

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

class CBRModel(BaseEstimator, ClassifierMixin):
    """
    Case-Based Reasoning Model using K-Nearest Neighbors.
    """

    def __init__(
        self,
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        p=2,
        metric_params=None,
        n_jobs=None
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        # Initialize the internal KNeighborsClassifier
        self.knn_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )

    def fit(self, X, y):
        """
        Fit the k-NN classifier to the training data.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,)

        Returns:
        - self: object
        """
        self.knn_.fit(X, y)
        self.classes_ = self.knn_.classes_
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters:
        - X: array-like of shape (n_queries, n_features)

        Returns:
        - y_pred: ndarray of shape (n_queries,)
        """
        return self.knn_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for the provided data.

        Parameters:
        - X: array-like of shape (n_queries, n_features)

        Returns:
        - proba: ndarray of shape (n_queries, n_classes)
        """
        return self.knn_.predict_proba(X)

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """
        Find the K-neighbors of a point.

        Parameters:
        - X: array-like of shape (n_queries, n_features)
        - n_neighbors: int, default None
        - return_distance: bool, default True

        Returns:
        - distances: array
        - indices: array
        """
        return self.knn_.kneighbors(X, n_neighbors=n_neighbors, return_distance=return_distance)

    def explain(self, X, X_train=None, y_train=None):
        """
        Provide explanations for predictions by showing nearest neighbors.

        Parameters:
        - X: array-like of shape (n_queries, n_features)
        - X_train: array-like of shape (n_samples, n_features), required
        - y_train: array-like of shape (n_samples,), required

        Returns:
        - explanations: list of dict
        """
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided to generate explanations.")

        distances, indices = self.kneighbors(X)
        explanations = []
        for i in range(len(X)):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            neighbor_features = X_train[neighbor_indices]
            neighbor_targets = y_train[neighbor_indices]

            explanation = {
                'input_sample': X[i],
                'nearest_neighbors': neighbor_features,
                'neighbor_targets': neighbor_targets,
                'distances': neighbor_distances
            }
            explanations.append(explanation)
        return explanations
