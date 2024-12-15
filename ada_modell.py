# ada_model.py

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class AdaBoostModel(BaseEstimator, ClassifierMixin):
    """
    AdaBoost Classifier with Decision Tree as the base estimator.
    """

    def __init__(
        self,
        n_estimators=50,
        learning_rate=1.0,
        estimator=None,  # Updated parameter name
        algorithm='SAMME.R',
        random_state=None
    ):
        """
        Initialize the AdaBoostModel.

        Parameters:
        - n_estimators (int): The maximum number of estimators at which boosting is terminated.
        - learning_rate (float): Learning rate shrinks the contribution of each classifier.
        - estimator (object): The base estimator from which the boosted ensemble is built.
        - algorithm (str): Algorithm to use ('SAMME' or 'SAMME.R').
        - random_state (int): Controls the randomness of the estimator.
        """
        if estimator is None:
            estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimator = estimator  # Updated attribute name
        self.algorithm = algorithm
        self.random_state = random_state
        self.ada_ = None

    def fit(self, X, y):
        """
        Fit the AdaBoost classifier.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,)
        
        Returns:
        - self: object
        """
        self.ada_ = AdaBoostClassifier(
            estimator=self.estimator,  # Updated parameter
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )
        self.ada_.fit(X, y)
        self.classes_ = self.ada_.classes_
        return self

    def predict(self, X):
        """
        Predict classes for X.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
        
        Returns:
        - y_pred: array of shape (n_samples,)
        """
        return self.ada_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
        
        Returns:
        - proba: array of shape (n_samples, n_classes)
        """
        return self.ada_.predict_proba(X)
