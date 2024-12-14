# predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

class Predictor:
    def __init__(self, df, target, scoring_metric='f1'):
        """
        Initialize the Predictor class.

        Parameters:
        - df: pandas DataFrame containing the data.
        - target: string, name of the target column.
        - scoring_metric: string, scoring metric for GridSearchCV (default: 'f1').
        """
        self.df = df.copy()
        self.target = target
        self.best_model = None  # To store the best estimator after GridSearchCV
        self.pipeline = None
        self.feature_names = None  # To store feature names after preprocessing
        self.preprocessor = None
        self.scoring_metric = scoring_metric  # Scoring metric for GridSearchCV

    def prepare_data(self):
        """
        Prepare the data by removing rows with missing target values.

        Returns:
        - X: pandas DataFrame of features.
        - y: pandas Series of target.
        """
        print(f"Before dropping missing target '{self.target}': {self.df.shape}")
        self.df = self.df.dropna(subset=[self.target])
        print(f"After dropping missing target '{self.target}': {self.df.shape}")

        y = self.df[self.target]
        X = self.df.drop(columns=self.target)

        return X, y

    def train_model(self, X, y):
        """
        Train the model using GridSearchCV to find the best classifier and hyperparameters.

        Parameters:
        - X: pandas DataFrame of features.
        - y: pandas Series of target.
        """
        # Identify numeric and categorical columns
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Define transformers for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine transformers into a ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )

        # **Add Variance Threshold Step**
        variance_threshold = VarianceThreshold()  # Remove features with zero variance

        # **Add Feature Selection Step**
        feature_selector = SelectKBest(score_func=f_classif)  # You can also try mutual_info_classif

        # Define classifiers and their parameter grids
        classifiers = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }

        param_grids = {
            'RandomForest': {
                'feature_selector__k': [5, 10, 15, 20, 25, 30, 35, 40, 45, 'all'],
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 5, 10],
                'classifier__min_samples_split': [2, 5],
            },
            'AdaBoost': {
                'feature_selector__k': [5, 10, 15, 20, 25, 30, 35, 40, 45, 'all'],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.5, 1.0, 1.5]
            },
            'LogisticRegression': {
                'feature_selector__k': [5, 10, 15, 20, 25, 30, 35, 40, 45, 'all'],
                'classifier__C': [0.1, 1.0, 10],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['lbfgs']
            }
        }

        # Set up the overall pipeline structure
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('variance_threshold', variance_threshold),  # Add VarianceThreshold here
            ('feature_selector', feature_selector),
            ('classifier', None)  # Placeholder, will be set in GridSearchCV
        ])

        best_score = -np.inf
        best_classifier_name = None

        # Iterate over classifiers
        for name, clf in classifiers.items():
            print(f"\nTraining and tuning {name}...")
            # Update the classifier in the pipeline
            self.pipeline.set_params(classifier=clf)

            # Set up GridSearchCV
            param_grid = param_grids[name]
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=5, scoring=self.scoring_metric, n_jobs=-1, verbose=0
            )
            grid_search.fit(X, y)

            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best cross-validation {self.scoring_metric.upper()} for {name}: {grid_search.best_score_:.4f}")

            # Update the best model if this one is better
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                best_classifier_name = name

        print(f"\nBest classifier after tuning: {best_classifier_name}")
        print(f"Best {self.scoring_metric.upper()} score: {best_score:.4f}")

        # Retrieve feature names after preprocessing and feature selection
        self._get_feature_names(num_cols, cat_cols)

        # Print feature importances or coefficients
        self._print_feature_importances(best_classifier_name)

    def _get_feature_names(self, num_cols, cat_cols):
        """
        Retrieve feature names after preprocessing, variance thresholding, and feature selection.

        Parameters:
        - num_cols: list of numeric column names.
        - cat_cols: list of categorical column names.
        """
        # Get names from numeric features
        feature_names = []
        if num_cols:
            feature_names.extend(num_cols)

        # Get names from categorical features after one-hot encoding
        if cat_cols:
            ohe = self.best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            ohe_features = ohe.get_feature_names_out(cat_cols)
            feature_names.extend(ohe_features)

        # Get support mask from variance threshold
        vt = self.best_model.named_steps['variance_threshold']
        vt_mask = vt.get_support()

        # Apply the variance threshold mask to feature names
        feature_names_vt = np.array(feature_names)[vt_mask]

        # Get support mask from feature selector
        selector = self.best_model.named_steps['feature_selector']
        support_mask = selector.get_support()

        # Apply the mask to the variance-thresholded feature names
        self.feature_names = feature_names_vt[support_mask]

    def _print_feature_importances(self, classifier_name):
        """
        Print the feature importances from the trained model.

        Parameters:
        - classifier_name: string, name of the classifier.
        """
        if self.feature_names is None:
            print("Feature names are not available.")
            return

        classifier = self.best_model.named_steps['classifier']

        if classifier_name in ['RandomForest', 'AdaBoost']:
            importances = classifier.feature_importances_
            feature_importances = pd.Series(importances, index=self.feature_names)
        elif classifier_name == 'LogisticRegression':
            importances = classifier.coef_[0]
            feature_importances = pd.Series(importances, index=self.feature_names)
        else:
            print("Feature importances are not available for this classifier.")
            return

        # Sort feature importances in descending order
        feature_importances_sorted = feature_importances.sort_values(ascending=False)

        print("\nTop Feature Importances:")
        print(feature_importances_sorted.head(20))

        # Optional: Plot feature importances
        plt.figure(figsize=(10, 6))
        feature_importances_sorted.head(20).plot(kind='barh')
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        """
        Predict probabilities using the best trained model.

        Parameters:
        - X: pandas DataFrame of features.

        Returns:
        - y_prob: numpy array of predicted probabilities for the positive class.
        """
        y_prob = self.best_model.predict_proba(X)[:, 1]
        return y_prob

    def cross_val_predict_proba(self, X, y):
        """
        Perform cross-validated predictions using the best model.

        Parameters:
        - X: pandas DataFrame of features.
        - y: pandas Series of target.

        Returns:
        - y_prob: numpy array of cross-validated predicted probabilities for the positive class.
        """
        # Use the best model pipeline for cross-validated predictions
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Cross-Validation: X shape: {X.shape}")
        y_prob = cross_val_predict(self.best_model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
        return y_prob

    def evaluate_model(self, y_true, y_prob):
        """
        Evaluate the model's performance.

        Parameters:
        - y_true: array-like of true labels.
        - y_prob: array-like of predicted probabilities.

        Prints:
        - Classification report including precision, recall, and F1-score.
        - F1 Score.
        - ROC AUC Score.
        """
        # Convert probabilities to binary predictions using 0.5 threshold
        y_pred = (y_prob >= 0.5).astype(int)
        print("\nEvaluation Results:")
        print(classification_report(y_true, y_pred, digits=4))
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        print(f"ROC AUC Score: {roc_auc_score(y_true, y_prob):.4f}")
