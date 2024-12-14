# predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class Predictor:
    def __init__(self, df, target):
        self.df = df.copy()
        self.target = target
        self.model = AdaBoostClassifier(n_estimators=100, random_state=42)
        self.pipeline = None
        
    def prepare_data(self):
        print(f"before: {self.df.shape}")
        self.df = self.df.dropna(subset=[self.target])
        print(f"after: {self.df.shape}")

        y = self.df[self.target]
        X = self.df.drop(columns=self.target)
        
        return X, y
        
    def train_model(self, X, y):
        # Build the preprocessing pipeline
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', dtype=int))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ])

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])

        # Fit the pipeline on the training data
        self.pipeline.fit(X, y)

    def predict(self, X):
        y_prob = self.pipeline.predict_proba(X)[:, 1]
        return y_prob

    def cross_val_predict_proba(self, X, y):
        # Build the preprocessing pipeline as in train_model
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', dtype=int))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', self.model)
        ])

        # Get cross-validated predicted probabilities for all data points
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"X shape: {X.shape}")
        y_prob = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]
        return y_prob

    def evaluate_model(self, y_true, y_prob):
        # Convert probabilities to binary predictions using 0.5 threshold
        y_pred = (y_prob >= 0.5).astype(int)
        print("\nEvaluation Results:")
        print(classification_report(y_true, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_true, y_prob):.2f}")