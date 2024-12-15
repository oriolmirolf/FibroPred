# pages/train_model.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from data_cleaner import DataCleaner
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def plot_confusion_matrix(cm, title):
    """
    Plots a confusion matrix using Seaborn's heatmap.

    Args:
        cm (array-like): Confusion matrix.
        title (str): Title for the plot.

    Returns:
        matplotlib.figure.Figure: The generated confusion matrix plot.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, title):
    """
    Plots the ROC curve.

    Args:
        y_true (array-like): True binary labels.
        y_proba (array-like): Target scores, can either be probability estimates or confidence values.
        title (str): Title for the plot.

    Returns:
        matplotlib.figure.Figure: The generated ROC curve plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def run():
    st.header("Train Model")
    st.write("Specify the features to include in the model and initiate training.")

    # Load data
    data_path = 'data/FibroPredCODIFICADA.xlsx'  # Adjust the path as needed
    if not os.path.exists(data_path):
        st.error(f"Data file not found at `{data_path}`. Please ensure the file exists.")
        return

    try:
        df = pd.read_excel(data_path, skiprows=1)
        #st.succeess("Data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Default features to drop
    features_to_drop_default = [
        'Detail', 'Detail on NON UIP', 'Pathology Pattern Binary',
        'Pathology pattern', 'Extras AP', 'Treatment', 'Extra',
        'Transplantation date', 'Date of death', 'Cause of death',
        'Identified Infection', 'Pathology pattern UIP, probable or CHP',
        'Severity of telomere shortening - Transform 4',
        'FVC (L) 1 year after diagnosis', 'FVC (%) 1 year after diagnosis',
        'DLCO (%) 1 year after diagnosis', 'RadioWorsening2y',
    ]

    # Default selected features
    selected_features_default = [
        'Sex', 'FamilialvsSporadic', 'Age at diagnosis', 'Comorbidities',
        'Radiological Pattern', 'Diagnosis after Biopsy',
        'Multidsciplinary committee', 'Pirfenidone', 'Nintedanib',
        'Antifibrotic Drug', 'Prednisone', 'Mycophenolate',
        'Extrapulmonary affectation', 'Associated lung cancer',
        'Other cancer', 'Type of neoplasia',
        'Blood count abnormality at diagnosis', 'Anemia',
        'Thrombocytopenia', 'Thrombocytosis', 'Lymphocytosis',
        'Lymphopenia', 'Neutrophilia', 'Neutropenia', 'Leukocytosis',
        'Leukopenia', 'LDH', 'ALT', 'AST', 'ALP', 'GGT', 'Transaminitis',
        'Cholestasis', 'Liver disease', 'FVC (%) at diagnosis',
        'DLCO (%) at diagnosis', 'Necessity of transplantation', 'Death',
        '1st degree relative', '2nd degree relative', 'More than 1 relative',
        'Genetic mutation studied in patient', 'Mutation Type',
        'Severity of telomere shortening', 'Progressive disease',
        'telomeric affectation', 'Hematologic Abnormalities',
        'Liver Problem', 'TERT', 'Final diagnosis', 'Event'
    ]

    # Feature Selection
    st.subheader("Feature Selection")
    available_features = df.columns.tolist()
    selected_features = st.multiselect(
        "Select features to include in the model:",
        options=selected_features_default,
        default=selected_features_default
    )

    if not selected_features:
        st.warning("Please select at least one feature to proceed.")
        return

    # Target Variable Selection
    target_options = ['Progressive disease', 'Event']
    st.subheader("Select Target Variable")
    target_choice = st.selectbox("Choose the target variable to predict:", target_options)

    # Train Button
    if st.button("Train Selected Model"):
        with st.spinner('Starting the training process...'):
            # Data Cleaning
            try:
                cleaner = DataCleaner(df)
                df_clean = cleaner.clean(selected_features=selected_features, features_to_drop=features_to_drop_default)
                #st.succeess("Data cleaning completed.")
            except Exception as e:
                st.error(f"Error during data cleaning: {e}")
                return

            # Handle missing values
            if df_clean.isnull().sum().sum() > 0:
                st.warning("Data contains missing values. Imputing missing values with median values...")
                df_clean.fillna(df_clean.median(), inplace=True)
                #st.succeess("Missing values imputed.")

            # Split the data
            try:
                df_train, df_test = train_test_split(
                    df_clean,
                    test_size=0.2,
                    shuffle=True,
                    random_state=42,
                    stratify=df_clean[target_choice]  # Ensures balanced splits
                )
                #st.succeess("Data split into training and testing sets.")
            except Exception as e:
                st.error(f"Error during train-test split: {e}")
                return

            if target_choice == 'Progressive disease':
                # Check if target column exists
                if 'Progressive disease' not in df_train.columns:
                    st.error("'Progressive disease' column not found in the dataset.")
                    return

                other_columns_to_drop = ['Death', 'Event', 'Necessity of transplantation']
                target_prog = 'Progressive disease'

                try:
                    # Prepare data
                    df_train_prog = df_train.drop(columns=other_columns_to_drop, errors='ignore')
                    df_test_prog = df_test.drop(columns=other_columns_to_drop, errors='ignore')

                    # Separate features and target
                    X_train_prog = df_train_prog.drop(columns=[target_prog])
                    y_train_prog = df_train_prog[target_prog].astype(int)

                    X_test_prog = df_test_prog.drop(columns=[target_prog])
                    y_test_prog = df_test_prog[target_prog].astype(int)

                    #st.succeess("Data prepared for Progressive Disease model.")
                except Exception as e:
                    st.error(f"Error preparing data for Progressive Disease model: {e}")
                    return

                # Handle Class Imbalance with SMOTE
                try:
                    smote = SMOTE(random_state=42)
                    X_train_prog_res, y_train_prog_res = smote.fit_resample(X_train_prog, y_train_prog)
                    #st.succeess("Applied SMOTE to handle class imbalance.")
                except Exception as e:
                    st.error(f"Error during SMOTE resampling: {e}")
                    return

                # Build the pipeline with AdaBoost
                pipeline_prog = Pipeline([
                    ('variance_threshold', VarianceThreshold()),
                    ('select_kbest', SelectKBest(score_func=f_classif)),
                    ('scaler', MinMaxScaler()),  # Scaling can be important for AdaBoost
                    ('ada', AdaBoostClassifier(
                        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                        random_state=42
                    ))
                ])

                # Define the parameter grid for AdaBoost
                param_grid_prog = {
                    'select_kbest__k': [5, 10, 15, 20, 'all'],
                    'ada__n_estimators': [50, 100, 200],
                    'ada__learning_rate': [0.01, 0.1, 1.0],
                    'ada__algorithm': ['SAMME.R', 'SAMME']
                }

                # Set up GridSearchCV with Stratified K-Fold
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                grid_search_prog = GridSearchCV(
                    pipeline_prog,
                    param_grid=param_grid_prog,
                    scoring='f1',
                    cv=cv,
                    n_jobs=-1,
                    verbose=0
                )

                # Training Progressive Disease model
                try:
                    grid_search_prog.fit(X_train_prog_res, y_train_prog_res)
                    #st.succeess("Progressive Disease model training completed.")
                except Exception as e:
                    st.error(f"Error during Progressive Disease model training: {e}")
                    return

                # Display best parameters
                st.subheader("Best Parameters for Progressive Disease Model")
                st.write(grid_search_prog.best_params_)

                # Evaluate on test data
                try:
                    best_pipeline_prog = grid_search_prog.best_estimator_
                    y_pred_test_prog = best_pipeline_prog.predict(X_test_prog)
                    y_pred_proba_test_prog = best_pipeline_prog.predict_proba(X_test_prog)[:, 1]

                    # Compute Final F1 Score
                    final_f1 = f1_score(y_test_prog, y_pred_test_prog)
                    st.subheader("Test Set Evaluation for Progressive Disease Model")
                    st.write(f"**Final F1 Score:** {final_f1:.4f}")

                    # Confusion Matrix for Test Set
                    cm_test_prog = confusion_matrix(y_test_prog, y_pred_test_prog)
                    fig_cm_test_prog = plot_confusion_matrix(cm_test_prog, "Progressive Disease - Test Set Confusion Matrix")
                except Exception as e:
                    st.error(f"Error during test set evaluation: {e}")
                    return

                # ROC Curve for Test Set
                try:
                    fig_roc_prog = plot_roc_curve(y_test_prog, y_pred_proba_test_prog, "Progressive Disease - Test Set ROC Curve")
                except Exception as e:
                    st.error(f"Error during ROC curve plotting: {e}")
                    fig_roc_prog = None

                # Feature Importances via Permutation Importance
                try:
                    perm_importance_prog = permutation_importance(
                        best_pipeline_prog, X_test_prog, y_test_prog, n_repeats=10, random_state=42, n_jobs=-1
                    )
                    feature_importances = pd.Series(perm_importance_prog.importances_mean, index=X_test_prog.columns)
                    feature_importances_sorted = feature_importances.sort_values(ascending=False)

                    st.subheader("Feature Importances for Progressive Disease Model")
                    fig_feat_prog, ax_feat_prog = plt.subplots(figsize=(10, 8))
                    sns.barplot(x=feature_importances_sorted.values, y=feature_importances_sorted.index, ax=ax_feat_prog)
                    ax_feat_prog.set_xlabel('Importance Score')
                    ax_feat_prog.set_ylabel('Features')
                    ax_feat_prog.set_title("Permutation Feature Importances")
                    plt.tight_layout()
                except Exception as e:
                    st.error(f"Error computing feature importances: {e}")
                    fig_feat_prog = None

                # Visualization of Confusion Matrix, ROC Curve, and Feature Importances in the Same Row
                st.subheader("Evaluation Metrics for Progressive Disease Model")
                cols = st.columns(3)
                with cols[0]:
                    st.pyplot(fig_cm_test_prog)
                with cols[1]:
                    if fig_roc_prog:
                        st.pyplot(fig_roc_prog)
                with cols[2]:
                    if fig_feat_prog:
                        st.pyplot(fig_feat_prog)

                # **Feature Selection Based on Permutation Importance**
                try:
                    selected_features_prog = feature_importances_sorted[feature_importances_sorted > 0].index.tolist()
                    if not selected_features_prog:
                        st.warning("No features with importance > 0 were found. Retaining all features.")
                        selected_features_prog = X_test_prog.columns.tolist()
                    # else:
                        #st.succeess(f"Selected {len(selected_features_prog)} features with importance > 0 for retraining.")
                except Exception as e:
                    st.error(f"Error during feature selection: {e}")
                    return

                # Save selected features to a JSON file
                selected_features_prog_path = 'models/selected_features_prog.json'
                try:
                    with open(selected_features_prog_path, 'w') as f:
                        json.dump(selected_features_prog, f)
                    #st.succeess(f"Selected features saved at `{selected_features_prog_path}`.")
                except Exception as e:
                    st.error(f"Error saving selected features for Progressive Disease model: {e}")
                    return

                # Retrain the model with selected features
                st.subheader("Retraining Progressive Disease Model with Selected Features")
                try:
                    # Prepare resampled training data with selected features
                    X_train_prog_selected = X_train_prog_res[selected_features_prog]
                    X_test_prog_selected = X_test_prog[selected_features_prog]

                    # Update the pipeline to use only selected features
                    # Removed SelectKBest since features are already selected
                    pipeline_prog_selected = Pipeline([
                        ('variance_threshold', VarianceThreshold()),
                        ('scaler', MinMaxScaler()),
                        ('ada', AdaBoostClassifier(
                            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                            n_estimators=grid_search_prog.best_params_['ada__n_estimators'],
                            learning_rate=grid_search_prog.best_params_['ada__learning_rate'],
                            algorithm=grid_search_prog.best_params_['ada__algorithm'],
                            random_state=42
                        ))
                    ])

                    # Train the model
                    pipeline_prog_selected.fit(X_train_prog_selected, y_train_prog_res)
                    #st.succeess("Progressive Disease model retrained with selected features.")
                except Exception as e:
                    st.error(f"Error during retraining with selected features: {e}")
                    return

                # Evaluate the retrained model
                try:
                    y_pred_test_prog_sel = pipeline_prog_selected.predict(X_test_prog_selected)
                    y_pred_proba_test_prog_sel = pipeline_prog_selected.predict_proba(X_test_prog_selected)[:, 1]

                    # Compute Final F1 Score
                    final_f1_sel = f1_score(y_test_prog, y_pred_test_prog_sel)
                    st.subheader("Test Set Evaluation for Retrained Progressive Disease Model")
                    st.write(f"**Final F1 Score:** {final_f1_sel:.4f}")

                    # Confusion Matrix for Test Set
                    cm_test_prog_sel = confusion_matrix(y_test_prog, y_pred_test_prog_sel)
                    fig_cm_test_prog_sel = plot_confusion_matrix(cm_test_prog_sel, "Retrained Progressive Disease - Test Set Confusion Matrix")
                except Exception as e:
                    st.error(f"Error during test set evaluation of retrained model: {e}")
                    return

                # ROC Curve for Test Set
                try:
                    fig_roc_prog_sel = plot_roc_curve(y_test_prog, y_pred_proba_test_prog_sel, "Retrained Progressive Disease - Test Set ROC Curve")
                except Exception as e:
                    st.error(f"Error during ROC curve plotting of retrained model: {e}")
                    fig_roc_prog_sel = None

                # Feature Importances via Permutation Importance for Retrained Model
                try:
                    perm_importance_prog_sel = permutation_importance(
                        pipeline_prog_selected, X_test_prog_selected, y_test_prog, n_repeats=10, random_state=42, n_jobs=-1
                    )
                    feature_importances_sel = pd.Series(perm_importance_prog_sel.importances_mean, index=X_test_prog_selected.columns)
                    feature_importances_sorted_sel = feature_importances_sel.sort_values(ascending=False)

                    st.subheader("Feature Importances for Retrained Progressive Disease Model")
                    fig_feat_prog_sel, ax_feat_prog_sel = plt.subplots(figsize=(10, 8))
                    sns.barplot(x=feature_importances_sorted_sel.values, y=feature_importances_sorted_sel.index, ax=ax_feat_prog_sel)
                    ax_feat_prog_sel.set_xlabel('Importance Score')
                    ax_feat_prog_sel.set_ylabel('Features')
                    ax_feat_prog_sel.set_title("Permutation Feature Importances (Retrained Model)")
                    plt.tight_layout()
                except Exception as e:
                    st.error(f"Error computing feature importances for retrained model: {e}")
                    fig_feat_prog_sel = None

                # Visualization of Confusion Matrix, ROC Curve, and Feature Importances in the Same Row
                st.subheader("Evaluation Metrics for Retrained Progressive Disease Model")
                cols_sel = st.columns(3)
                with cols_sel[0]:
                    st.pyplot(fig_cm_test_prog_sel)
                with cols_sel[1]:
                    if fig_roc_prog_sel:
                        st.pyplot(fig_roc_prog_sel)
                with cols_sel[2]:
                    if fig_feat_prog_sel:
                        st.pyplot(fig_feat_prog_sel)

                # **Final Retraining on the Entire Dataset with Selected Features**
                st.subheader("Final Retraining on Entire Dataset with Selected Features")
                try:
                    # Prepare entire dataset
                    df_full_prog = df_clean.drop(columns=other_columns_to_drop, errors='ignore')
                    X_full_prog = df_full_prog[selected_features_prog]
                    y_full_prog = df_full_prog[target_prog].astype(int)

                    # Apply SMOTE on the entire dataset with selected features
                    smote_full = SMOTE(random_state=42)
                    X_full_prog_res, y_full_prog_res = smote_full.fit_resample(X_full_prog, y_full_prog)

                    # Retrain the pipeline with selected features and best parameters
                    final_pipeline_prog = Pipeline([
                        ('variance_threshold', VarianceThreshold()),
                        ('scaler', MinMaxScaler()),
                        ('ada', AdaBoostClassifier(
                            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                            n_estimators=grid_search_prog.best_params_['ada__n_estimators'],
                            learning_rate=grid_search_prog.best_params_['ada__learning_rate'],
                            algorithm=grid_search_prog.best_params_['ada__algorithm'],
                            random_state=42
                        ))
                    ])

                    final_pipeline_prog.fit(X_full_prog_res, y_full_prog_res)
                    #st.succeess("Final Progressive Disease model retrained on the entire dataset with selected features.")
                except Exception as e:
                    st.error(f"Error during final retraining on entire dataset: {e}")
                    return

                # Save the final model
                try:
                    final_model_path = 'models/progressive_disease_model_final.pkl'
                    joblib.dump(final_pipeline_prog, final_model_path)
                    #st.succeess(f"Final Progressive Disease model saved at `{final_model_path}`.")
                except Exception as e:
                    st.error(f"Error saving the final Progressive Disease model: {e}")
                    return

            elif target_choice == 'Event':
                # Event Prediction depends on Progressive Disease predictions
                # st.warning("Event prediction model requires Progressive Disease probabilities as input.")
                # st.write("Please ensure the Progressive Disease model is trained and available.")

                # Check if Progressive Disease model exists
                prog_model_path = 'models/progressive_disease_model_final.pkl'
                prog_features_path = 'models/selected_features_prog.json'
                if not os.path.exists(prog_model_path):
                    st.error(f"Final Progressive Disease model not found at `{prog_model_path}`. Please train it first.")
                    return

                try:
                    # Load Progressive Disease model
                    progressive_model = joblib.load(prog_model_path)
                    #st.succeess("Final Progressive Disease model loaded successfully.")
                except Exception as e:
                    st.error(f"Error loading Progressive Disease model: {e}")
                    return

                try:
                    # Prepare data for Event prediction
                    df_event = df_clean.copy()

                    with open(prog_features_path, 'r') as f:
                        selected_features_prog = json.load(f)

                    # Generate Progressive Disease probabilities for the entire dataset
                    X_event = df_event.drop(columns=['Progressive disease', 'Death', 'Event', 'Necessity of transplantation'], errors='ignore')
                    
                    # Filter columns based on the selected features from the JSON file
                    X_event = X_event[selected_features_prog]
                    prog_probs = progressive_model.predict_proba(X_event)[:, 1]
                    df_event['ProgressiveDisease_Prob'] = prog_probs > 0.5

                    df_event.drop(columns=['Progressive disease', 'Death', 'Necessity of transplantation'], inplace=True)

                    # Define target for Event
                    target_event = 'Event'

                    X_event_final = df_event.drop(columns=[target_event], errors='ignore')
                    y_event_final = df_event[target_event].astype(int)

                    #st.succeess("Data prepared for Event model.")
                except Exception as e:
                    st.error(f"Error preparing data for Event model: {e}")
                    return

                # Split Event data into train/test
                try:
                    df_train_event, df_test_event = train_test_split(
                        df_event,
                        test_size=0.2,
                        shuffle=True,
                        random_state=42,
                        stratify=df_event[target_event]
                    )
                    #st.succeess("Event data split into training and testing sets.")
                except Exception as e:
                    st.error(f"Error during Event data train-test split: {e}")
                    return

                try:
                    # Prepare training data with SMOTE
                    X_train_event = df_train_event.drop(columns=[target_event], errors='ignore')
                    y_train_event = df_train_event[target_event].astype(int)

                    smote_event = SMOTE(random_state=42)
                    X_train_event_res, y_train_event_res = smote_event.fit_resample(X_train_event, y_train_event)
                    #st.succeess("Applied SMOTE to handle class imbalance for Event model.")
                except Exception as e:
                    st.error(f"Error during SMOTE resampling for Event model: {e}")
                    return

                # Build the pipeline for Event model with AdaBoost
                pipeline_event = Pipeline([
                    ('variance_threshold', VarianceThreshold()),
                    ('select_kbest', SelectKBest(score_func=f_classif)),
                    ('scaler', MinMaxScaler()),  # Scaling can be important for AdaBoost
                    ('ada', AdaBoostClassifier(
                        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                        random_state=42
                    ))
                ])

                # Define the parameter grid for Event model
                param_grid_event = {
                    'select_kbest__k': [5, 10, 15, 20, 25, 30, 35, 40, 45, 'all'],
                    'ada__n_estimators': [5, 10, 15, 25, 50, 100, 200],
                    'ada__learning_rate': [0.1, 1.0, 1.5],
                    'ada__algorithm': ['SAMME.R', 'SAMME']
                }

                # Set up GridSearchCV with Stratified K-Fold
                cv_event = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                grid_search_event = GridSearchCV(
                    pipeline_event,
                    param_grid=param_grid_event,
                    scoring='f1',
                    cv=cv_event,
                    n_jobs=-1,
                    verbose=0
                )

                # Training Event model
                try:
                    grid_search_event.fit(X_train_event_res, y_train_event_res)
                    #st.succeess("Event model training completed.")
                except Exception as e:
                    st.error(f"Error during Event model training: {e}")
                    return

                # Display best parameters for Event model
                st.subheader("Best Parameters for Event Model")
                st.write(grid_search_event.best_params_)

                # Evaluate on test data for Event model
                try:
                    best_pipeline_event = grid_search_event.best_estimator_
                    y_pred_test_event = best_pipeline_event.predict(df_test_event.drop(columns=[target_event], errors='ignore'))
                    y_pred_proba_test_event = best_pipeline_event.predict_proba(df_test_event.drop(columns=[target_event], errors='ignore'))[:, 1]

                    # Compute Final F1 Score
                    final_f1_event = f1_score(df_test_event[target_event], y_pred_test_event)
                    st.subheader("Test Set Evaluation for Event Model")
                    st.write(f"**Final F1 Score:** {final_f1_event:.4f}")

                    # Confusion Matrix for Test Set
                    cm_test_event = confusion_matrix(df_test_event[target_event], y_pred_test_event)
                    fig_cm_test_event = plot_confusion_matrix(cm_test_event, "Event Model - Test Set Confusion Matrix")
                except Exception as e:
                    st.error(f"Error during test set evaluation for Event model: {e}")
                    return

                # ROC Curve for Test Set
                try:
                    fig_roc_event = plot_roc_curve(df_test_event[target_event], y_pred_proba_test_event, "Event Model - Test Set ROC Curve")
                except Exception as e:
                    st.error(f"Error during ROC curve plotting for Event model: {e}")
                    fig_roc_event = None

                # Feature Importances via Permutation Importance for Event model
                try:
                    perm_importance_event = permutation_importance(
                        best_pipeline_event, df_test_event.drop(columns=[target_event], errors='ignore'), df_test_event[target_event], n_repeats=10, random_state=42, n_jobs=-1
                    )
                    feature_importances_event = pd.Series(perm_importance_event.importances_mean, index=df_test_event.drop(columns=[target_event], errors='ignore').columns)
                    feature_importances_sorted_event = feature_importances_event.sort_values(ascending=False)

                    st.subheader("Feature Importances for Event Model")
                    fig_feat_event, ax_feat_event = plt.subplots(figsize=(10, 8))
                    sns.barplot(x=feature_importances_sorted_event.values, y=feature_importances_sorted_event.index, ax=ax_feat_event)
                    ax_feat_event.set_xlabel('Importance Score')
                    ax_feat_event.set_ylabel('Features')
                    ax_feat_event.set_title("Permutation Feature Importances")
                    plt.tight_layout()
                except Exception as e:
                    st.error(f"Error computing feature importances for Event model: {e}")
                    fig_feat_event = None

                # Visualization of Confusion Matrix, ROC Curve, and Feature Importances in the Same Row
                st.subheader("Evaluation Metrics for Event Model")
                cols_event = st.columns(3)
                with cols_event[0]:
                    st.pyplot(fig_cm_test_event)
                with cols_event[1]:
                    if fig_roc_event:
                        st.pyplot(fig_roc_event)
                with cols_event[2]:
                    if fig_feat_event:
                        st.pyplot(fig_feat_event)

                # **Feature Selection Based on Permutation Importance**
                try:
                    selected_features_event = feature_importances_sorted_event[feature_importances_sorted_event > 0].index.tolist() + ['ProgressiveDisease_Prob']
                    if not selected_features_event:
                        st.warning("No features with importance > 0 were found. Retaining all features.")
                        selected_features_event = df_test_event.drop(columns=[target_event], errors='ignore').columns.tolist()
                    # else:
                        #st.succeess(f"Selected {len(selected_features_event)} features with importance > 0 for retraining.")
                except Exception as e:
                    st.error(f"Error during feature selection: {e}")
                    return

                # Save selected features to a JSON file
                selected_features_event_path = 'models/selected_features_event.json'
                try:
                    with open(selected_features_event_path, 'w') as f:
                        json.dump(selected_features_event, f)
                    #st.succeess(f"Selected features saved at `{selected_features_event_path}`.")
                except Exception as e:
                    st.error(f"Error saving selected features for Event model: {e}")
                    return

                # Retrain the model with selected features
                st.subheader("Retraining Event Model with Selected Features")
                try:
                    # Prepare resampled training data with selected features
                    X_train_event_selected = X_train_event_res[selected_features_event]
                    X_test_event_selected = df_test_event.drop(columns=[target_event], errors='ignore')[selected_features_event]

                    # Update the pipeline to use only selected features
                    # Removed SelectKBest since features are already selected
                    pipeline_event_selected = Pipeline([
                        ('variance_threshold', VarianceThreshold()),
                        ('scaler', MinMaxScaler()),
                        ('ada', AdaBoostClassifier(
                            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                            n_estimators=grid_search_event.best_params_['ada__n_estimators'],
                            learning_rate=grid_search_event.best_params_['ada__learning_rate'],
                            algorithm=grid_search_event.best_params_['ada__algorithm'],
                            random_state=42
                        ))
                    ])

                    # Train the model
                    pipeline_event_selected.fit(X_train_event_selected, y_train_event_res)
                    #st.succeess("Event model retrained with selected features.")
                except Exception as e:
                    st.error(f"Error during retraining with selected features: {e}")
                    return

                # Evaluate the retrained model
                try:
                    y_pred_test_event_sel = pipeline_event_selected.predict(X_test_event_selected)
                    y_pred_proba_test_event_sel = pipeline_event_selected.predict_proba(X_test_event_selected)[:, 1]

                    # Compute Final F1 Score
                    final_f1_event_sel = f1_score(df_test_event[target_event], y_pred_test_event_sel)
                    st.subheader("Test Set Evaluation for Retrained Event Model")
                    st.write(f"**Final F1 Score:** {final_f1_event_sel:.4f}")

                    # Confusion Matrix for Test Set
                    cm_test_event_sel = confusion_matrix(df_test_event[target_event], y_pred_test_event_sel)
                    fig_cm_test_event_sel = plot_confusion_matrix(cm_test_event_sel, "Retrained Event - Test Set Confusion Matrix")
                except Exception as e:
                    st.error(f"Error during test set evaluation of retrained Event model: {e}")
                    return

                # ROC Curve for Test Set
                try:
                    fig_roc_event_sel = plot_roc_curve(df_test_event[target_event], y_pred_proba_test_event_sel, "Retrained Event - Test Set ROC Curve")
                except Exception as e:
                    st.error(f"Error during ROC curve plotting of retrained Event model: {e}")
                    fig_roc_event_sel = None

                # Feature Importances via Permutation Importance for Retrained Event model
                try:
                    perm_importance_event_sel = permutation_importance(
                        pipeline_event_selected, X_test_event_selected, df_test_event[target_event], n_repeats=10, random_state=42, n_jobs=-1
                    )
                    feature_importances_event_sel = pd.Series(perm_importance_event_sel.importances_mean, index=X_test_event_selected.columns)
                    feature_importances_sorted_event_sel = feature_importances_event_sel.sort_values(ascending=False)

                    st.subheader("Feature Importances for Retrained Event Model")
                    fig_feat_event_sel, ax_feat_event_sel = plt.subplots(figsize=(10, 8))
                    sns.barplot(x=feature_importances_sorted_event_sel.values, y=feature_importances_sorted_event_sel.index, ax=ax_feat_event_sel)
                    ax_feat_event_sel.set_xlabel('Importance Score')
                    ax_feat_event_sel.set_ylabel('Features')
                    ax_feat_event_sel.set_title("Permutation Feature Importances (Retrained Event Model)")
                    plt.tight_layout()
                except Exception as e:
                    st.error(f"Error computing feature importances for retrained Event model: {e}")
                    fig_feat_event_sel = None

                # Visualization of Confusion Matrix, ROC Curve, and Feature Importances in the Same Row
                st.subheader("Evaluation Metrics for Retrained Event Model")
                cols_event_sel = st.columns(3)
                with cols_event_sel[0]:
                    st.pyplot(fig_cm_test_event_sel)
                with cols_event_sel[1]:
                    if fig_roc_event_sel:
                        st.pyplot(fig_roc_event_sel)
                with cols_event_sel[2]:
                    if fig_feat_event_sel:
                        st.pyplot(fig_feat_event_sel)

                # **Final Retraining on the Entire Dataset with Selected Features**
                st.subheader("Final Retraining on Entire Dataset with Selected Features")
                try:
                    # Prepare entire dataset
                    X_full_event = X_event_final[selected_features_event]
                    y_full_event = y_event_final

                    # Apply SMOTE on the entire dataset with selected features
                    smote_full_event = SMOTE(random_state=42)
                    X_full_event_res, y_full_event_res = smote_full_event.fit_resample(X_full_event, y_full_event)

                    # Retrain the pipeline with selected features and best parameters
                    final_pipeline_event = Pipeline([
                        ('variance_threshold', VarianceThreshold()),
                        ('scaler', MinMaxScaler()),
                        ('ada', AdaBoostClassifier(
                            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                            n_estimators=grid_search_event.best_params_['ada__n_estimators'],
                            learning_rate=grid_search_event.best_params_['ada__learning_rate'],
                            algorithm=grid_search_event.best_params_['ada__algorithm'],
                            random_state=42
                        ))
                    ])

                    final_pipeline_event.fit(X_full_event_res, y_full_event_res)
                    #st.succeess("Final Event model retrained on the entire dataset with selected features.")
                except Exception as e:
                    st.error(f"Error during final retraining on entire dataset for Event model: {e}")
                    return

                # Save the final Event model
                try:
                    final_event_model_path = 'models/event_model_final.pkl'
                    joblib.dump(final_pipeline_event, final_event_model_path)
                    #st.succeess(f"Final Event model saved at `{final_event_model_path}`.")
                except Exception as e:
                    st.error(f"Error saving the final Event model: {e}")
                    return
