# predict.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_cleaner import DataCleaner
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from math import ceil

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


features_to_drop = [
    'Detail', 'Detail on NON UIP', 'Pathology Pattern Binary',
    'Pathology pattern', 'Extras AP', 'Treatment', 'Extra',
    'Transplantation date', 'Date of death', 'Cause of death',
    'Identified Infection', 'Pathology pattern UIP, probable or CHP',
    'Severity of telomere shortening - Transform 4',
    'FVC (L) 1 year after diagnosis', 'FVC (%) 1 year after diagnosis',
    'DLCO (%) 1 year after diagnosis', 'RadioWorsening2y',
]
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


@st.cache_data
def load_and_clean_data():
    """
    Loads the Excel data, cleans it using DataCleaner, and returns both
    the cleaned and original DataFrames.
    
    Returns:
        tuple: (cleaned DataFrame, original DataFrame)
    """
    data_path = 'data/FibroPredCODIFICADA.xlsx'
    if not os.path.exists(data_path):
        st.error(f"Data file not found at `{data_path}`. Please ensure the file exists.")
        return None, None

    try:
        df_original = pd.read_excel(data_path, skiprows=1)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

    try:
        cleaner = DataCleaner(df_original)

        df_clean = cleaner.clean(selected_features=selected_features_default, features_to_drop=features_to_drop)
        # Fill missing values with median to handle any remaining NaNs
        if df_clean.isnull().sum().sum() > 0:
            df_clean.fillna(df_clean.median(), inplace=True)
    except Exception as e:
        st.error(f"Error during data cleaning: {e}")
        return None, None

    return df_clean, df_original


def plot_probability(prob, title):
    """
    Plots a simple probability bar.

    Args:
        prob (float): Probability value between 0 and 1.
        title (str): Title for the plot.

    Returns:
        matplotlib.figure.Figure: The generated probability plot.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['Negative', 'Positive'], [1 - prob, prob], color=['skyblue', 'salmon'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title(title)
    for i, v in enumerate([1 - prob, prob]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    return fig


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


def load_selected_features(path):
    """
    Loads selected features from a JSON file.

    Args:
        path (str): Path to the JSON file containing selected features.

    Returns:
        list: List of selected feature names.
    """
    try:
        with open(path, 'r') as f:
            features = json.load(f)
        return features
    except Exception as e:
        st.error(f"Error loading selected features from `{path}`: {e}")
        return []


def load_models():
    """
    Loads the trained models and selected feature lists.

    Returns:
        tuple: Loaded progressive disease model, event model, selected features for progressive disease, and selected features for event.
    """
    try:
        progressive_model = joblib.load('models/progressive_disease_model_final.pkl')
    except FileNotFoundError:
        st.error("Progressive Disease model not found. Please train the model first.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading Progressive Disease model: {e}")
        return None, None, None, None

    try:
        event_model = joblib.load('models/event_model_final.pkl')
    except FileNotFoundError:
        st.error("Event model not found. Please train the model first.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading Event model: {e}")
        return None, None, None, None

    selected_features_prog = load_selected_features('models/selected_features_prog.json')
    selected_features_event = load_selected_features('models/selected_features_event.json')

    return progressive_model, event_model, selected_features_prog, selected_features_event


def get_feature_types(df, features):
    """
    Determines which features are categorical and which are numerical.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        features (list): List of feature names.

    Returns:
        tuple: Lists of categorical and numerical feature names.
    """
    categorical_features = []
    numeric_features = []
    for f in features:
        if f in df.columns:
            unique_vals = df[f].dropna().unique()
            if df[f].dtype == 'object' or len(unique_vals) < 10:
                categorical_features.append(f)
            else:
                numeric_features.append(f)
        else:
            # If feature not in df, assume numeric
            numeric_features.append(f)
    return categorical_features, numeric_features


def load_random_patient(df_clean):
    """
    Loads a random patient from the cleaned dataset.

    Args:
        df_clean (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.Series: The cleaned data of the random patient.
    """
    try:
        random_row = df_clean.sample(n=1, random_state=None).iloc[0]
        return random_row
    except Exception as e:
        st.error(f"Error loading random patient: {e}")
        return None


def run():
    st.header("Predict")

    st.write("Input patient data or load a random patient to get predictions for Progressive Disease and Event outcomes.")

    # Load models and selected features
    progressive_model, event_model, selected_features_prog, selected_features_event = load_models()

    if not all([progressive_model, event_model, selected_features_prog, selected_features_event]):
        st.stop()

    # Load and clean data once using cached function
    df_clean, df_original = load_and_clean_data()
    if df_clean is None or df_original is None:
        st.stop()

    # Load selected features
    st.subheader("Select Prediction Target")
    target_options = ['Progressive disease', 'Event']
    target_choice = st.selectbox("Select the target variable to predict:", target_options)

    if target_choice == 'Progressive disease':
        input_features = selected_features_prog
    else:
        input_features = selected_features_event

    if not input_features:
        st.error(f"No selected features found for {target_choice} model.")
        st.stop()

    # Determine feature types
    categorical_features, numeric_features = get_feature_types(df_clean, input_features)

    # Initialize session state for inputs
    if 'user_input' not in st.session_state:
        st.session_state.user_input = {}
        for f in input_features:
            if f in categorical_features:
                options = df_clean[f].dropna().unique().tolist()
                st.session_state.user_input[f] = options[0] if options else ""
            else:
                median = df_clean[f].median() if not df_clean[f].isnull().all() else 0.0
                st.session_state.user_input[f] = float(median)

    # Function to reset random patient flag
    if 'random_patient' not in st.session_state:
        st.session_state.random_patient = None

    # Input Form
    st.subheader("Input Patient Features")
    n_cols = 4
    cols = st.columns(n_cols)
    for i, feature in enumerate(input_features):
        col = cols[i % n_cols]
        if feature in categorical_features:
            options = df_clean[feature].dropna().unique().tolist()
            # Handle case where feature might have multiple categories due to one-hot encoding
            st.session_state.user_input[feature] = col.selectbox(
                label=feature,
                options=options,
                index=options.index(st.session_state.user_input[feature]) if st.session_state.user_input[feature] in options else 0,
                key=f"{feature}_selectbox"
            )
        else:
            default_value = st.session_state.user_input[feature]
            st.session_state.user_input[feature] = col.number_input(
                label=feature,
                value=float(default_value),
                format="%.4f",
                key=f"{feature}_number_input"
            )

    # Prediction Buttons
    st.markdown("### Actions")
    colA, colB = st.columns(2)
    with colA:
        predict_manual = st.button("Predict with Input Data")
    with colB:
        predict_random = st.button("Load Random Patient")

    # Function to prepare input data
    def prepare_input(user_inputs, features):
        input_dict = {}
        for f in features:
            input_dict[f] = user_inputs[f]
        input_df = pd.DataFrame([input_dict])
        # Clean the input data
        try:
            # Combine the input with the original data to apply the same cleaning
            combined = pd.concat([df_original, input_df], ignore_index=True)
            cleaner = DataCleaner(combined)
            combined_clean = cleaner.clean(selected_features=selected_features_default, features_to_drop=features_to_drop)
            combined_clean.fillna(combined_clean.median(), inplace=True)
            # Extract the last row which corresponds to the input
            cleaned_input = combined_clean.iloc[[-1]].copy()
            return cleaned_input
        except Exception as e:
            st.error(f"Error during input data cleaning: {e}")
            return None

    # Function to display prediction results
    def display_results(model_name, prob):
        st.markdown(f"**{model_name} Probability:** {prob:.4f}")
        fig = plot_probability(prob, f"{model_name} Probability")
        st.pyplot(fig)

    # Handle Random Patient Loading
    if predict_random:
        with st.spinner("Loading random patient..."):
            try:
                cleaned_random_patient = load_random_patient(df_clean)
                if cleaned_random_patient is not None:
                    # Update user inputs
                    for f in input_features:
                        if f in categorical_features:
                            val = cleaned_random_patient[f]
                            options = df_clean[f].dropna().unique().tolist()
                            if val not in options:
                                val = options[0] if options else ""
                            st.session_state.user_input[f] = val
                        else:
                            val = cleaned_random_patient[f]
                            st.session_state.user_input[f] = float(val) if not pd.isnull(val) else 0.0
                    st.success("Random patient loaded successfully.")
                    st.session_state.random_patient = cleaned_random_patient
            except Exception as e:
                st.error(f"Error loading random patient: {e}")

    # Handle Prediction
    if predict_manual:
        with st.spinner('Making predictions...'):
            try:
                # Prepare input data
                cleaned_input = st.session_state.user_input,# prepare_input(st.session_state.user_input, input_features)
                if cleaned_input is None:
                    st.error("Failed to prepare input data for prediction.")
                    st.stop()

                # Predict Progressive Disease if needed
                if target_choice == 'Progressive disease':
                    # Ensure that the input has the correct features required by the model

                    print(cleaned_input)

                    missing_features = [feat for feat in selected_features_prog if feat not in cleaned_input.columns]
                    if missing_features:
                        st.error(f"Missing features for Progressive Disease model: {missing_features}")
                        st.stop()

                    prog_prob = progressive_model.predict_proba(cleaned_input[selected_features_prog])[:, 1][0]
                    display_results("Progressive Disease", prog_prob)

                    # If a random patient was loaded, display true value
                    if st.session_state.random_patient is not None:
                        true_prog = st.session_state.random_patient.get('Progressive disease', None)
                        if true_prog is not None:
                            st.markdown(f"**True Progressive Disease Value:** {'Positive' if true_prog == 1 else 'Negative'}")

                else:
                    # Predict Progressive Disease first
                    missing_features_prog = [feat for feat in selected_features_prog if feat not in cleaned_input.columns]
                    if missing_features_prog:
                        st.error(f"Missing features for Progressive Disease prediction: {missing_features_prog}")
                        st.stop()

                    prog_prob = progressive_model.predict_proba(cleaned_input[selected_features_prog])[:, 1][0]
                    st.markdown("### Progressive Disease Prediction")
                    display_results("Progressive Disease", prog_prob)

                    # Add Progressive Disease probability to input for Event prediction
                    cleaned_input['ProgressiveDisease_Prob'] = prog_prob

                    # Predict Event
                    # Ensure that 'ProgressiveDisease_Prob' is part of selected_features_event
                    if 'ProgressiveDisease_Prob' not in selected_features_event:
                        st.error("'ProgressiveDisease_Prob' is missing from Event selected features.")
                        st.stop()

                    missing_features_event = [feat for feat in selected_features_event if feat not in cleaned_input.columns]
                    if missing_features_event:
                        st.error(f"Missing features for Event model: {missing_features_event}")
                        st.stop()

                    event_prob = event_model.predict_proba(cleaned_input[selected_features_event])[:, 1][0]
                    st.markdown("### Event Prediction")
                    display_results("Event (Death/Transplantation)", event_prob)

                    # If a random patient was loaded, display true value
                    if st.session_state.random_patient is not None:
                        true_event = st.session_state.random_patient.get('Event', None)
                        if true_event is not None:
                            st.markdown(f"**True Event Value:** {'Positive' if true_event == 1 else 'Negative'}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")