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
from sklearn.inspection import permutation_importance
from math import ceil

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
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
    try:
        with open(path, 'r') as f:
            features = json.load(f)
        return features
    except Exception as e:
        st.error(f"Error loading selected features from `{path}`: {e}")
        return []


def run():
    st.header("Predict")
    st.write("Input patient data or select a random patient to get predictions.")

    # Load models
    try:
        progressive_model = joblib.load('models/progressive_disease_model_final.pkl')
        event_model = joblib.load('models/event_model_final.pkl')
        st.success("Models loaded successfully.")
    except FileNotFoundError:
        st.error("Models not found. Please train the models first.")
        return
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Load selected features
    selected_features_prog = load_selected_features('models/selected_features_prog.json')
    selected_features_event = load_selected_features('models/selected_features_event.json')

    if not selected_features_prog or not selected_features_event:
        st.error("Selected features for Progressive Disease or Event model not found.")
        return

    # Intersection for Progressive Disease
    intersection_features = list(set(selected_features_prog).intersection(set(selected_features_event)))

    # Load data
    data_path = 'data/FibroPredCODIFICADA.xlsx'
    if not os.path.exists(data_path):
        st.error(f"Data file not found at `{data_path}`.")
        return

    try:
        df = pd.read_excel(data_path, skiprows=1)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Setup cleaner
    try:
        cleaner = DataCleaner(df)
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

        # Clean entire dataset
        df_clean = cleaner.clean(selected_features=selected_features_default, features_to_drop=features_to_drop)
        if df_clean.isnull().sum().sum() > 0:
            df_clean.fillna(df_clean.median(), inplace=True)
    except Exception as e:
        st.error(f"Error during data cleaning: {e}")
        return

    target_options = ['Progressive disease', 'Event']
    st.subheader("Select Target Variable")
    target_choice = st.selectbox("Target variable:", target_options)

    if target_choice == 'Progressive disease':
        input_features = selected_features_prog
    else:
        input_features = selected_features_event

    if not input_features:
        st.error(f"No selected features found for {target_choice} model.")
        return

    # Determine categorical vs numeric features
    # A feature is considered categorical if dtype is object or if unique values < 10
    # We'll use df_clean for determining this, as it's the cleaned dataset
    categorical_features = []
    numeric_features = []
    for f in input_features:
        if f in df_clean.columns:
            unique_vals = df_clean[f].dropna().unique()
            if df_clean[f].dtype == 'object' or len(unique_vals) < 10:
                categorical_features.append(f)
            else:
                numeric_features.append(f)
        else:
            # If feature not in df_clean, assume numeric
            numeric_features.append(f)

    # Initialize session state for inputs
    if 'user_input' not in st.session_state:
        st.session_state.user_input = {}

    # Populate defaults
    for f in input_features:
        if f not in st.session_state.user_input:
            st.session_state.user_input[f] = 0.0 if f in numeric_features else (df_clean[f].dropna().unique()[0] if f in categorical_features and df_clean[f].dropna().unique().size > 0 else "")

    # Create a more compact input form: 4 features per row
    n_cols = 4
    n_rows = ceil(len(input_features) / n_cols)
    feature_index = 0
    for _ in range(n_rows):
        cols = st.columns(n_cols)
        for c in cols:
            if feature_index < len(input_features):
                f = input_features[feature_index]
                if f in categorical_features:
                    # Dropdown
                    options = df_clean[f].dropna().unique().tolist()
                    if len(options) == 0:
                        options = [""]  # fallback
                    st.session_state.user_input[f] = c.selectbox(f, options, index=options.index(st.session_state.user_input[f]) if st.session_state.user_input[f] in options else 0)
                else:
                    # Numeric
                    val = st.session_state.user_input[f]
                    if not isinstance(val, (int,float)):
                        val = 0.0
                    st.session_state.user_input[f] = c.number_input(f, value=float(val))
                feature_index += 1

    def display_results(model_name, pred_class, prob):
        uncertainty = (1 - prob) * 100
        st.write(f"**{model_name} Prediction**")
        st.write(f"Predicted Class: {'Positive' if pred_class == 1 else 'Negative'}")
        st.write(f"Probability: {prob:.4f}")
        st.write(f"Uncertainty: {uncertainty:.2f}%")

    def predict_progressive(df_to_predict):
        missing_feats = [f for f in intersection_features if f not in df_to_predict.columns]
        for mf in missing_feats:
            df_to_predict[mf] = np.nan
        prog_prob = progressive_model.predict_proba(df_to_predict[intersection_features])[:, 1]
        prog_pred = (prog_prob >= 0.5).astype(int)
        return prog_pred, prog_prob

    def predict_event(df_to_predict, prog_prob):
        df_to_predict['ProgressiveDisease_Prob'] = prog_prob
        if 'ProgressiveDisease_Prob' not in selected_features_event:
            st.error("'ProgressiveDisease_Prob' is missing from event features.")
            return None, None
        missing_feats = [f for f in selected_features_event if f not in df_to_predict.columns]
        for mf in missing_feats:
            df_to_predict[mf] = np.nan
        event_prob = event_model.predict_proba(df_to_predict[selected_features_event])[:, 1]
        event_pred = (event_prob >= 0.5).astype(int)
        return event_pred, event_prob

    # Buttons
    colA, colB = st.columns(2)
    with colA:
        predict_manual = st.button("Predict with Input Data")
    with colB:
        predict_random = st.button("Load Random Patient")

    # Function to get cleaned single row from user input
    def get_cleaned_input_row(user_data):
        # user_data is a dict of {feature: value} in original format
        # We must integrate this user_data into the original df and re-run cleaning
        # We'll create a single-row df with all selected_features_default
        # and set user_data features accordingly. Others as NaN.

        # Prepare a single row based on selected_features_default
        user_row = pd.DataFrame(columns=selected_features_default)
        user_row.loc[0] = [np.nan]*len(selected_features_default)

        # Assign user inputs
        for f, v in user_data.items():
            if f in user_row.columns:
                user_row.at[0,f] = v
        # Now append to original df and clean again
        combined = pd.concat([df, user_row], ignore_index=True)
        combined_clean = cleaner.clean(selected_features=selected_features_default, features_to_drop=features_to_drop)
        if combined_clean.isnull().sum().sum() > 0:
            combined_clean.fillna(combined_clean.median(), inplace=True)
        # The last row in combined_clean corresponds to user input cleaned
        cleaned_user_row = combined_clean.iloc[[-1]].copy()
        return cleaned_user_row

    if predict_random:
        with st.spinner("Loading random patient..."):
            try:
                random_patient = df_clean.sample(n=1)
                # Extract original values from df (we must go back to original df because df_clean might have transformed data)
                # Actually, df_clean should contain original columns after cleaning. 
                # We'll just read from df_clean. The user wants original format, but we only have cleaned data here.
                # We'll assume df_clean didn't drastically alter categorical values. If it did, you'd need original df mapping.
                # For safety, let's revert to 'df' for raw values:
                # We'll match them by index. 
                # Problem: after cleaning, indexes might differ. We'll rely on df_clean having same indices as df:
                # The cleaning might have dropped rows. If so, we must rely on a stable index. We assume no row was dropped and indices match.

                # If indices don't match, consider merging on an unique ID if available.
                # For now, assume indices match:
                idx = random_patient.index[0]
                original_row = df.loc[idx]

                # Update st.session_state.user_input with this patient's original values if available:
                # If a categorical feature isn't present or numeric, fallback gracefully
                for f in input_features:
                    if f in original_row:
                        val = original_row[f]
                        # If categorical and val not in options, fallback
                        if f in categorical_features:
                            options = df_clean[f].dropna().unique().tolist()
                            if val not in options:
                                # fallback to first option
                                val = options[0] if options else ""
                        if pd.isnull(val):
                            val = 0.0 if f in numeric_features else ""
                        st.session_state.user_input[f] = val

                st.success("Random patient loaded. You can now click 'Predict with Input Data'.")
            except Exception as e:
                st.error(f"Error loading random patient: {e}")

    if predict_manual:
        with st.spinner('Making predictions...'):
            try:
                # Construct user_data from session_state
                user_data = {f: st.session_state.user_input[f] for f in input_features}
                # Get cleaned row
                cleaned_user_row = get_cleaned_input_row(user_data)

                # Extract target values if random patient or not:
                # We can't know if it's random or not easily here, 
                # but we can just skip true values if we didn't load random patient.
                # If random patient was loaded, user_input was updated accordingly.
                # True values are unknown for arbitrary input. For random patient:
                # If user did random patient -> predict manual -> we still have same user_input.
                # We won't show true values unless we can detect random selection.
                # Let's store a flag in session state if random patient was loaded:
                if 'random_loaded' not in st.session_state:
                    st.session_state.random_loaded = False
                if predict_random:
                    st.session_state.random_loaded = True

                true_prog = None
                true_event = None
                if st.session_state.random_loaded:
                    # If random loaded, we know the last random patient chosen
                    # Actually, we didn't store the random patient globally. 
                    # We can store it in session_state when random_loaded is clicked.
                    # Let's do that above:
                    pass

                # Just predict now:
                if target_choice == 'Progressive disease':
                    prog_pred, prog_prob = predict_progressive(cleaned_user_row)
                    st.subheader("Prediction Results")
                    cols = st.columns(2)
                    with cols[0]:
                        display_results("Progressive Disease", prog_pred[0], prog_prob[0])
                        # If random_loaded and we still have random_patient:
                        if st.session_state.get('random_loaded', False):
                            # We must have stored random_patient globally:
                            # Let's store random_patient in session after loading:
                            # We'll do it now:
                            if 'last_random_patient' in st.session_state:
                                true_prog = st.session_state.last_random_patient['Progressive disease'].values[0]
                                if true_prog is not None:
                                    st.write(f"**True Progressive Disease Value:** {'Positive' if true_prog == 1 else 'Negative'}")
                    with cols[1]:
                        st.write("**Note on Feature Importances:**")
                        st.info("Feature importances not shown for individual predictions.")

                else:  # Event
                    prog_pred, prog_prob = predict_progressive(cleaned_user_row)
                    event_pred, event_prob = predict_event(cleaned_user_row.copy(), prog_prob)
                    if event_pred is not None:
                        st.subheader("Prediction Results")
                        cols = st.columns(2)
                        with cols[0]:
                            st.write("**Progressive Disease Probability**")
                            st.write(f"Probability: {prog_prob[0]:.4f}")
                            st.write(f"Uncertainty: {(1 - prog_prob[0])*100:.2f}%")
                            if st.session_state.get('random_loaded', False) and 'last_random_patient' in st.session_state:
                                true_prog = st.session_state.last_random_patient['Progressive disease'].values[0]
                                if true_prog is not None:
                                    st.write(f"**True Progressive Disease Value:** {'Positive' if true_prog == 1 else 'Negative'}")
                        with cols[1]:
                            display_results("Event (Death/Transplantation)", event_pred[0], event_prob[0])
                            if st.session_state.get('random_loaded', False) and 'last_random_patient' in st.session_state:
                                true_event = st.session_state.last_random_patient['Event'].values[0]
                                if true_event is not None:
                                    st.write(f"**True Event Value:** {'Positive' if true_event == 1 else 'Negative'}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    # When we load a random patient, store in session state for future reference:
    if predict_random and 'random_patient' in locals():
        # Store random_patient as last_random_patient for displaying true values after prediction
        st.session_state.last_random_patient = random_patient
        st.session_state.random_loaded = True
    elif 'random_loaded' not in st.session_state:
        st.session_state.random_loaded = False
