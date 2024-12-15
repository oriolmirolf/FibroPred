# data_cleaner.py

import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def drop_features(self, features):
        self.df.drop(columns=features, inplace=True)
        print(f"Dropped features: {features}")

    def encode_categorical(self):
        if 'Sex' in self.df.columns:
            self.df['Sex'] = self.df['Sex'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})

        if 'Death' in self.df.columns:
            self.df['Death'] = self.df['Death'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})

        if 'Necessity of transplantation' in self.df.columns:
            self.df['Necessity of transplantation'] = self.df['Necessity of transplantation'].apply(lambda x: 1 if x == 1 else 0)

        # Map binary categorical variables
        binary_cols = [col for col in self.df.select_dtypes(include=['object']).columns if self.df[col].nunique() == 2]
        for col in binary_cols:
            unique_values = self.df[col].unique()
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            self.df[col] = self.df[col].map(mapping)

        # One-hot encode remaining categorical variables
        cat_cols = self.df.select_dtypes(include=['object']).columns
        self.df = pd.get_dummies(self.df, columns=cat_cols, dtype=int)

    def telomeric_illness(self):
        self.df['telomeric affectation'] = self.df['Type of telomeric extrapulmonary affectation'].apply(lambda x: 0 if pd.isna(x) else 1)
        self.df.drop(columns=['Type of telomeric extrapulmonary affectation'], inplace=True)

    def neoplastia(self):
        self.df['Type of neoplasia'] = self.df['Type of neoplasia'].apply(lambda x: 0 if pd.isna(x) else 1)

    def hematological(self):
        for i, row in self.df.iterrows():
            if (row['Hematologic Disease'] == 'No' or pd.isna(row['Hematologic Disease'])) and pd.isna(row['Hematological abnormality before diagnosis']):
                self.df.at[i, 'Hematologic Abnormalities'] = 0
            else:
                self.df.at[i, 'Hematologic Abnormalities'] = 1
        self.df.drop(columns=['Hematologic Disease', 'Hematological abnormality before diagnosis'], inplace=True)

    def liver_abnormality(self):
        for i, _ in self.df.iterrows():
            if (self.df.at[i, 'LDH'] == 1 or self.df.at[i, 'ALT'] == 1 or self.df.at[i, 'AST'] == 1 or 
                self.df.at[i, 'GGT'] == 1 or self.df.at[i, 'Transaminitis'] == 1 or self.df.at[i, 'Cholestasis'] == 1):
                self.df.at[i,'Liver Problem'] = 1
            else:
                self.df.at[i,'Liver Problem'] = 0
        self.df.drop(columns=['Liver abnormality before diagnosis','Liver abnormality','Type of liver abnormality'], inplace=True)

    def liver_disease(self):
        self.df['Liver disease'] = self.df['Liver disease'].apply(lambda x: 0 if pd.isna(x) or x == 'No' else 1)

    def mutations(self):
        self.df['TERT'] = self.df['Mutation Type'].apply(lambda x: 1 if 'TERT' in str(x) else 0)
        self.df['Mutation Type'] = self.df['Mutation Type'].apply(lambda x: 0 if pd.isna(x) else 1)

    def diag_after_biopsy(self):
        self.df['Diagnosis after Biopsy'] = self.df['Diagnosis after Biopsy'].apply(lambda x: 0 if (x == -9 or pd.isna(x)) else x)

    def tobaco(self):
        self.df['Fumador'] = self.df['TOBACCO'].apply(lambda x: 1 if x == 1 else 0)
        self.df['Exfumador'] = self.df['TOBACCO'].apply(lambda x: 1 if x == 2 else 0)
        self.df.drop(columns=['TOBACCO'], inplace=True)

    def biopsia(self):
        self.df['endoscopic cryobiopsy'] = self.df['Biopsy'].apply(lambda x: 1 if x == 1 else 0)
        self.df['surgical biopsy'] = self.df['Biopsy'].apply(lambda x: 1 if x == 2 else 0)
        self.df.drop(columns=['Biopsy'], inplace=True)

    def create_event_variable(self):
        # Create 'Event' variable: 1 if Death or Necessity of transplantation, else 0
        if 'Death' in self.df.columns and 'Necessity of transplantation' in self.df.columns:
            self.df['Event'] = ((self.df['Death'] == 1) | (self.df['Necessity of transplantation'] == 1)).astype(int)
        else:
            print("Warning: 'Death' or 'Necessity of transplantation' columns not found. 'Event' variable not created.")

    def fill_progressive_disease(self):
        if 'Progressive disease' in self.df.columns:
            missing_before = self.df['Progressive disease'].isnull().sum()
            self.df['Progressive disease'].fillna(0, inplace=True)
            missing_after = self.df['Progressive disease'].isnull().sum()
            print(f"Filled {missing_before - missing_after} missing 'Progressive disease' values with 0.")
        else:
            print("Warning: 'Progressive disease' column not found.")

    def handle_missing_values(self):
        # Identify numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'bool', 'category']).columns

        # Impute numeric columns with median
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())

        # Impute categorical columns with mode
        self.df[categorical_cols] = self.df[categorical_cols].fillna(self.df[categorical_cols].mode().iloc[0])

    def clean(self, selected_features, features_to_drop):
        self.drop_features(features_to_drop)
        self.handle_missing_values()
        self.telomeric_illness()
        self.neoplastia()
        self.hematological()
        self.liver_abnormality()
        self.liver_disease()
        self.mutations()
        self.biopsia()
        self.tobaco()
        self.create_event_variable()
        self.diag_after_biopsy()
        self.fill_progressive_disease()
        self.df = self.df[selected_features]
        self.encode_categorical()
        return self.df
