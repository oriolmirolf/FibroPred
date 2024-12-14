# data_cleaner.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.preprocessor = None  # Will hold the preprocessing pipeline
    
    # Existing data cleaning methods
    def handle_duplicates(self):
        if 'COD NUMBER' in self.df.columns:
            self.df.sort_values(by=['COD NUMBER', 'Age at diagnosis'], inplace=True)
            self.df.drop_duplicates(subset='COD NUMBER', keep='first', inplace=True)
            self.df.reset_index(drop=True, inplace=True)
        else:
            print("Warning: 'COD NUMBER' column not found for deduplication.")
    
    def drop_high_missing(self, threshold=0.4):
        missing = self.df.isnull().mean()
        cols_to_drop = missing[missing > threshold].index
        self.df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped columns with more than {threshold*100}% missing values: {list(cols_to_drop)}")
    
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
        
        cat_cols = self.df.select_dtypes(include=['object']).columns
        binary_cols = [col for col in cat_cols if self.df[col].nunique() == 2]
        
        for col in binary_cols:
            unique_values = self.df[col].unique()
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            self.df[col] = self.df[col].map(mapping)
        
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
        for i, _ in self.df.iterrows():
            if pd.isna(self.df.at[i, 'Liver disease']) or self.df.at[i, 'Liver disease'] == "No":
                self.df.at[i, 'Liver disease'] = 0
            else:
                self.df.at[i, 'Liver disease'] = 1

    def mutations(self):
        for i, _ in self.df.iterrows():
            if 'TERT' in str(self.df.at[i, 'Mutation Type']):
                self.df.at[i, 'TERT'] = 1
            else:
                self.df.at[i, 'TERT'] = 0
        self.df['Mutation Type'] = self.df['Mutation Type'].apply(lambda x: 0 if pd.isna(x) else 1)

    def diag_after_biopsy(self):
        self.df['Diagnosis after Biopsy'] = self.df['Diagnosis after Biopsy'].apply(lambda x: 0 if (x == -9 or pd.isna(x))  else x)

    def tobaco(self):
        self.df['Fumador'] = self.df['TOBACCO'].apply(lambda x: 1 if x == 1 else 0)
        self.df['Exfumador'] = self.df['TOBACCO'].apply(lambda x: 1 if x == 2 else 0)
        self.df.drop(columns=['TOBACCO'], inplace=True)

    def biopsia(self):
        self.df['endoscopic cryobiopsy'] = self.df['Biopsy'].apply(lambda x: 1 if x == 1 else 0)
        self.df['surgical biopsy'] = self.df['Biopsy'].apply(lambda x: 1 if x == 2 else 0)
        self.df.drop(columns=['Biopsy'], inplace=True)
    
    def remove_duplicate(self):
        self.df = pd.concat([
            self.df[self.df['COD NUMBER'] != 17822919], 
            self.df[self.df['COD NUMBER'] == 17822919].loc[[self.df[self.df['COD NUMBER'] == 17822919]['Age at diagnosis'].idxmax()]]
        ], ignore_index=True)
        
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
    
    def clean(self, selected_features, features_to_drop):
        self.drop_features(features_to_drop)

        # Manual manipulation
        self.telomeric_illness()
        self.neoplastia()
        self.hematological()
        self.liver_abnormality()
        self.liver_disease()
        self.mutations()
        self.biopsia()
        self.tobaco()
        self.remove_duplicate()
        self.create_event_variable()

        # Handling Diagnosis after biopsy
        self.diag_after_biopsy()
        
        # Fill missing "Progressive disease" with 0
        self.fill_progressive_disease()

        # Feature selection
        self.df = self.df[selected_features]

        # Duplicates
        fake_duplicates = [12908554, 13022784, 10553362,11425345]
        # self.handle_duplicates(fake_duplicates)
        # self.drop_high_missing()
        # self.impute_missing()
        self.encode_categorical()
        return self.df
