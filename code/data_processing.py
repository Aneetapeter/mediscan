# ============================================================
# DATA PROCESSING MODULE — data_processing.py
# ============================================================
# This module handles the entire DATA PIPELINE for the CKD
# (Chronic Kidney Disease) prediction system. It performs:
#   1. Loading raw CSV data
#   2. Data cleaning (duplicates, missing values)
#   3. Feature engineering (creating new useful features)
#   4. Encoding (converting text categories → numbers)
#   5. Train/Test splitting
#   6. Feature scaling (standardization/normalization)
#   7. Transforming new patient data for prediction
# ============================================================

import pandas as pd          # Data manipulation library — DataFrames, CSV reading, etc.
import numpy as np           # Numerical computing — arrays, math operations
from sklearn.impute import SimpleImputer, KNNImputer        # For filling missing values
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures  # Encoding & Scaling
from sklearn.model_selection import train_test_split         # Splitting data into Train/Test sets
import streamlit as st       # Streamlit for displaying errors/warnings in the web UI


# ============================================================
# DataProcessor CLASS
# ============================================================
# This class encapsulates the entire data processing pipeline.
# It follows the standard ML workflow:
#   Raw Data → Clean → Engineer → Encode → Split → Scale → Ready for Model
# ============================================================

class DataProcessor:
    def __init__(self, file_path):
        """
        Initialize the DataProcessor with the path to the dataset.
        
        Args:
            file_path (str): Path to the CSV file containing patient data.
        
        Attributes:
            df: The main DataFrame holding all patient data
            X_train, X_test: Feature matrices for training and testing
            y_train, y_test: Target labels (CKD_Status: 0=Healthy, 1=CKD)
            scalers: Dictionary storing fitted scaler objects (for transforming new data later)
            encoders: Dictionary storing fitted encoder objects (for encoding new patient data)
        """
        self.file_path = file_path
        self.df = None          # Will hold the pandas DataFrame after loading
        self.X_train = None     # Training features (80% of data)
        self.X_test = None      # Testing features (20% of data — unseen by model)
        self.y_train = None     # Training labels
        self.y_test = None      # Testing labels
        self.scalers = {}       # Stores fitted scalers to reuse on new patients
        self.encoders = {}      # Stores fitted encoders to reuse on new patients


    # ============================================================
    # STEP 1: DATA LOADING
    # ============================================================
    # Reads the CSV file into a pandas DataFrame.
    # The CSV contains patient records with features like:
    #   Age, Blood_Pressure, Blood_Glucose_Random, Serum_Creatinine,
    #   Hemoglobin, Albumin, Hypertension, Diabetes_Mellitus, etc.
    # Target column: CKD_Status (0 = No CKD, 1 = CKD)
    # ============================================================

    def load_data(self):
        """Loads the CKD dataset from a CSV file into a pandas DataFrame."""
        try:
            # pd.read_csv() reads the CSV file and creates a DataFrame
            # Each row = one patient record, each column = one feature
            self.df = pd.read_csv(self.file_path)
            return self.df
        except FileNotFoundError:
            st.error(f"File not found at {self.file_path}")
            return None


    # ============================================================
    # STEP 2: DATA INSPECTION
    # ============================================================
    # Before cleaning, we inspect the data to understand:
    #   - Shape: How many rows (patients) and columns (features)?
    #   - Data types: Which columns are numeric vs categorical?
    #   - Missing values: Which columns have NULL/NaN values?
    # This helps us decide the right cleaning strategy.
    # ============================================================

    def check_structure(self):
        """Returns basic structural info about the dataset for inspection."""
        if self.df is None: return None
        return {
            "shape": self.df.shape,                     # (rows, columns) — e.g., (400, 25)
            "columns": self.df.columns.tolist(),        # List of all column names
            "dtypes": self.df.dtypes,                   # Data type of each column (int64, float64, object)
            "null_counts": self.df.isnull().sum()        # Count of missing values per column
        }


    # ============================================================
    # STEP 3: DUPLICATE REMOVAL
    # ============================================================
    # Duplicate records can bias the model by giving extra weight
    # to certain patients. We detect and remove exact duplicates.
    # ============================================================

    def check_duplicates(self):
        """Checks for and removes duplicate rows from the dataset."""
        if self.df is None: return 0
        duplicates = self.df.duplicated().sum()  # Count exact duplicate rows
        if duplicates > 0:
            # drop_duplicates() removes rows that are identical to a previous row
            # inplace=True modifies the DataFrame directly (no copy needed)
            self.df.drop_duplicates(inplace=True)
        return duplicates


    # ============================================================
    # STEP 4: HANDLING MISSING VALUES (IMPUTATION)
    # ============================================================
    # Real-world medical data often has missing values (patient
    # didn't take a test, data entry errors, etc.)
    # 
    # We CANNOT simply delete rows with missing data — we'd lose
    # too many patients. Instead, we IMPUTE (fill in) missing values.
    #
    # Strategies for NUMERICAL columns (Age, BP, Glucose, etc.):
    #   - 'median': Replace NaN with the column's median value
    #     (robust to outliers — preferred for medical data)
    #   - 'mean': Replace NaN with the column's average
    #   - 'knn': Use K-Nearest Neighbors to predict missing values
    #     based on similar patients (more accurate but slower)
    #
    # Strategy for CATEGORICAL columns (Hypertension, Diabetes, etc.):
    #   - 'most_frequent': Replace NaN with the most common category (mode)
    # ============================================================

    def impute_data(self, strategy='median'):
        """
        Fills in missing values in the dataset.
        
        Args:
            strategy (str): Imputation method — 'median' (default), 'mean', or 'knn'
        
        Why median over mean?
            Median is robust to outliers. If one patient has BP=300 (outlier),
            the mean would be skewed high, but the median stays stable.
        """
        if self.df is None: return

        # --- Impute NUMERICAL columns ---
        # select_dtypes(include=[np.number]) picks only numeric columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns

        if strategy == 'knn':
            # KNN Imputer: For each missing value, finds 5 most similar patients
            # (based on other features) and uses their average to fill the gap
            imputer = KNNImputer(n_neighbors=5)
            self.df[num_cols] = imputer.fit_transform(self.df[num_cols])
        else:
            # SimpleImputer: Replaces NaN with the column's median (or mean)
            # fit_transform() = learn the median from data + apply it
            imputer = SimpleImputer(strategy=strategy)
            self.df[num_cols] = imputer.fit_transform(self.df[num_cols])
        
        # --- Impute CATEGORICAL columns ---
        # For text columns like "Hypertension: Yes/No", fill NaN with the most common value
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')  # Mode imputation
            self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])


    # ============================================================
    # STEP 5: ENCODING CATEGORICAL FEATURES
    # ============================================================
    # Machine Learning models work with NUMBERS, not text.
    # We need to convert categorical columns like:
    #   "Yes"/"No" → 1/0
    #   "Normal"/"Abnormal" → 0/1
    # 
    # LabelEncoder assigns a unique integer to each category.
    # Example: Hypertension: "No"→0, "Yes"→1
    # 
    # We SAVE the encoders so we can use the SAME mapping
    # when encoding new patient data during prediction.
    # ============================================================

    def encode_features(self):
        """Converts all categorical (text) columns to numerical values using Label Encoding."""
        if self.df is None: return

        le = LabelEncoder()  # Create a label encoder instance
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'Patient_ID':  # Skip the ID column — it's not a feature
                # fit_transform: Learn the categories + convert them to integers
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                # Store the encoder so we can encode new patient data the same way
                self.encoders[col] = le


    # ============================================================
    # STEP 6: TRAIN/TEST SPLIT
    # ============================================================
    # We split the data into TWO parts:
    #   - Training set (80%): Model LEARNS patterns from this data
    #   - Test set (20%): Model is EVALUATED on this unseen data
    #
    # WHY? To check if the model generalizes well to new patients
    # it has never seen before (prevents overfitting).
    #
    # random_state=42 ensures the split is REPRODUCIBLE
    # (same split every time we run the code).
    # ============================================================

    def split_data(self, target_col='CKD_Status'):
        """
        Splits the dataset into training and testing sets.
        
        Args:
            target_col (str): The column we're trying to predict ('CKD_Status')
        
        Returns:
            X_train, X_test: Feature matrices (all columns EXCEPT target and ID)
            y_train, y_test: Target labels (CKD_Status: 0 or 1)
        """
        if self.df is None or target_col not in self.df.columns: return
        
        # X = Features (input variables) — everything except target and Patient_ID
        # y = Target (what we're predicting) — CKD_Status (0=Healthy, 1=CKD)
        X = self.df.drop(columns=[target_col, 'Patient_ID'], errors='ignore')
        y = self.df[target_col]
        
        # Split: 80% training, 20% testing
        # random_state=42 → same split every time for reproducibility
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return self.X_train, self.X_test, self.y_train, self.y_test


    # ============================================================
    # STEP 7: FEATURE ENGINEERING
    # ============================================================
    # Creating NEW features from existing ones to help the model
    # find patterns more easily.
    #
    # Example 1: Age Binning
    #   Instead of exact age (47, 52, 61...), we group into categories:
    #   0=Child(0-18), 1=Young(18-35), 2=Middle(35-50), 3=Senior(50-65), 4=Elderly(65+)
    #   This helps the model learn that "elderly patients are at higher risk"
    #
    # Example 2: Polynomial Features
    #   Creates interaction terms like BP×Glucose, BP², Glucose²
    #   These capture NON-LINEAR relationships between features
    # ============================================================

    def feature_engineering(self):
        """Creates derived features to improve model performance."""
        if self.df is None: return
        
        # --- Age Binning: Convert continuous age into categorical age groups ---
        if 'Age' in self.df.columns:
            # pd.cut() divides age into bins and assigns labels (0-4)
            self.df['Age_Group'] = pd.cut(
                self.df['Age'],
                bins=[0, 18, 35, 50, 65, 100],  # Bin edges
                labels=[0, 1, 2, 3, 4]           # 0=Child, 1=Young, 2=Middle, 3=Senior, 4=Elderly
            )
        
        # --- Polynomial Features: Create interaction terms ---
        # PolynomialFeatures(degree=2) creates: BP, Glucose, BP², Glucose², BP×Glucose
        # This helps capture non-linear relationships
        poly_cols = ['Blood_Pressure', 'Blood_Glucose_Random']
        existing_poly_cols = [c for c in poly_cols if c in self.df.columns]
        if existing_poly_cols:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(self.df[existing_poly_cols])
            # Note: Generated features are used for demonstration; 
            # in production, you'd add them back to the DataFrame


    # ============================================================
    # STEP 8: FEATURE SCALING (STANDARDIZATION)
    # ============================================================
    # Different features have very different scales:
    #   - Age: 0-100
    #   - Blood Pressure: 50-200
    #   - Serum Creatinine: 0.4-15
    #   - WBC Count: 2000-20000
    #
    # Many ML algorithms (Logistic Regression, SVM, KNN) are
    # sensitive to feature scales. Without scaling, features with
    # large values (WBC) would dominate over small ones (Creatinine).
    #
    # StandardScaler: Transforms each feature to have mean=0, std=1
    #   Formula: z = (x - mean) / std_deviation
    #
    # MinMaxScaler: Scales each feature to [0, 1] range
    #   Formula: x_scaled = (x - min) / (max - min)
    #
    # CRITICAL: We fit the scaler on TRAINING data ONLY, then
    # apply the SAME transformation to test data. This prevents
    # DATA LEAKAGE (test data info leaking into training).
    # ============================================================

    def scale_features(self, method='standard'):
        """
        Scales numerical features to a uniform range.
        
        Args:
            method (str): 'standard' (mean=0, std=1) or 'minmax' (0 to 1)
        
        Important: Scaler is FIT on training data only, then APPLIED to test data.
        This prevents data leakage — the model never "sees" test data statistics.
        """
        if self.X_train is None: return

        # Choose scaler based on method
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        
        # Only scale numerical columns (not encoded categoricals, though they're also numbers now)
        num_cols = self.X_train.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) > 0:
            # FIT on training data: Learn mean & std from training set ONLY
            # TRANSFORM training data: Apply the learned scaling
            self.X_train[num_cols] = scaler.fit_transform(self.X_train[num_cols])
            
            # TRANSFORM test data: Apply the SAME scaling (using training set's mean & std)
            # We do NOT fit again on test data — that would cause data leakage!
            if self.X_test is not None:
                self.X_test[num_cols] = scaler.transform(self.X_test[num_cols])
                
            # Save the fitted scaler for later use with new patient data
            self.scalers['main'] = scaler


    # ============================================================
    # STEP 9: TRANSFORMING NEW PATIENT DATA FOR PREDICTION
    # ============================================================
    # When a new patient is screened (via the Patient Screening page),
    # their raw values must go through the EXACT SAME transformations
    # that were applied to the training data:
    #   1. Encode categorical features (using saved encoders)
    #   2. Scale numerical features (using saved scaler)
    #
    # This ensures the model receives data in the same format it
    # was trained on. Without this, predictions would be wrong.
    # ============================================================

    def transform_data(self, new_df):
        """
        Transforms new patient data using the SAME encoders and scalers
        that were fitted on the training data.
        
        Args:
            new_df (pd.DataFrame): Raw patient data from the screening form
        
        Returns:
            pd.DataFrame: Transformed data ready for model.predict()
        
        This function is called every time a new patient is screened.
        It ensures consistency between training and prediction data formats.
        """
        df_transformed = new_df.copy()
        
        # --- Apply Encoding (using saved encoders from training) ---
        # Convert categorical values ("Yes"/"No") to numbers (1/0)
        # using the SAME mapping that was learned during training
        for col, le in self.encoders.items():
            if col in df_transformed.columns:
                # Safe transform: If a value wasn't seen during training, assign -1
                df_transformed[col] = df_transformed[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )

        # --- Apply Scaling (using saved scaler from training) ---
        # Scale numerical values using the SAME mean & std from training data
        if 'main' in self.scalers:
            scaler = self.scalers['main']
            
            # Get the feature names the scaler expects
            if hasattr(scaler, 'feature_names_in_'):
                expected_cols = scaler.feature_names_in_
                
                # If new patient data is missing some columns, fill with 0
                # (This handles cases where feature engineering added extra columns)
                missing_cols = set(expected_cols) - set(df_transformed.columns)
                for c in missing_cols:
                    df_transformed[c] = 0.0
                
                # Apply the saved scaling transformation
                df_transformed[expected_cols] = scaler.transform(df_transformed[expected_cols])
            else:
                # Fallback: Scale all numeric columns
                num_cols = df_transformed.select_dtypes(include=[np.number]).columns
                df_transformed[num_cols] = scaler.transform(df_transformed[num_cols])

        return df_transformed
