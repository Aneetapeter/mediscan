# ============================================================
# ML PREDICTION MODULE — prediction.py
# ============================================================
# This module handles the MACHINE LEARNING side of MediScan:
#   1. Model Training — Multiple algorithms (Logistic Regression,
#      Random Forest, SVM, KNN, XGBoost, etc.)
#   2. Model Evaluation — Accuracy, Precision, Recall, F1-Score,
#      ROC-AUC, Confusion Matrix, Cross-Validation
#   3. Feature Selection — RFE (Recursive Feature Elimination)
#   4. Explainability — SHAP values for model interpretability
#   5. Visualizations — ROC curves, feature importance, learning
#      curves, confusion matrices, etc.
#
# The core idea: Train a classifier on historical patient data,
# then use it to predict CKD risk for new patients.
# ============================================================

import pandas as pd          # Data manipulation
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # Plotting library for static charts
import seaborn as sns         # Statistical visualization (heatmaps, etc.)

# --- scikit-learn (sklearn) Imports ---
# sklearn is the main ML library in Python. Each module serves a purpose:

from sklearn.model_selection import train_test_split  # Split data into train/test
from sklearn.preprocessing import StandardScaler       # Feature scaling

# --- ML ALGORITHMS ---
# Classification: Predicts a CATEGORY (CKD: Yes/No)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
# Logistic Regression: Linear classifier — good baseline, fast, interpretable
# Linear Regression: Predicts continuous values (used for regression tasks)
# Ridge/Lasso/ElasticNet: Regularized regression models that prevent overfitting

from sklearn.tree import DecisionTreeClassifier, plot_tree
# Decision Tree: Makes decisions by asking yes/no questions about features
# Like a flowchart: "Is Creatinine > 1.2?" → Yes → "Is BP > 140?" → Yes → High Risk

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# Random Forest (BAGGING): Builds 100+ decision trees and takes a VOTE
#   - Each tree sees a random subset of data → reduces overfitting
#   - Most accurate general-purpose classifier
# AdaBoost (BOOSTING): Trains trees sequentially, each fixing the previous one's mistakes
# Gradient Boosting: Like AdaBoost but uses gradient descent to minimize errors

from sklearn.svm import SVC
# SVM (Support Vector Machine): Finds the best boundary (hyperplane) between classes
# Good for high-dimensional data, but slow on large datasets

from sklearn.naive_bayes import GaussianNB
# Naive Bayes: Fast probabilistic classifier based on Bayes' theorem
# Assumes features are independent (naive assumption)

from sklearn.neighbors import KNeighborsClassifier
# KNN (K-Nearest Neighbors): Classifies based on the K closest data points
# "You are the average of the 5 people around you"

from sklearn.feature_selection import RFE
# RFE (Recursive Feature Elimination): Selects the most important features
# by repeatedly training a model and removing the least important feature

# --- EVALUATION METRICS ---
from sklearn.metrics import (
    accuracy_score,           # % of correct predictions overall
    classification_report,    # Precision, Recall, F1 per class
    confusion_matrix,         # True/False Positives/Negatives matrix
    mean_squared_error,       # Average squared error (for regression)
    r2_score,                 # R² — how well model explains variance
    mean_absolute_error,      # Average absolute error (for regression)
    precision_score,          # Of all predicted CKD, how many actually have CKD?
    recall_score,             # Of all actual CKD patients, how many did we catch?
    f1_score,                 # Harmonic mean of Precision and Recall
    roc_curve,                # ROC curve data points
    roc_auc_score             # Area Under ROC Curve (overall discriminative ability)
)

# --- Optional Libraries (graceful fallback if not installed) ---

# SHAP: SHapley Additive exPlanations — explains WHY a model made a prediction
# Shows which features pushed the prediction toward CKD or Healthy
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
# XGBoost: eXtreme Gradient Boosting — highly optimized gradient boosting library
# Often wins ML competitions due to speed and accuracy
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
try:
    from xgboost import XGBClassifier
except ImportError:
    pass

# SMOTE: Synthetic Minority Over-sampling Technique
# When CKD patients are rare (imbalanced data), SMOTE creates synthetic
# CKD samples so the model doesn't just predict "Healthy" for everyone
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

import streamlit as st  # For displaying warnings/errors in the web UI


# ============================================================
# Predictor CLASS — Core ML Engine
# ============================================================
# This class handles model training, evaluation, prediction,
# and all ML visualizations. It's the "brain" of MediScan.
# ============================================================

class Predictor:
    def __init__(self):
        """
        Initialize the Predictor.
        
        Attributes:
            model: The trained sklearn model object
            selected_features: List of features chosen by RFE
            model_type: 'classification' (CKD risk) or 'regression'
        """
        self.model = None
        self.selected_features = None
        self.model_type = None  # 'classification' or 'regression'


    # ============================================================
    # MODEL TRAINING
    # ============================================================
    # This is the core ML step. We:
    #   1. Select an algorithm (e.g., Random Forest)
    #   2. Optionally apply SMOTE for class balancing
    #   3. Call model.fit(X_train, y_train) to LEARN patterns
    #   4. The model learns a mapping: features → CKD risk
    #
    # model.fit() is where the actual LEARNING happens:
    #   - Logistic Regression: Finds optimal weight for each feature
    #   - Random Forest: Builds 100 decision trees on random data subsets
    #   - SVM: Finds the best separating hyperplane
    #   - KNN: Simply memorizes all training data points
    # ============================================================

    def train_model(self, X_train, y_train, algorithm='Logistic Regression', params=None, use_smote=False):
        """
        Trains a machine learning model on the training data.
        
        Args:
            X_train (DataFrame): Training features (scaled patient data)
            y_train (Series): Training labels (0=Healthy, 1=CKD)
            algorithm (str): Which ML algorithm to use
            params (dict): Hyperparameters for fine-tuning the model
            use_smote (bool): Whether to balance classes using SMOTE
        
        Returns:
            The trained model object, or None if training failed
        """
        
        # Default params if None
        if params is None: params = {}

        # --- Handle SMOTE (Class Balancing) ---
        # Problem: If 250 patients are healthy and only 150 have CKD,
        # the model might just predict "Healthy" for everyone (75% accuracy!)
        # SMOTE fixes this by creating synthetic CKD patient samples.
        if use_smote and SMOTE_AVAILABLE:
            try:
                # IMPORTANT: SMOTE should ONLY be applied to TRAINING data
                # Never apply to test data — that would be data leakage!
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                st.warning(f"SMOTE failed: {e}. Proceeding with original data.")

        # ============================================
        # REGRESSION MODELS (Predict continuous values)
        # ============================================
        # These predict a NUMBER, not a category
        # Used when target is continuous (e.g., kidney function score)

        if algorithm == 'Linear Regression':
            # Simplest regression: y = w1*x1 + w2*x2 + ... + bias
            self.model = LinearRegression()
            self.model_type = 'regression'
        
        elif algorithm in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
            # Regularized regression: Adds a penalty to prevent overfitting
            # Ridge (L2): Shrinks coefficients toward zero
            # Lasso (L1): Can set coefficients exactly to zero (feature selection!)
            # ElasticNet: Combination of Ridge + Lasso
            alpha = params.get('alpha', 1.0)  # Regularization strength
            if algorithm == 'Ridge Regression': self.model = Ridge(alpha=alpha)
            elif algorithm == 'Lasso Regression': self.model = Lasso(alpha=alpha)
            elif algorithm == 'ElasticNet': self.model = ElasticNet(alpha=alpha, l1_ratio=params.get('l1_ratio', 0.5))
            self.model_type = 'regression'

        # ============================================
        # CLASSIFICATION MODELS (Predict categories)
        # ============================================
        # These predict a CLASS: CKD (1) or Healthy (0)
        # This is our primary task for CKD risk assessment

        elif algorithm == 'Logistic Regression':
            # Despite the name, this is a CLASSIFIER (not regression!)
            # It predicts the PROBABILITY of CKD using a sigmoid function
            # P(CKD) = 1 / (1 + e^-(w1*x1 + w2*x2 + ... + bias))
            # If P(CKD) > 0.5 → predict CKD, else → predict Healthy
            # class_weight='balanced': Automatically adjusts for imbalanced classes
            # max_iter=1000: Allow enough iterations for convergence
            self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.model_type = 'classification'
            
        elif algorithm == 'Decision Tree':
            # Builds a tree of if-else rules:
            # "Is Creatinine > 1.2? → Yes → Is Age > 60? → Yes → HIGH RISK"
            # criterion='gini': Measures node impurity (how mixed the classes are)
            # max_depth: Limits tree depth to prevent overfitting
            self.model = DecisionTreeClassifier(
                criterion=params.get('criterion', 'gini'), 
                max_depth=params.get('max_depth', None)
            )
            self.model_type = 'classification'

        elif algorithm == 'Random Forest':
            # ENSEMBLE METHOD (Bagging): Combines 100 decision trees
            # Each tree is trained on a random subset of data & features
            # Final prediction = MAJORITY VOTE of all trees
            # This reduces overfitting compared to a single Decision Tree
            # n_estimators: Number of trees in the forest
            # n_jobs=-1: Use all CPU cores for parallel training (speed!)
            # class_weight='balanced': Adjusts for imbalanced CKD/Healthy
            self.model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100), 
                max_depth=params.get('max_depth', None),
                class_weight='balanced',
                n_jobs=-1
            )
            self.model_type = 'classification'

        elif algorithm == 'AdaBoost':
            # ENSEMBLE METHOD (Boosting): Trains trees SEQUENTIALLY
            # Each new tree focuses on the MISTAKES of the previous one
            # Assigns higher weights to misclassified samples
            self.model = AdaBoostClassifier(n_estimators=params.get('n_estimators', 50))
            self.model_type = 'classification'

        elif algorithm == 'Gradient Boosting':
            # ENSEMBLE METHOD (Boosting): Like AdaBoost, but uses GRADIENT DESCENT
            # to minimize the loss function. Generally more accurate.
            # learning_rate: How much each tree contributes (smaller = more trees needed, but more robust)
            self.model = GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1)
            )
            self.model_type = 'classification'

        elif algorithm == 'XGBoost':
            # eXtreme Gradient Boosting: Optimized version of Gradient Boosting
            # Uses regularization, parallel processing, and handling of missing values
            # Often the top performer in ML competitions
            if XGB_AVAILABLE:
                self.model = XGBClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    use_label_encoder=False,
                    eval_metric='logloss'  # Binary cross-entropy loss
                )
                self.model_type = 'classification'
            else:
                st.error("XGBoost is not installed.")
                return None

        elif algorithm == 'SVM':
            # Support Vector Machine: Finds the optimal HYPERPLANE that
            # separates CKD and Healthy patients with the maximum MARGIN
            # kernel='rbf': Handles non-linear boundaries using the Gaussian kernel
            # probability=True: Enables probability estimates (needed for ROC curves)
            # OPTIMIZATION: SVM is O(n²) — very slow on large datasets
            # So we subsample to 2000 rows for practical training speed
            if len(X_train) > 2000:
                try:
                    indices = np.random.choice(len(X_train), 2000, replace=False)
                    X_train_svc = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
                    y_train_svc = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
                except:
                    X_train_svc, y_train_svc = X_train, y_train
            else:
                X_train_svc, y_train_svc = X_train, y_train

            self.model = SVC(
                kernel=params.get('kernel', 'rbf'), 
                C=params.get('C', 1.0),       # Regularization — higher C = less regularization
                probability=True,              # Required for predict_proba() and ROC/AUC
                class_weight='balanced'
            )
            
            # Train on potentially subsampled data (SVM handles its own fit)
            self.model.fit(X_train_svc, y_train_svc)
            self.model_type = 'classification'
            return self.model  # Return early as we fit manually above

        elif algorithm == 'Naive Bayes':
            # Based on Bayes' Theorem: P(CKD|features) ∝ P(features|CKD) × P(CKD)
            # Assumes all features are INDEPENDENT (naive assumption)
            # Very fast, works well with small datasets
            self.model = GaussianNB()
            self.model_type = 'classification'

        elif algorithm == 'KNN':
            # K-Nearest Neighbors: "Tell me who your neighbors are, I'll tell you who you are"
            # For a new patient, finds the K most similar patients in training data
            # and takes a majority vote of their labels
            # n_neighbors: How many neighbors to consider (default: 5)
            self.model = KNeighborsClassifier(n_neighbors=params.get('n_neighbors', 5))
            self.model_type = 'classification'

        else:
            st.error(f"Algorithm {algorithm} not supported.")
            return None

        # ============================================
        # ACTUAL MODEL TRAINING — model.fit()
        # ============================================
        # This is THE most important line in ML!
        # model.fit(X, y) tells the algorithm to LEARN patterns from the data.
        #   - X_train: Input features (what the model sees)
        #   - y_train: Correct answers (what the model should predict)
        #
        # After fit(), the model has learned:
        #   - Logistic Regression: Optimal weights for each feature
        #   - Random Forest: 100 decision trees with splits
        #   - KNN: Stored all training data points
        # ============================================
        try:
            if use_smote:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            # Store training data for cross-validation later
            self.X_train = X_train
            self.y_train = y_train

            # ★★★ THE CORE ML STEP ★★★
            # model.fit() = LEARN from data
            self.model.fit(X_train, y_train)
            
            # Calculate Training Accuracy (how well model fits training data)
            # High training accuracy but low test accuracy = OVERFITTING
            if self.model_type == 'classification':
                self.train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
            else:
                self.train_accuracy = self.model.score(X_train, y_train)  # R² for regression

        except Exception as e:
            st.error(f"Model training failed: {e}")
            self.model = None
            return None
            
        return self.model



    # ============================================================
    # MODEL EVALUATION
    # ============================================================
    # After training, we evaluate the model on TEST data (data it
    # has never seen before) to check if it generalizes well.
    #
    # Key Classification Metrics:
    # ─────────────────────────────────────────────────────
    # Accuracy:  Correct Predictions / Total Predictions
    #            (Can be misleading with imbalanced data!)
    #
    # Precision: Of all patients we PREDICTED as CKD,
    #            how many actually have CKD?
    #            (Low precision = many false alarms)
    #
    # Recall:    Of all patients who ACTUALLY have CKD,
    #            how many did we correctly identify?
    #            (Low recall = we're MISSING real CKD cases!)
    #            ★ In medical diagnosis, Recall is CRITICAL ★
    #
    # F1-Score:  Harmonic mean of Precision and Recall
    #            (Balances both metrics)
    #
    # ROC-AUC:   Area Under the ROC Curve (0.5 = random, 1.0 = perfect)
    #            Measures overall ability to distinguish CKD from Healthy
    #
    # Confusion Matrix:
    #            [True Negatives   | False Positives]
    #            [False Negatives  | True Positives ]
    #            - True Pos: Correctly predicted CKD
    #            - False Neg: Missed CKD (DANGEROUS in medical context!)
    # ─────────────────────────────────────────────────────
    # ============================================================

    def evaluate_model(self, X_test, y_test, skip_cv=False):
        """
        Evaluates the trained model on unseen test data.
        
        Args:
            X_test (DataFrame): Test features (20% of data, never seen during training)
            y_test (Series): True labels for test data
            skip_cv (bool): Skip cross-validation for faster comparison
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        if not self.model: return {}
        
        # model.predict() = Use the trained model to predict labels for test data
        y_pred = self.model.predict(X_test)
        metrics_dict = {}
        
        # Include training accuracy for overfitting detection
        if hasattr(self, 'train_accuracy'):
            metrics_dict['train_accuracy'] = self.train_accuracy
        
        if self.model_type == 'classification':
            # --- Core Classification Metrics ---
            metrics_dict['accuracy'] = accuracy_score(y_test, y_pred)
            metrics_dict['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics_dict['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics_dict['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics_dict['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            metrics_dict['report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # --- Cross-Validation ---
            # Instead of evaluating on just one test split, we evaluate on 5 different splits
            # This gives a more robust estimate of model performance
            # cv=5 means 5-fold cross-validation
            if not skip_cv:
                try:
                    cv_scores = cross_val_score(
                        self.model,
                        np.vstack((self.X_train, X_test)),  # Use all data
                        np.hstack((self.y_train, y_test)),
                        cv=5,
                        scoring='recall'  # Optimize for Recall (catching CKD cases)
                    )
                    metrics_dict['cv_recall_mean'] = cv_scores.mean()
                    metrics_dict['cv_recall_std'] = cv_scores.std()
                except:
                     metrics_dict['cv_recall_mean'] = 0.0
                     metrics_dict['cv_recall_std'] = 0.0
            else:
                metrics_dict['cv_recall_mean'] = 0.0
                metrics_dict['cv_recall_std'] = 0.0

            # --- ROC Curve & AUC ---
            # ROC (Receiver Operating Characteristic) plots True Positive Rate vs False Positive Rate
            # AUC (Area Under Curve) summarizes ROC into a single number
            # AUC = 1.0 means perfect classifier, AUC = 0.5 means random guessing
            if hasattr(self.model, "predict_proba"):
                try:
                    if len(np.unique(y_test)) == 2:  # Binary classification only
                        # predict_proba returns probability for each class [P(Healthy), P(CKD)]
                        # We take [:, 1] = probability of CKD
                        y_prob = self.model.predict_proba(X_test)[:, 1]
                        metrics_dict['roc_auc'] = roc_auc_score(y_test, y_prob)
                        
                        # ROC curve data points for plotting
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        metrics_dict['roc_curve'] = (fpr, tpr)
                        
                        # Precision-Recall Curve
                        from sklearn.metrics import precision_recall_curve, average_precision_score
                        precision, recall, _ = precision_recall_curve(y_test, y_prob)
                        avg_precision = average_precision_score(y_test, y_prob)
                        metrics_dict['pr_curve'] = (precision, recall, avg_precision)
                        metrics_dict['y_prob'] = y_prob
                    else:
                        metrics_dict['roc_auc'] = "N/A (Multiclass)"
                except Exception as e:
                    metrics_dict['roc_auc'] = 0.0
            else:
                metrics_dict['roc_auc'] = 0.0

        elif self.model_type == 'regression':
            # --- Regression Metrics ---
            # MSE: Average of squared errors (penalizes large errors more)
            # RMSE: Square root of MSE (same scale as target variable)
            # MAE: Average of absolute errors (more intuitive)
            # R²: How much variance in the target is explained by the model (1.0 = perfect)
            metrics_dict['mse'] = mean_squared_error(y_test, y_pred)
            metrics_dict['rmse'] = np.sqrt(metrics_dict['mse'])
            metrics_dict['mae'] = mean_absolute_error(y_test, y_pred)
            metrics_dict['r2'] = r2_score(y_test, y_pred)
            metrics_dict['y_pred'] = y_pred
            
        return metrics_dict


    # ============================================================
    # VISUALIZATION: Precision-Recall Curve
    # ============================================================
    # Shows the trade-off between Precision and Recall at different
    # classification thresholds.
    # - High Precision + Low Recall = Few false alarms, but missing cases
    # - Low Precision + High Recall = Catching most cases, but more false alarms
    # AP (Average Precision) summarizes the curve into one number.
    # ============================================================

    def plot_precision_recall_curve(self, precision, recall, avg_precision):
        """Plots the Precision-Recall tradeoff curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='purple', lw=2, label=f'AP = {avg_precision:.2f}')
        ax.set_title('Precision-Recall Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


    # ============================================================
    # VISUALIZATION: Probability Distribution
    # ============================================================
    # Shows how the model's predicted probabilities are distributed
    # for CKD vs Healthy patients.
    # Ideal: Two separate peaks — Healthy near 0, CKD near 1
    # Overlapping = model is confused and can't distinguish well
    # ============================================================

    def plot_probability_distribution(self, y_test, y_prob):
        """Plots the distribution of predicted CKD probabilities for each class."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(x=y_prob, hue=y_test, kde=True, element="step",
                     stat="density", common_norm=False, ax=ax, palette=['green', 'red'])
        ax.set_title("Probability Distribution (Risk vs Healthy)")
        ax.set_xlabel("Predicted Probability of Risk (CKD)")
        ax.legend(labels=['High Risk', 'Healthy'])
        fig.tight_layout()
        return fig


    # ============================================================
    # VISUALIZATION: Feature Coefficients (Linear Models)
    # ============================================================
    # For Logistic Regression, the learned coefficients tell us:
    #   - Positive coefficient → increases CKD risk (RED)
    #   - Negative coefficient → decreases CKD risk / protective (BLUE)
    #   - Larger absolute value → stronger influence
    # ============================================================

    def plot_coefficients(self, feature_names):
        """Plots learned coefficients showing which features increase/decrease risk."""
        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_.flatten()
            
            coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
            coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, len(feature_names)*0.4 + 2))
            # Red bars = increase risk, Blue bars = decrease risk (protective)
            colors = ['red' if c > 0 else 'blue' for c in coef_df['Coefficient']]
            sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette=colors, ax=ax)
            ax.set_title("Feature Coefficients (Red=Risk, Blue=Protective)")
            ax.axvline(x=0, color='black', linestyle='--')
            fig.tight_layout()
            return fig
        return None


    # ============================================================
    # SHAP: Model Explainability (GLOBAL - Summary Plot)
    # ============================================================
    # SHAP (SHapley Additive exPlanations) is the gold standard
    # for ML model interpretability.
    #
    # SHAP values explain each feature's contribution to a prediction:
    #   - Each dot = one patient
    #   - X-axis = impact on prediction (left = lower risk, right = higher risk)
    #   - Color = feature value (red = high value, blue = low value)
    #
    # Example: Red dots on the right for Serum_Creatinine means
    # "high creatinine values push the prediction toward CKD"
    #
    # Different explainers for different model types:
    #   - TreeExplainer: For tree-based models (fast, exact)
    #   - LinearExplainer: For linear models (fast, exact)
    #   - KernelExplainer: For any model (slow, approximate)
    #
    # We use aggressive SAMPLING (50 rows) for demo speed.
    # ============================================================

    def plot_shap_summary(self, X_train, X_test):
        """
        Generates a SHAP Summary Plot showing global feature importance.
        Each dot represents how much a feature contributed to one prediction.
        """
        if not SHAP_AVAILABLE:
            st.warning("SHAP library not installed. Install via `pip install shap`.")
            return None
            
        try:
            # OPTIMIZATION: Sample only 50 rows for speed (sufficient for visualization)
            sample_size = min(50, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=42)

            # Select the appropriate SHAP explainer based on model type
            if isinstance(self.model, (RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
                # TreeExplainer: Fast & exact for tree-based models
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample, check_additivity=False)
                
                # For binary classification, shap_values is [negative_class, positive_class]
                # We want the positive class (CKD=1) SHAP values
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                    
            elif isinstance(self.model, (LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet)):
                # LinearExplainer: Fast & exact for linear models
                explainer = shap.LinearExplainer(self.model, X_train.sample(min(50, len(X_train)), random_state=42))
                shap_values = explainer.shap_values(X_sample)
                
            else:
                # KernelExplainer: Works for ANY model (model-agnostic), but SLOW
                # Uses a weighted linear regression to approximate SHAP values
                explainer = shap.KernelExplainer(self.model.predict, shap.sample(X_train, 20))
                shap_values = explainer.shap_values(X_sample, nsamples=50)

            # Generate the SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            fig = plt.gcf()  # Get the figure created internally by SHAP
            fig.tight_layout()
            return fig
        except Exception as e:
            st.error(f"SHAP Error: {e}")
            return None


    # ============================================================
    # SHAP: Individual Patient Explanation (Waterfall Plot)
    # ============================================================
    # While the summary plot shows GLOBAL importance, the waterfall
    # plot explains a SINGLE patient's prediction:
    #   "Why was THIS patient classified as High Risk?"
    # 
    # It shows each feature's contribution:
    #   - Base value (average prediction)
    #   - + features pushing toward CKD
    #   - - features pushing toward Healthy
    #   = Final prediction
    # ============================================================

    def plot_shap_waterfall(self, X_train, X_single_instance, instance_index=0):
        """Generates a SHAP Waterfall Plot explaining one patient's prediction."""
        if not SHAP_AVAILABLE: return None
        try:
            # Get SHAP explanation object for the single patient
            if isinstance(self.model, (RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier, AdaBoostClassifier, XGBClassifier)):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer(X_single_instance)
                
            elif isinstance(self.model, (LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet)):
                explainer = shap.LinearExplainer(self.model, X_train.sample(min(50, len(X_train)), random_state=42))
                shap_values = explainer(X_single_instance)
            else:
                explainer = shap.KernelExplainer(self.model.predict, shap.sample(X_train, 20))
                shap_values = explainer(X_single_instance, nsamples=50)

            # Handle binary classification: select positive class SHAP values
            sv = shap_values[0,:,1] if len(shap_values.shape) > 2 else shap_values[0]
            
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(sv, show=False)
            fig = plt.gcf()
            fig.tight_layout()
            return fig
        except Exception as e:
            return None


    # ============================================================
    # SHAP: Top Feature Contributions for a Patient
    # ============================================================
    # Returns the top 5 features that most influenced a specific
    # patient's prediction, with their values and SHAP contributions.
    # Used in the Patient Screening results section.
    # ============================================================

    def get_feature_contributions(self, X_train, X_single_instance):
        """Returns top 5 features contributing to a single patient's CKD prediction."""
        if not SHAP_AVAILABLE: return None
        try:
            if isinstance(self.model, (RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier, AdaBoostClassifier, XGBClassifier)):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_single_instance, check_additivity=False)
                if isinstance(shap_values, list): shap_values = shap_values[1]
            elif isinstance(self.model, (LogisticRegression, LinearRegression)):
                explainer = shap.LinearExplainer(self.model, X_train.sample(min(50, len(X_train)), random_state=42))
                shap_values = explainer.shap_values(X_single_instance)
            else:
                explainer = shap.KernelExplainer(self.model.predict, shap.sample(X_train, 20))
                shap_values = explainer.shap_values(X_single_instance, nsamples=50)
            
            # Create a DataFrame of feature contributions, sorted by absolute impact
            df_contrib = pd.DataFrame({
                'Feature': X_single_instance.columns,
                'Contribution': shap_values[0],
                'Value': X_single_instance.values[0]
            })
            df_contrib['AbsContrib'] = df_contrib['Contribution'].abs()
            df_contrib = df_contrib.sort_values(by='AbsContrib', ascending=False).head(5)
            return df_contrib[['Feature', 'Value', 'Contribution']]
        except:
            return None


    # ============================================================
    # NATURAL LANGUAGE EXPLANATION
    # ============================================================
    # Generates a human-readable explanation for a patient's risk.
    # Combines global feature importance with patient-specific
    # critical values to create a text explanation like:
    # "The model relies heavily on Creatinine, Hemoglobin, Albumin.
    #  For this patient, Elevated Serum Creatinine and Low Hemoglobin
    #  significantly increased the risk score."
    # ============================================================

    def generate_patient_explanation(self, patient_data, feature_names):
        """Generates a text-based explanation for a patient's CKD risk prediction."""
        explanation = []
        
        # 1. Identify top globally important features
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models: feature_importances_ gives Gini importance
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = [feature_names[i] for i in indices[:3]]
            explanation.append(f"The model relies most heavily on **{', '.join(top_features)}**.")
            
        elif hasattr(self.model, 'coef_'):
            # For linear models: coefficients indicate feature weights
            coefs = pd.DataFrame({'Feature': feature_names, 'Coef': self.model.coef_.flatten()})
            coefs['AbsCoef'] = coefs['Coef'].abs()
            top_features = coefs.sort_values(by='AbsCoef', ascending=False).head(3)['Feature'].tolist()
            explanation.append(f"The strongest predictors are **{', '.join(top_features)}**.")
            
        # 2. Check patient-specific critical values (medical domain knowledge)
        critical_factors = []
        if 'Serum_Creatinine' in patient_data and patient_data['Serum_Creatinine'].values[0] > 1.2:
            critical_factors.append("Elevated Serum Creatinine")
        if 'Hemoglobin' in patient_data and patient_data['Hemoglobin'].values[0] < 12:
            critical_factors.append("Low Hemoglobin")
        if 'Albumin' in patient_data and patient_data['Albumin'].values[0] > 0:
            critical_factors.append("Albumin Check")
            
        if critical_factors:
            explanation.append(f"For this patient, **{', '.join(critical_factors)}** significantly increased the risk score.")
        else:
            explanation.append("The patient's values are largely within normal ranges, contributing to a lower risk assessment.")
            
        return " ".join(explanation)


    # ============================================================
    # FEATURE SELECTION: RFE (Recursive Feature Elimination)
    # ============================================================
    # Not all features are equally useful. Some may be noise.
    # RFE systematically removes the LEAST important features:
    #   1. Train model on all features
    #   2. Remove the least important feature
    #   3. Repeat until only n_features remain
    #
    # This improves model accuracy and reduces overfitting.
    # We use Logistic Regression as the base estimator for RFE
    # because it's fast and provides clear feature rankings.
    # ============================================================

    def select_features_rfe(self, X_train, y_train, n_features=5):
        """
        Selects the top N most important features using Recursive Feature Elimination.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_features: How many features to keep (default: 5)
        
        Returns:
            List of selected feature names
        """
        try:
            # Use Logistic Regression as the estimator for RFE
            model = LogisticRegression(max_iter=2000)
            rfe = RFE(estimator=model, n_features_to_select=n_features)
            rfe.fit(X_train, y_train)
            # rfe.support_ is a boolean mask of selected features
            self.selected_features = X_train.columns[rfe.support_]
            return self.selected_features
        except:
            # Fallback: Just take the first n_features columns
            return X_train.columns[:n_features]


    # ============================================================
    # VISUALIZATION METHODS
    # ============================================================
    # All methods below generate matplotlib figures for the dashboard.
    # They visualize model performance and help users understand results.
    # ============================================================

    def plot_residual_plot(self, y_test, y_pred):
        """
        Residual Plot (for Regression models):
        Shows the difference between actual and predicted values.
        Ideally, residuals should be randomly scattered around 0.
        Patterns in residuals indicate the model is missing something.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        residuals = y_test - y_pred
        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
        ax.axhline(0, color='r', linestyle='--')  # Zero line
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        fig.tight_layout()
        return fig
    
    # Alias for backward compatibility
    plot_residuals = plot_residual_plot

    def plot_tree_diagram(self, feature_names):
        """
        Visualizes a Decision Tree as a flowchart.
        Shows how the model makes decisions: each node is a question
        (e.g., "Is Creatinine > 1.2?") and leaves are predictions.
        For Random Forest, shows the first tree as a representative.
        max_depth=3 limits display to 3 levels for readability.
        """
        model_to_plot = None
        
        if isinstance(self.model, DecisionTreeClassifier):
            model_to_plot = self.model
        elif isinstance(self.model, RandomForestClassifier):
            # Plot the first tree in the forest as a representative
            model_to_plot = self.model.estimators_[0]
            
        if model_to_plot is None: return None

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(model_to_plot, filled=True, feature_names=feature_names,
                  ax=ax, max_depth=3, fontsize=10)
        ax.set_title(f"Tree Visualization ({'Random Forest - Tree 0' if isinstance(self.model, RandomForestClassifier) else 'Decision Tree'})")
        fig.tight_layout()
        return fig

    def plot_roc_curve(self, X_test, y_test):
        """
        ROC Curve (Receiver Operating Characteristic):
        Plots True Positive Rate vs False Positive Rate at all thresholds.
        
        - Diagonal line = random guessing (AUC = 0.5)
        - Curve hugging top-left = excellent model (AUC close to 1.0)
        - AUC = Area Under Curve = single number summarizing performance
        
        Medical context: A high AUC means the model can reliably
        distinguish CKD patients from healthy patients.
        """
        if not hasattr(self.model, "predict_proba"): return None
        
        try:
            y_prob = self.model.predict_proba(X_test)[:, 1]  # P(CKD)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', linestyle='--')  # Random baseline
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc='lower right')
            fig.tight_layout(pad=2.0)
            return fig
        except:
            return None

    def plot_confusion_matrix(self, y_test, y_pred, labels=["Healthy", "Risk"]):
        """
        Confusion Matrix:
        A 2x2 grid showing prediction outcomes:
        
        ┌─────────────────┬──────────────────┬───────────────────┐
        │                 │ Predicted Healthy │ Predicted CKD     │
        ├─────────────────┼──────────────────┼───────────────────┤
        │ Actually Healthy│ TRUE NEGATIVE ✓  │ FALSE POSITIVE ✗  │
        │ Actually CKD    │ FALSE NEGATIVE ✗ │ TRUE POSITIVE ✓   │
        └─────────────────┴──────────────────┴───────────────────┘
        
        In medical diagnosis:
        - False Negatives are DANGEROUS (patient has CKD but model says healthy)
        - We want to MINIMIZE false negatives (maximize Recall)
        """
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        return fig

    def plot_knn_elbow(self, X_train, y_train, X_test, y_test):
        """
        KNN Elbow Plot:
        Tests different values of K (number of neighbors) and plots
        the error rate. The "elbow" point where error stops decreasing
        significantly is the optimal K value.
        
        Low K (e.g., 1): Overfitting — too sensitive to noise
        High K (e.g., 30): Underfitting — too generalized
        """
        error_rates = []
        k_values = range(1, 40)
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rates.append(np.mean(pred_i != y_test))  # Error rate = wrong predictions / total
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, error_rates, color='blue', linestyle='dashed',
                marker='o', markerfacecolor='red', markersize=10)
        ax.set_title('Error Rate vs. K Value')
        ax.set_xlabel('K')
        ax.set_ylabel('Error Rate')
        fig.tight_layout()
        return fig
    
    def plot_learning_curve(self, X, y):
        """
        Learning Curve:
        Shows how model performance changes as we add more training data.
        
        - Gap between training & validation score = OVERFITTING
        - Both scores low = UNDERFITTING
        - Both scores high & converging = GOOD FIT
        
        Helps answer: "Do I need more data or a better model?"
        """
        from sklearn.model_selection import learning_curve
        
        try:
            # Train on increasing portions of data (10%, 30%, 50%, 75%, 100%)
            train_sizes, train_scores, test_scores = learning_curve(
                self.model, X, y, cv=5, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 5)
            )
            
            # Average scores across the 5 CV folds
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.grid()
            # Shaded bands show ±1 standard deviation (uncertainty)
            ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            ax.set_title(f"Learning Curve ({type(self.model).__name__})")
            ax.set_xlabel("Training examples")
            ax.set_ylabel("Score")
            ax.legend(loc="best")
            fig.tight_layout()
            return fig
        except:
            return None

    def plot_feature_importance(self, features):
        """
        Feature Importance Plot:
        Shows which features the model considers most important.
        
        For tree-based models: Uses Gini importance (how much each feature
        reduces impurity across all trees)
        
        For linear models: Uses absolute coefficient values (how much
        each feature contributes to the prediction)
        
        Higher bar = more important for the model's decisions
        """
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models store importance directly
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models: Use absolute coefficient values as importance
            importances = np.abs(self.model.coef_[0]) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
        else:
            return None
            
        # Create and sort a DataFrame of feature importances
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax, palette='viridis')
        ax.set_title("Feature Importance")
        return fig
