import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
import optuna
import pickle
import os
from datetime import datetime
import featuretools as ft
from boruta import BorutaPy
# import tensorflow as tf  # TensorFlowは後で必要に応じて追加
# from tensorflow import keras
# from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data_for_modeling(data, target_column=None):
    """
    Prepare data for machine learning modeling

    Args:
        data (pandas.DataFrame): Input data
        target_column (str): Name of the target column

    Returns:
        tuple: (X, y, feature_names, target_encoder, scaler)
    """
    try:
        # Automatically detect target column (last numeric column)
        if target_column is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                target_column = numeric_columns[-1]
            else:
                # If no numeric columns, use the last column
                target_column = data.columns[-1]

        # Separate features and target
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            X = data
            y = data.iloc[:, -1]  # Use last column as target

        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        # Handle target variable if it's categorical
        target_encoder = None
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # Convert to numpy arrays
        X = X.values
        y = y.values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info(f"Data prepared for modeling. Features: {X_scaled.shape[1]}, Samples: {X_scaled.shape[0]}")
        return X_scaled, y, X.columns.tolist() if hasattr(X, 'columns') else None, target_encoder, scaler

    except Exception as e:
        logger.error(f"Error preparing data for modeling: {str(e)}")
        raise

def create_ml_model(X, y, model_type="xgboost", target_encoder=None):
    """
    Create and train a machine learning model

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        model_type (str): Type of model to create ("xgboost", "lightgbm")
        target_encoder: Encoder for target variable (if needed)

    Returns:
        tuple: (model, model_params, feature_importance)
    """
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create model based on type
        if model_type == "xgboost":
            model = xgb.XGBRegressor(random_state=42)
        elif model_type == "lightgbm":
            model = lgb.LGBMRegressor(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        if len(np.unique(y)) <= 2:  # Binary classification
            accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
            metrics = {"accuracy": accuracy}
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            metrics = {"mse": mse, "rmse": rmse, "r2": r2}

        # Get feature importance
        feature_importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else None

        logger.info(f"Model created successfully with {model_type}")
        return model, metrics, feature_importance

    except Exception as e:
        logger.error(f"Error creating ML model: {str(e)}")
        raise

def create_advanced_ml_model(X, y, model_type="xgboost", target_encoder=None, feature_selection_method="boruta", hyperparameter_tuning=True):
    """
    Create and train an advanced machine learning model with feature engineering and hyperparameter tuning

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        model_type (str): Type of model to create ("xgboost", "lightgbm", "tensorflow")
        target_encoder: Encoder for target variable (if needed)
        feature_selection_method (str): Method for feature selection ("boruta", "rfe", "selectkbest")
        hyperparameter_tuning (bool): Whether to perform hyperparameter tuning

    Returns:
        tuple: (model, model_params, feature_importance, feature_names)
    """
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature selection
        X_selected = X_train
        feature_names = None
        feature_selector = None

        if feature_selection_method == "boruta":
            # Boruta feature selection
            boruta_selector = BorutaPy(
                estimator=xgb.XGBRegressor(random_state=42) if model_type == "xgboost" else lgb.LGBMRegressor(random_state=42),
                n_estimators='auto',
                random_state=42
            )
            boruta_selector.fit(X_train, y_train)
            X_selected = boruta_selector.transform(X_train)
            feature_names = [f"feature_{i}" for i in range(X_selected.shape[1])]
            feature_selector = boruta_selector
        elif feature_selection_method == "rfe":
            # Recursive Feature Elimination
            estimator = xgb.XGBRegressor(random_state=42) if model_type == "xgboost" else lgb.LGBMRegressor(random_state=42)
            rfe_selector = RFE(estimator, n_features_to_select=min(10, X_train.shape[1]))
            rfe_selector.fit(X_train, y_train)
            X_selected = rfe_selector.transform(X_train)
            feature_names = [f"feature_{i}" for i in range(X_selected.shape[1])]
            feature_selector = rfe_selector
        elif feature_selection_method == "selectkbest":
            # SelectKBest
            selector = SelectKBest(score_func=f_classif, k=min(10, X_train.shape[1]))
            X_selected = selector.fit_transform(X_train, y_train)
            feature_names = [f"feature_{i}" for i in range(X_selected.shape[1])]
            feature_selector = selector
        else:
            # No feature selection
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Hyperparameter tuning
        best_params = {}
        if hyperparameter_tuning:
            if model_type == "xgboost":
                best_params, _ = perform_hyperparameter_tuning(X_selected, y_train, "xgboost", 20)
                model = xgb.XGBRegressor(**best_params, random_state=42)
            elif model_type == "lightgbm":
                best_params, _ = perform_hyperparameter_tuning(X_selected, y_train, "lightgbm", 20)
                model = lgb.LGBMRegressor(**best_params, random_state=42)
            else:
                raise ValueError(f"Unsupported model type for hyperparameter tuning: {model_type}")
        else:
            # Use default parameters
            if model_type == "xgboost":
                model = xgb.XGBRegressor(random_state=42)
            elif model_type == "lightgbm":
                model = lgb.LGBMRegressor(random_state=42)
            else:
                raise ValueError(f"Unsupported model type for hyperparameter tuning: {model_type}")

        # Train model
        model.fit(X_selected, y_train)

        # Make predictions
        if feature_selection_method != "none":
            X_test_selected = feature_selector.transform(X_test)
        else:
            X_test_selected = X_test
        y_pred = model.predict(X_test_selected)

        # Calculate metrics
        if len(np.unique(y)) <= 2:  # Binary classification
            accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
            metrics = {"accuracy": accuracy}
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            metrics = {"mse": mse, "rmse": rmse, "r2": r2}

        # Get feature importance
        feature_importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else None

        logger.info(f"Advanced model created successfully with {model_type}")
        return model, metrics, feature_importance, feature_names

    except Exception as e:
        logger.error(f"Error creating advanced ML model: {str(e)}")
        raise

def perform_hyperparameter_tuning(X, y, model_type="xgboost", n_trials=20):
    """
    Perform hyperparameter tuning using Optuna

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        model_type (str): Type of model to tune ("xgboost", "lightgbm")
        n_trials (int): Number of trials for optimization

    Returns:
        dict: Best parameters and best score
    """
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective(trial):
            if model_type == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
                }
                model = xgb.XGBRegressor(**params, random_state=42)
            elif model_type == "lightgbm":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
                }
                model = lgb.LGBMRegressor(**params, random_state=42)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return mean_squared_error(y_test, y_pred)

        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Hyperparameter tuning completed for {model_type}")
        return study.best_params, study.best_value

    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        raise

def feature_selection(X, y, k=10):
    """
    Perform feature selection

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        k (int): Number of top features to select

    Returns:
        tuple: (selected_features, feature_scores)
    """
    try:
        # Use SelectKBest for feature selection
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        # Get feature scores
        feature_scores = selector.scores_

        logger.info(f"Feature selection completed. Selected {X_selected.shape[1]} features")
        return X_selected, feature_scores

    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        raise

def save_model(model, model_path, model_info=None):
    """
    Save trained model to file

    Args:
        model: Trained model
        model_path (str): Path to save the model
        model_info (dict): Additional information about the model

    Returns:
        str: Path where model was saved
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save model info if provided
        if model_info:
            info_path = model_path.replace('.pkl', '_info.json')
            import json
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)

        logger.info(f"Model saved successfully to {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(model_path):
    """
    Load trained model from file

    Args:
        model_path (str): Path to the saved model

    Returns:
        object: Loaded model
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_model_report(model, metrics, feature_importance, model_type):
    """
    Generate a report about the trained model

    Args:
        model: Trained model
        metrics (dict): Model metrics
        feature_importance (list): Feature importance scores
        model_type (str): Type of model

    Returns:
        str: Formatted model report
    """
    try:
        report = f"## モデル分析レポート ({model_type})\n\n"

        # Model metrics
        report += "## モデル評価指標\n"
        for metric, value in metrics.items():
            report += f"- {metric}: {value:.4f}\n"

        # Feature importance
        if feature_importance is not None:
            report += "\n### 特徴量重要度\n"
            # For simplicity, we'll just show the top 5 features
            top_features = np.argsort(feature_importance)[-5:][::-1]
            for i, idx in enumerate(top_features):
                report += f"{i+1}. 特徴量 {idx}: {feature_importance[idx]:.4f}\n"

        logger.info("Model report generated")
        return report

    except Exception as e:
        logger.error(f"Error generating model report: {str(e)}")
        return f"モデルレポートの生成に失敗しました: {str(e)}"

def perform_statistical_analysis(data, analysis_type, target_column=None):
    """
    Perform various statistical analyses on the data

    Args:
        data (pandas.DataFrame): Input data
        analysis_type (str): Type of analysis to perform
        target_column (str): Target column for analysis

    Returns:
        dict: Analysis results
    """
    try:
        results = {}

        if analysis_type == "logistic_regression":
            # Logistic Regression Analysis
            if target_column is None:
                # Automatically detect target column (last numeric column)
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    target_column = numeric_columns[-1]
                else:
                    # If no numeric columns, use the last column
                    target_column = data.columns[-1]

            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            # Handle target variable
            # Check if target is categorical (object type) or binary numeric
            if y.dtype == 'object':
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))
            elif y.dtype in [np.float64, np.float32, np.int64, np.int32]:
                # If target is numeric, check if it's actually a classification problem
                # by checking if it has only a few distinct values (e.g., 0 and 1)
                unique_values = np.unique(y)
                # Check if all values are integers (for numeric classification)
                if all(val == int(val) for val in unique_values):
                    # Convert to int for comparison
                    unique_values = [int(val) for val in unique_values]
                    if len(unique_values) <= 10 and all(val in [0, 1] for val in unique_values):
                        # Treat as binary classification
                        pass
                    else:
                        # Treat as regression problem - cannot use logistic regression
                        results = {"error": "Logistic regression requires a binary target variable for classification, but the target is continuous."}
                        return results
                else:
                    # If values are not all integers, treat as regression problem
                    results = {"error": "Logistic regression requires a binary target variable for classification, but the target is continuous."}
                    return results
            else:
                # For other types, assume it's a regression problem
                results = {"error": "Logistic regression requires a binary target variable for classification, but the target is continuous."}
                return results

            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create and train model
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {"accuracy": accuracy}

            # Get feature importance (coefficients)
            feature_importance = model.coef_[0] if hasattr(model, 'coef_') else None

            results = {
                "model_type": "logistic_regression",
                "metrics": metrics,
                "feature_importance": feature_importance,
                "model": model
            }

        elif analysis_type == "linear_regression":
            # Linear Regression Analysis
            if target_column is None:
                # Automatically detect target column (last numeric column)
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    target_column = numeric_columns[-1]
                else:
                    # If no numeric columns, use the last column
                    target_column = data.columns[-1]

            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            # Handle target variable
            # Check if target is categorical (object type)
            if y.dtype == 'object':
                # If target is categorical, we can't do linear regression
                results = {"error": "Linear regression requires a continuous target variable, but the target is categorical."}
                return results
            elif y.dtype in [np.float64, np.float32, np.int64, np.int32]:
                # If target is numeric, it's suitable for linear regression
                pass
            else:
                # For other types, we can't do linear regression
                results = {"error": "Linear regression requires a continuous target variable, but the target is not numeric."}
                return results

            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create and train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            metrics = {"mse": mse, "rmse": rmse, "r2": r2}

            # Get feature importance (coefficients)
            feature_importance = model.coef_ if hasattr(model, 'coef_') else None

            results = {
                "model_type": "linear_regression",
                "metrics": metrics,
                "feature_importance": feature_importance,
                "model": model
            }

        elif analysis_type == "association_analysis":
            # Association Analysis (Simple correlation)
            numeric_data = data.select_dtypes(include=[np.number])

            if len(numeric_data.columns) < 2:
                results = {"error": "Not enough numeric columns for correlation analysis"}
            else:
                # Calculate correlation matrix
                correlation_matrix = numeric_data.corr()

                # Find strong correlations
                strong_correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:  # Strong correlation threshold
                            strong_correlations.append({
                                "feature1": correlation_matrix.columns[i],
                                "feature2": correlation_matrix.columns[j],
                                "correlation": corr_value
                            })

                results = {
                    "model_type": "association_analysis",
                    "correlation_matrix": correlation_matrix,
                    "strong_correlations": strong_correlations
                }

        elif analysis_type == "clustering":
            # Clustering Analysis
            numeric_data = data.select_dtypes(include=[np.number])

            if len(numeric_data.columns) < 2:
                results = {"error": "Not enough numeric columns for clustering analysis"}
            else:
                # Handle missing values
                numeric_data = numeric_data.fillna(numeric_data.mean())

                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)

                # Determine optimal number of clusters using elbow method
                # For simplicity, we'll use 3 clusters
                n_clusters = 3

                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)

                # Calculate silhouette score
                silhouette_avg = silhouette_score(scaled_data, cluster_labels)

                results = {
                    "model_type": "clustering",
                    "n_clusters": n_clusters,
                    "silhouette_score": silhouette_avg,
                    "cluster_labels": cluster_labels,
                    "cluster_centers": kmeans.cluster_centers_
                }

        else:
            results = {"error": f"Unsupported analysis type: {analysis_type}"}

        logger.info(f"Statistical analysis completed for {analysis_type}")
        return results

    except Exception as e:
        logger.error(f"Error in statistical analysis: {str(e)}")
        raise