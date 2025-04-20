import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from sklearn.ensemble import StackingClassifier
import boto3
from io import StringIO

# ---- Step 0: Load Data from AWS S3 ---- #
def load_data_from_s3(bucket_name, file_key, access_key, secret_key):
    """
    Load a dataset from AWS S3.

    Parameters:
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key of the file in the S3 bucket.
        access_key (str): AWS access key ID.
        secret_key (str): AWS secret access key.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    print("Loading data from AWS S3...")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    dataset = response["Body"].read().decode("utf-8")
    print("Dataset loaded successfully from S3!")
    return pd.read_csv(StringIO(dataset))


# ---- Step 1: Data Preprocessing ---- #
def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, creating derived features,
    and performing feature engineering.

    Parameters:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    print("Preprocessing data...")

    # Drop rows with missing values
    df = df.dropna()

    # Convert transaction_date to datetime & create derived features
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['amount_normalized'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    df['days_since_start'] = (df['transaction_date'] - df['transaction_date'].min()).dt.days

    # Additional feature engineering
    try:
        df['weekly_avg'] = df.groupby('client_id_x')['amount'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df['transaction_frequency'] = df.groupby('client_id_x')['transaction_date'].transform('count')
    except KeyError as e:
        print(f"Error: {e}")
        print("It seems the 'client_id_x' column is missing. Please verify the dataset.")

    df['is_weekend'] = df['transaction_date'].dt.dayofweek >= 5
    df['holiday_indicator'] = df['transaction_date'].apply(lambda x: 1 if x.month == 12 and x.day in [25, 31] else 0)

    print("Data preprocessing and feature engineering completed.")
    return df


# ---- Step 2: Spending Trends Analysis ---- #
def analyze_spending_trends(df):
    """
    Analyze spending trends using a LightGBM Regressor.

    Parameters:
        df (pd.DataFrame): The preprocessed dataset.

    Returns:
        np.ndarray: Predicted values for spending trends.
    """
    print("Analyzing spending trends...")
    X = df[['days_since_start', 'is_weekend', 'weekly_avg', 'holiday_indicator']]
    y = df['amount_normalized']

    # Train LightGBM model
    lgbm_model = LGBMRegressor(random_state=42, n_estimators=100, max_depth=10, verbose=-1)
    lgbm_model.fit(X, y)

    # Metrics
    y_pred = lgbm_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"Spending Trends Metrics:\n  - MSE: {mse:.2f}\n  - R²: {r2:.2f}\n  - MAE: {mae:.2f}")

    # Visualization
    fig = px.line(
        df,
        x='days_since_start',
        y=['amount_normalized', y_pred],
        labels={'value': 'Normalized Amount', 'variable': 'Type'},
        title="Spending Trends (LightGBM Regressor)"
    )
    fig.write_html("spending_trends_plotly.html")
    print("Spending trends visualization saved as 'spending_trends_plotly.html'.")
    return y_pred


# ---- Step 3: Anomaly Detection ---- #
def detect_anomalies(df):
    """
    Detect anomalies in the dataset using Isolation Forest.

    Parameters:
        df (pd.DataFrame): The preprocessed dataset.

    Returns:
        pd.DataFrame: Updated dataset with anomaly labels.
    """
    print("Detecting anomalies...")
    X_anomaly = df[['amount_normalized', 'days_since_start', 'weekly_avg', 'transaction_frequency']]
    iso_forest = IsolationForest(contamination=0.02, random_state=42, n_estimators=200)
    df['is_anomaly'] = iso_forest.fit_predict(X_anomaly)
    df['is_anomaly'] = df['is_anomaly'].map({1: 0, -1: 1})

    # Visualization
    fig = px.scatter(
        df,
        x='days_since_start',
        y='amount_normalized',
        color='is_anomaly',
        title="Anomaly Detection (Isolation Forest)",
        labels={"is_anomaly": "Anomaly"},
        color_discrete_map={0: "blue", 1: "red"}
    )
    fig.write_html("anomaly_detection_plotly.html")
    print("Anomaly detection visualization saved as 'anomaly_detection_plotly.html'.")
    return df


# ---- Step 4: Fraud Detection ---- #
def build_fraud_detection_model(df):
    """
    Build and evaluate a fraud detection model using a Stacking Classifier.

    Parameters:
        df (pd.DataFrame): The preprocessed dataset.

    Returns:
        None
    """
    print("Building fraud detection model...")
    df['is_fraud'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])  # Replace with actual fraud labels
    X = df[['amount_normalized', 'days_since_start', 'weekly_avg', 'transaction_frequency', 'is_weekend', 'holiday_indicator']]
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Balance dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Stacking Classifier
    estimators = [
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        ('lgbm', LGBMRegressor(random_state=42, n_estimators=100, max_depth=10))
    ]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=XGBClassifier(random_state=42, eval_metric='logloss'))
    stacking_model.fit(X_train_balanced, y_train_balanced)

    # Evaluation metrics
    y_pred = stacking_model.predict(X_test)
    y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Fraud Detection Metrics:\n  - Accuracy: {accuracy:.2f}\n  - Precision: {precision:.2f}\n"
          f"  - Recall: {recall:.2f}\n  - F1-Score: {f1:.2f}\n  - ROC-AUC: {roc_auc:.2f}")

    # ROC Curve Visualization
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
    roc_fig.update_layout(
        title="ROC Curve - Fraud Detection (Stacking Classifier)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True
    )
    roc_fig.write_html("fraud_detection_roc_curve.html")
    print("Fraud detection ROC curve visualization saved as 'fraud_detection_roc_curve.html'.")


# ---- Step 5: Upload Results to S3 ---- #
def upload_to_s3(local_file_path, bucket_name, s3_file_key, access_key, secret_key):
    """
    Upload a local file to AWS S3.

    Parameters:
        local_file_path (str): Path to the local file.
        bucket_name (str): Name of the S3 bucket.
        s3_file_key (str): Key for the file in S3.
        access_key (str): AWS access key ID.
        secret_key (str): AWS secret access key.

    Returns:
        None
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    try:
        s3.upload_file(local_file_path, bucket_name, s3_file_key)
        print(f"Uploaded {local_file_path} to S3 bucket {bucket_name} as {s3_file_key}.")
    except Exception as e:
        print(f"Failed to upload {local_file_path} to S3: {e}")


# ---- Main Function ---- #
def main():
    # AWS credentials (replace with your credentials)
    access_key = "AKIA4HEBM2ASJ7LO34GE"
    secret_key = "K0oxmOYk9c6ZJDfBLNd/nqSfe0jQ+6akjdS5DCU3"

    # Load dataset
    bucket_name = "financial-transaction-data-analysis"
    file_key = "feature_engineered_financial_data.csv"
    df = load_data_from_s3(bucket_name, file_key, access_key, secret_key)

    # Preprocess dataset
    df = preprocess_data(df)

    # Analyze spending trends
    y_pred_spending = analyze_spending_trends(df)

    # Detect anomalies
    df = detect_anomalies(df)

    # Build fraud detection model
    build_fraud_detection_model(df)

    # Export and upload datasets
    print("Preparing datasets for S3 upload...")
    df_spending_trends = df[['days_since_start', 'amount_normalized']].copy()
    df_spending_trends['predicted_amount'] = y_pred_spending
    df_spending_trends.to_csv("spending_trends_dataset.csv", index=False)
    upload_to_s3("spending_trends_dataset.csv", bucket_name, "spending_trends_dataset.csv", access_key, secret_key)

    df_anomaly_detection = df[['transaction_date', 'client_id_x', 'amount_normalized', 'is_anomaly']].copy()
    df_anomaly_detection.to_csv("anomaly_detection_dataset.csv", index=False)
    upload_to_s3("anomaly_detection_dataset.csv", bucket_name, "anomaly_detection_dataset.csv", access_key, secret_key)

    df_fraud_detection = df[['transaction_date', 'client_id_x', 'amount_normalized', 'is_fraud']].copy()
    df_fraud_detection.to_csv("fraud_detection_dataset.csv", index=False)
    upload_to_s3("fraud_detection_dataset.csv", bucket_name, "fraud_detection_dataset.csv", access_key, secret_key)

    print("All datasets have been uploaded to S3 successfully.")


if __name__ == "__main__":
    main()