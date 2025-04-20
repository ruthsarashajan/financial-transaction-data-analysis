import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
from io import StringIO

# AWS S3 Configuration
BUCKET_NAME = 'financial-transaction-data-analysis'
FILE_KEY = 'processed_financial_transaction_data.csv'

def load_processed_data(bucket_name, file_key):
    """
    Load the processed dataset from an S3 bucket.

    Parameters:
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key of the file in the bucket.

    Returns:
        pd.DataFrame: Processed dataset as a DataFrame, or None if an error occurs.
    """
    s3_client = boto3.client('s3')
    try:
        print(f"Loading processed dataset from S3 bucket: {bucket_name}, key: {file_key}")
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_csv(obj['Body'])
        print("Processed dataset successfully loaded!")
        return df
    except Exception as e:
        print(f"Error loading processed dataset: {e}")
        return None

def transaction_volume_over_time(df):
    """
    Analyze and visualize transaction volume over time.

    Parameters:
        df (pd.DataFrame): The dataset containing transaction data.
    """
    if 'date' in df.columns:
        print("Analyzing transaction volume over time...")
        df['date'] = pd.to_datetime(df['date'])
        daily_volume = df.groupby('date')['amount'].sum()

        # Plot transaction volume
        plt.figure(figsize=(12, 6))
        plt.plot(daily_volume, marker='o', color='b', alpha=0.7)
        plt.title('Transaction Volume Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Transaction Amount', fontsize=12)
        plt.grid()
        plt.show()
    else:
        print("Error: 'date' column is missing. Cannot analyze transaction volume over time.")

def spending_distribution(df):
    """
    Visualize the distribution of transaction amounts.

    Parameters:
        df (pd.DataFrame): The dataset containing transaction data.
    """
    if 'amount' in df.columns:
        print("Visualizing transaction amount distribution...")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['amount'], kde=True, bins=50, color='blue', alpha=0.7)
        plt.title('Transaction Amount Distribution', fontsize=16)
        plt.xlabel('Transaction Amount', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid()
        plt.show()
    else:
        print("Error: 'amount' column is missing. Cannot visualize transaction amount distribution.")

def spending_trends(df):
    """
    Analyze spending trends by client.

    Parameters:
        df (pd.DataFrame): The dataset containing transaction data.
    """
    if 'client_id_x' in df.columns and 'amount' in df.columns:
        print("Analyzing spending trends by client...")
        client_spending = df.groupby('client_id_x')['amount'].sum().sort_values(ascending=False)

        # Top 20 clients by spending
        top_clients = client_spending.head(20)

        # Plot spending trends
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_clients.index, y=top_clients.values, palette='viridis', legend=False)
        plt.title('Top 20 Clients by Total Spending', fontsize=16)
        plt.xlabel('Client ID', fontsize=12)
        plt.ylabel('Total Spending', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()
    else:
        print("Error: Required columns ('client_id_x', 'amount') are missing. Cannot analyze spending trends.")

def detect_anomalies(df):
    """
    Detect anomalies in transaction amounts using a simple threshold-based method.

    Parameters:
        df (pd.DataFrame): The dataset containing transaction data.

    Returns:
        pd.DataFrame: DataFrame containing anomalies, or None if no anomalies are detected.
    """
    if 'amount' in df.columns:
        print("Detecting anomalies in transaction amounts...")

        # Ensure the 'amount' column is numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

        # Drop rows where 'amount' is NaN
        df = df.dropna(subset=['amount'])

        # Calculate the threshold for anomalies (e.g., 99th percentile)
        threshold = df['amount'].quantile(0.99)  # Transactions above the 99th percentile
        anomalies = df[df['amount'] > threshold]
        print(f"Number of anomalies detected: {len(anomalies)}")

        # Plot anomalies
        if not anomalies.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=anomalies, x='date', y='amount', color='red', alpha=0.6, label='Anomalies')
            plt.title('Anomalies in Transaction Amounts', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Transaction Amount', fontsize=12)
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print("No anomalies to plot.")

        return anomalies
    else:
        print("Error: 'amount' column is missing. Cannot detect anomalies.")
        return None

def main():
    """
    Main function to load data, perform EDA, and save anomalies.
    """
    # Step 1: Load the processed dataset
    df = load_processed_data(BUCKET_NAME, FILE_KEY)
    if df is None:
        return

    # Step 2: Perform EDA
    transaction_volume_over_time(df)
    spending_distribution(df)
    spending_trends(df)
    anomalies = detect_anomalies(df)

    # Save anomalies for further analysis
    if anomalies is not None and not anomalies.empty:
        print("Saving anomalies to a CSV file...")
        anomalies_file_key = 'anomalies_in_transactions.csv'
        csv_buffer = StringIO()
        anomalies.to_csv(csv_buffer, index=False)

        # Save to S3
        s3_client = boto3.client('s3')
        s3_client.put_object(Bucket=BUCKET_NAME, Key=anomalies_file_key, Body=csv_buffer.getvalue())
        print(f"Anomalies saved to S3 at: {BUCKET_NAME}/{anomalies_file_key}")
    else:
        print("No anomalies detected. Skipping CSV file saving...")

if __name__ == "__main__":
    main()