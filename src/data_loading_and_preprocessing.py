import pandas as pd
import boto3


# AWS S3 Configuration
BUCKET_NAME = 'financial-transaction-data-analysis'
FILE_KEY = 'merged_transactions_data.csv'


def load_dataset_from_s3(bucket_name, file_key):
    """
    Load the dataset from an S3 bucket.

    Parameters:
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key of the file in the S3 bucket.

    Returns:
        pd.DataFrame or None: Loaded DataFrame if successful, None otherwise.
    """
    s3_client = boto3.client('s3')
    try:
        print(f"Loading dataset from S3 bucket: {bucket_name}, key: {file_key}")
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_csv(obj['Body'])
        print("Dataset successfully loaded!")
        print("Columns in the dataset:", df.columns.tolist())  # Debugging: Print column names
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def preprocess_data(df):
    """
    Preprocess the dataset:
    - Handle missing values
    - Ensure timestamp column is ready

    Parameters:
        df (pd.DataFrame): Input DataFrame to be preprocessed.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    print("Starting data preprocessing...")

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        df.fillna(0, inplace=True)  # Replace missing values with 0 (or use domain-specific logic)

    # Check if 'timestamp' column exists
    if 'timestamp' in df.columns:
        print("Converting 'timestamp' column to datetime...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp')
    else:
        print("Warning: 'timestamp' column is missing. Skipping timestamp-based operations.")

    print("Preprocessing completed!")
    return df


def save_to_s3(df, bucket_name, file_key):
    """
    Save the processed dataset back to S3.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key for the file in the S3 bucket.

    Returns:
        None
    """
    s3_client = boto3.client('s3')
    try:
        print(f"Saving processed dataset to S3 bucket: {bucket_name}, key: {file_key}")
        csv_buffer = pd.io.common.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())
        print("Dataset successfully saved to S3!")
    except Exception as e:
        print(f"Error saving dataset: {e}")


def main():
    """
    Main function to orchestrate the data loading, preprocessing, and saving.
    """
    # Step 1: Load the dataset
    df = load_dataset_from_s3(BUCKET_NAME, FILE_KEY)
    if df is None:
        return

    # Step 2: Preprocess the data
    df = preprocess_data(df)

    # Step 3: Save the processed dataset back to S3
    processed_file_key = 'processed_financial_transaction_data.csv'
    save_to_s3(df, BUCKET_NAME, processed_file_key)


if __name__ == "__main__":
    main()