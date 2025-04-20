import pandas as pd
import boto3
import io

# S3 Bucket Configuration
BUCKET_NAME = "financial-transaction-data-analysis"  # Replace with your S3 bucket name
FILE_KEY = "processed_financial_transaction_data.csv"  # Replace with your file path in S3

# Initialize S3 Client
s3 = boto3.client("s3")

def load_data_from_s3(bucket_name, file_key):
    """
    Load a CSV file from an S3 bucket into a Pandas DataFrame.

    Parameters:
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key (path) of the file in the S3 bucket.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    print("Downloading file from S3...")
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    print("File loaded successfully from S3.")
    return df

def clean_and_prepare_data(df):
    """
    Clean and preprocess the dataset, including handling missing values and creating new features.

    Parameters:
        df (pd.DataFrame): The original dataset.

    Returns:
        pd.DataFrame: The cleaned and feature-engineered dataset.
    """
    # Inspect dataset columns
    print("Columns in dataset:", df.columns.tolist())

    # Clean and convert the 'amount' column to numeric
    print("Cleaning and converting 'amount' column to numeric...")
    df['amount'] = df['amount'].replace('[\$,]', '', regex=True)  # Remove dollar signs and commas
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')  # Convert to numeric, set invalid entries to NaN

    # Check for invalid values
    if df['amount'].isnull().any():
        print("Warning: Some 'amount' values could not be converted to numeric and were set to NaN.")
        print(df[df['amount'].isnull()][['amount']].head())  # Print examples of invalid values

    # Create spending trend features
    print("Creating spending trend features...")
    client_spending_stats = df.groupby('client_id_x')['amount'].agg(['mean', 'min', 'max']).rename(
        columns={'mean': 'avg_spending', 'min': 'min_spending', 'max': 'max_spending'}
    )
    df = df.merge(client_spending_stats, on='client_id_x')

    # Ensure 'date' column exists
    if 'date' in df.columns:
        df.rename(columns={'date': 'transaction_date'}, inplace=True)
    else:
        raise KeyError("The dataset does not contain a 'date' column or an equivalent transaction date column.")

    # Create transaction frequency features
    print("Creating transaction frequency features...")
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])  # Ensure date column is in datetime format
    df['day_of_week'] = df['transaction_date'].dt.dayofweek  # Day of the week (0 = Monday, 6 = Sunday)
    df['hour_of_day'] = df['transaction_date'].dt.hour  # Hour of day
    df['month'] = df['transaction_date'].dt.month  # Month of the year

    # Transaction frequency by client
    transaction_freq_by_client = df.groupby('client_id_x')['id_x'].count().rename('transaction_count')
    df = df.merge(transaction_freq_by_client, on='client_id_x')

    # Transaction frequency by merchant
    transaction_freq_by_merchant = df.groupby('merchant_id')['id_x'].count().rename('merchant_transaction_count')
    df = df.merge(transaction_freq_by_merchant, on='merchant_id')

    # Spending trends by merchant
    print("Creating spending trend features...")
    merchant_spending_stats = df.groupby('merchant_id')['amount'].agg(['mean', 'min', 'max']).rename(
        columns={'mean': 'avg_merchant_spending', 'min': 'min_merchant_spending', 'max': 'max_merchant_spending'}
    )
    df = df.merge(merchant_spending_stats, on='merchant_id')

    # Normalize transaction amounts
    print("Normalizing transaction amounts...")
    df['amount_normalized'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()

    # Encode categorical variables
    print("Encoding categorical variables...")
    df['client_id_encoded'] = df['client_id_x'].astype('category').cat.codes
    df['merchant_id_encoded'] = df['merchant_id'].astype('category').cat.codes

    return df

def save_data_to_s3(df, bucket_name, output_file_key):
    """
    Save the processed dataset to an S3 bucket.

    Parameters:
        df (pd.DataFrame): The processed dataset.
        bucket_name (str): Name of the S3 bucket.
        output_file_key (str): Key (path) for the output file in the S3 bucket.
    """
    print("Uploading feature-engineered dataset back to S3...")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=output_file_key, Body=csv_buffer.getvalue())
    print(f"Feature-engineered dataset saved to S3 at {output_file_key}.")

def main():
    """
    Main function to load, process, and save the dataset.
    """
    # Load the processed dataset from S3
    df = load_data_from_s3(BUCKET_NAME, FILE_KEY)

    # Clean and process the dataset
    df = clean_and_prepare_data(df)

    # Save the processed dataset back to S3
    OUTPUT_FILE_KEY = "feature_engineered_financial_data.csv"  # Replace with your desired output file path in S3
    save_data_to_s3(df, BUCKET_NAME, OUTPUT_FILE_KEY)

if __name__ == "__main__":
    main()