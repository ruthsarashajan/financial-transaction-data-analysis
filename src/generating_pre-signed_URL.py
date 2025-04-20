import boto3

s3 = boto3.client(
    's3',
    aws_access_key_id='AKIA4HEBM2ASJ7LO34GE',
    aws_secret_access_key='K0oxmOYk9c6ZJDfBLNd/nqSfe0jQ+6akjdS5DCU3',
    region_name='us-east-1'  # Replace with your bucket's region
)

bucket_name = "financial-transaction-data-analysis"
file_key = "combined_financial_data.csv"

# Generate a pre-signed URL valid for 1 hour (3600 seconds)
pre_signed_url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': bucket_name, 'Key': file_key},
    ExpiresIn=3600
)

print("Pre-Signed URL:", pre_signed_url)