import boto3
import pandas as pd
import numpy as np
import pydicom
import cv2
from io import BytesIO
from sklearn.model_selection import train_test_split

# Define the S3 bucket and metadata file paths
bucket_name = 'your-s3-bucket-name'
train_metadata_files = [
    'path/to/calc_case_description_train_set.csv',
    'path/to/mass_case_description_train_set.csv'
]
test_metadata_files = [
    'path/to/calc_case_description_test_set.csv',
    'path/to/mass_case_description_test_set.csv'
]

# Initialize S3 client
s3_client = boto3.client('s3')

# Function to load metadata from S3
def load_metadata(bucket, key):
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj['Body'].read()))

# Load and concatenate training metadata files
train_metadata_df_list = [load_metadata(bucket_name, file) for file in train_metadata_files]
train_metadata_df = pd.concat(train_metadata_df_list, ignore_index=True)

# Load and concatenate test metadata files
test_metadata_df_list = [load_metadata(bucket_name, file) for file in test_metadata_files]
test_metadata_df = pd.concat(test_metadata_df_list, ignore_index=True)

# Function to read and preprocess a DICOM image from S3
def read_and_preprocess_dicom(bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    dicom_bytes = response['Body'].read()
    dicom = pydicom.dcmread(BytesIO(dicom_bytes))
    image = dicom.pixel_array
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized / np.max(image_resized)
    return image_normalized

# Generate training data
def generate_training_data(metadata_df, bucket):
    images = []
    labels = []
    for _, row in metadata_df.iterrows():
        file_path = row['image file path']
        label = 1 if row['pathology'] == 'MALIGNANT' else 0
        try:
            image = read_and_preprocess_dicom(bucket, file_path)
            images.append(image)
            labels.append(label)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
    return np.array(images), np.array(labels)

# Generate training data
X_train, y_train = generate_training_data(train_metadata_df, bucket_name)
X_train = X_train[..., np.newaxis]  # Add a channel dimension for grayscale images

# Generate test data
X_test, y_test = generate_training_data(test_metadata_df, bucket_name)
X_test = X_test[..., np.newaxis]  # Add a channel dimension for grayscale images

