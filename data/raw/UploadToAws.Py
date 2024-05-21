import os
import pandas as pd
import pydicom
import numpy as np
import cv2
import boto3
import requests
from io import BytesIO
from tcia_utils import nbia

# AWS S3 Configuration
bucket_name = 'cancerdetectiondata'
s3_client = boto3.client('s3')

# Load metadata CSV files
metadata_dir = 'metadata'  # Directory where metadata files are stored
mass_train_df = pd.read_csv(os.path.join(metadata_dir, 'mass_case_description_train_set.csv'))
calc_train_df = pd.read_csv(os.path.join(metadata_dir, 'calc_case_description_train_set.csv'))
mass_test_df = pd.read_csv(os.path.join(metadata_dir, 'mass_case_description_test_set.csv'))
calc_test_df = pd.read_csv(os.path.join(metadata_dir, 'calc_case_description_test_set.csv'))

# Combine all metadata into one DataFrame
metadata_df = pd.concat([mass_train_df, calc_train_df, mass_test_df, calc_test_df], ignore_index=True)

# Filter images based on annotations
relevant_images = metadata_df[metadata_df['pathology'].notnull()]

def read_dicom_image_from_bytes(dicom_bytes):
    dicom = pydicom.dcmread(BytesIO(dicom_bytes))
    image = dicom.pixel_array
    return image

def preprocess_image(image):
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized / np.max(image_resized)
    return image_normalized

def upload_to_s3(image, bucket_name, s3_key):
    _, buffer = cv2.imencode('.jpg', image * 255)
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.tobytes())

for index, row in relevant_images.iterrows():
    series_instance_uid = row['SeriesInstanceUID']  # Assuming there's a column with the Series Instance UID
    try:
        # Get the download URLs for the DICOM files in the series
        image_urls = nbia.get_image(series_instance_uid)
        for image_url in image_urls:
            dicom_bytes = requests.get(image_url).content
            image = read_dicom_image_from_bytes(dicom_bytes)
            processed_image = preprocess_image(image)
            s3_key = f"preprocessed_images/{os.path.basename(image_url).replace('.dcm', '.jpg')}"
            upload_to_s3(processed_image, bucket_name, s3_key)
            print(f"Uploaded {s3_key} to S3")
    except Exception as e:
        print(f"Failed to process {series_instance_uid}: {e}")

