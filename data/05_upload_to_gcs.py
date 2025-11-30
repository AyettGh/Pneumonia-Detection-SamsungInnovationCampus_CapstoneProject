import os
from google.cloud import storage
from glob import glob

# --- Configuration ---
BUCKET_NAME = "YOUR_BUCKET_NAME"  
LOCAL_DATA_PATH = "data/processed"
GCS_DESTINATION_PATH = "processed"
# ---------------------

def upload_to_gcs():
    """
    Uploads the entire 'data/processed' directory to GCS.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        print(f"Connected to bucket: {BUCKET_NAME}")
    except Exception as e:
        print(f"Error connecting to GCS: {e}")
        print("Please ensure you are authenticated ('gcloud auth login')")
        return

    # Use glob to find all files recursively
    # The '**' means it will search all subdirectories
    files_to_upload = glob(os.path.join(LOCAL_DATA_PATH, "**", "*"), recursive=True)

    print(f"Found {len(files_to_upload)} files to upload...")

    for local_file_path in files_to_upload:
        if os.path.isdir(local_file_path):
            continue  # Skip directories, only upload files

        # Create the destination path in the bucket
        # This removes the 'data/processed' prefix from the local path
        relative_path = os.path.relpath(local_file_path, LOCAL_DATA_PATH)
        # Use forward slashes for GCS paths
        gcs_blob_name = os.path.join(GCS_DESTINATION_PATH, relative_path).replace("\\", "/") 

        # Upload the file
        blob = bucket.blob(gcs_blob_name)
        try:
            blob.upload_from_filename(local_file_path)
        except Exception as e:
            print(f"Error uploading {local_file_path}: {e}")

    print(f"\nUpload complete. All files are in: gs://{BUCKET_NAME}/{GCS_DESTINATION_PATH}")


if __name__ == "__main__":
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"Error: Local path not found: {LOCAL_DATA_PATH}")
        print("Please run the 02 and 03 preprocessing scripts first.")
    else:
        upload_to_gcs()