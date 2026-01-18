"""
Google Cloud Storage utilities for the Financial RAG system.

Handles uploading and downloading the vector database to/from GCS.

Usage (CLI):
    python -m financial_agent.gcs_storage upload --bucket my-bucket
    python -m financial_agent.gcs_storage download --bucket my-bucket
    python -m financial_agent.gcs_storage list --bucket my-bucket

Usage (Python):
    from financial_agent.gcs_storage import upload_vector_db, download_vector_db

    # Upload local vector_db to GCS
    upload_vector_db("vector_db", "my-bucket")

    # Download from GCS to local
    download_vector_db("my-bucket", "vector_db")
"""

from pathlib import Path
from google.cloud import storage


def upload_vector_db(local_path: str, bucket_name: str, blob_prefix: str = "vector_db/"):
    """
    Upload the vector database files to GCS.

    Args:
        local_path: Path to local vector_db directory
        bucket_name: GCS bucket name
        blob_prefix: Prefix for blobs in GCS (default: "vector_db/")
    """
    local_path = Path(local_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files_to_upload = [
        "faiss_index.index",
        "chunks_metadata.json"
    ]

    for filename in files_to_upload:
        file_path = local_path / filename
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        blob_name = f"{blob_prefix}{filename}"
        blob = bucket.blob(blob_name)

        print(f"Uploading {file_path} to gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(str(file_path))

    print("Upload complete!")


def download_vector_db(bucket_name: str, local_path: str, blob_prefix: str = "vector_db/"):
    """
    Download the vector database files from GCS.

    Args:
        bucket_name: GCS bucket name
        local_path: Path to local vector_db directory
        blob_prefix: Prefix for blobs in GCS (default: "vector_db/")
    """
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files_to_download = [
        "faiss_index.index",
        "chunks_metadata.json"
    ]

    for filename in files_to_download:
        blob_name = f"{blob_prefix}{filename}"
        blob = bucket.blob(blob_name)

        file_path = local_path / filename
        print(f"Downloading gs://{bucket_name}/{blob_name} to {file_path}")
        blob.download_to_filename(str(file_path))

    print("Download complete!")


def list_available_dbs(bucket_name: str, prefix: str = ""):
    """
    List available vector databases in a GCS bucket.

    Args:
        bucket_name: GCS bucket name
        prefix: Optional prefix to filter by

    Returns:
        List of available vector_db prefixes
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)

    # Find unique prefixes that contain faiss_index.index
    dbs = set()
    for blob in blobs:
        if blob.name.endswith("faiss_index.index"):
            # Extract the prefix (everything before faiss_index.index)
            db_prefix = blob.name.replace("faiss_index.index", "")
            dbs.add(db_prefix)

    return sorted(list(dbs))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GCS storage utilities for vector database")
    parser.add_argument("action", choices=["upload", "download", "list"])
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--local-path", default="vector_db", help="Local vector_db directory")
    parser.add_argument("--prefix", default="vector_db/", help="GCS blob prefix")

    args = parser.parse_args()

    if args.action == "upload":
        upload_vector_db(args.local_path, args.bucket, args.prefix)
    elif args.action == "download":
        download_vector_db(args.bucket, args.local_path, args.prefix)
    elif args.action == "list":
        dbs = list_available_dbs(args.bucket, args.prefix)
        print("Available vector databases:")
        for db in dbs:
            print(f"  - {db}")
