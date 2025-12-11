#!/usr/bin/env python3
"""
Upload compressed search indexes to Cloudflare R2
Alternative to the bash script - uses boto3 directly
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

try:
    import boto3
    from botocore.config import Config
except ImportError:
    print("boto3 not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3", "python-dotenv"])
    import boto3
    from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "foia-documents")

if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
    print("Missing R2 credentials in .env file")
    print("Required: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
    sys.exit(1)

s3 = boto3.client(
    's3',
    endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)

def upload_file(local_path: Path, s3_key: str):
    file_size = local_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"Uploading {local_path.name} ({file_size_mb:.1f} MB)...")
    
    try:
        s3.upload_file(
            str(local_path),
            BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': 'application/gzip'}
        )
        print(f"{local_path.name} uploaded successfully")
        return True
    except Exception as e:
        print(f"Failed to upload {local_path.name}: {e}")
        return False

def main():
    print("Uploading search indexes to Cloudflare R2...")
    print(f"Bucket: {BUCKET_NAME}")
    print("")
    
    files_to_upload = [
        ("lancedb_store.tar.gz", "indexes/lancedb_store.tar.gz"),
        ("tantivy_store.tar.gz", "indexes/tantivy_store.tar.gz"),
        ("search_indexes.tar.gz", "indexes/search_indexes.tar.gz"),
        ("foia_ai_db.tar.gz", "database/foia_ai_db.tar.gz"),
    ]
    
    uploaded = []
    skipped = []
    
    for local_file, s3_key in files_to_upload:
        local_path = DATA_DIR / local_file
        
        if not local_path.exists():
            print(f"{local_file} not found - skipping")
            skipped.append(local_file)
            continue
        
        if upload_file(local_path, s3_key):
            uploaded.append(local_file)
    
    print("\n" + "="*60)
    if uploaded:
        print(f"Successfully uploaded {len(uploaded)} file(s):")
        for f in uploaded:
            print(f"   - {f}")
    
    if skipped:
        print(f"\nSkipped {len(skipped)} file(s) (not found):")
        for f in skipped:
            print(f"   - {f}")
        print("\nRun ./scripts/compress_indexes.sh first to create these files")
    
    print("\n" + "="*60)
    print("Public URLs (update these in download_indexes.sh):")
    print(f"https://pub-{R2_ACCOUNT_ID[:6]}xxxx.r2.dev/indexes/lancedb_store.tar.gz")
    print(f"https://pub-{R2_ACCOUNT_ID[:6]}xxxx.r2.dev/indexes/tantivy_store.tar.gz")
    print("\nTo get your actual R2.dev URL:")
    print("1. Go to Cloudflare R2 dashboard")
    print("2. Select your bucket")
    print("3. Settings → Public Access → Enable")

if __name__ == "__main__":
    main()

