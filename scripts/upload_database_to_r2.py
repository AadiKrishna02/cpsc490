#!/usr/bin/env python3
"""
Upload database to Cloudflare R2 (separate from indexes)
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
    sys.exit(1)

s3 = boto3.client(
    's3',
    endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)

def upload_file(local_path: Path, s3_key: str):
    """Upload a file to R2 with progress"""
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
    print("Uploading database to Cloudflare R2...")
    print(f"Bucket: {BUCKET_NAME}")
    print("")
    
    local_path = DATA_DIR / "foia_ai_db.tar.gz"
    s3_key = "database/foia_ai_db.tar.gz"
    
    if not local_path.exists():
        print(f"{local_path} not found!")
        print("   Run: cd data && tar -czf foia_ai_db.tar.gz foia_ai.db foia_ai.db-shm foia_ai.db-wal")
        sys.exit(1)
    
    if upload_file(local_path, s3_key):
        print("\n" + "="*60)
        print("Database uploaded successfully")
        print("\nPublic URL:")
        print(f"https://pub-{R2_ACCOUNT_ID[:6]}xxxx.r2.dev/database/foia_ai_db.tar.gz")
        print("\nUpdate scripts/download_indexes.sh with this URL if needed")
    else:
        print("\nUpload failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

