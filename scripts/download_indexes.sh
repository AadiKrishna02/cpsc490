#!/bin/bash
# Download pre-built search indexes
# Users run this after cloning the repo

echo "Downloading pre-built search indexes..."
echo "This will download ~10GB of data and may take 10-30 minutes."
echo ""

# Cloudflare R2 public URLs
LANCEDB_URL="https://pub-9a2bbfbefc804ee6b876c870df20d034.r2.dev/indexes/lancedb_store.tar.gz"
TANTIVY_URL="https://pub-9a2bbfbefc804ee6b876c870df20d034.r2.dev/indexes/tantivy_store.tar.gz"
DATABASE_URL="https://pub-9a2bbfbefc804ee6b876c870df20d034.r2.dev/database/foia_ai_db.tar.gz"

# Create data directory if it doesn't exist
mkdir -p data
cd data

# Download LanceDB
echo "Downloading LanceDB store (~6GB)..."
if command -v wget &> /dev/null; then
    wget -c $LANCEDB_URL -O lancedb_store.tar.gz
elif command -v curl &> /dev/null; then
    curl -L -C - $LANCEDB_URL -o lancedb_store.tar.gz
else
    echo "Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Download Tantivy
echo "Downloading Tantivy store (~5GB)..."
if command -v wget &> /dev/null; then
    wget -c $TANTIVY_URL -O tantivy_store.tar.gz
elif command -v curl &> /dev/null; then
    curl -L -C - $TANTIVY_URL -o tantivy_store.tar.gz
else
    echo "Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Extract LanceDB
echo "Extracting LanceDB..."
tar -xzf lancedb_store.tar.gz
rm lancedb_store.tar.gz
echo "LanceDB ready"

# Extract Tantivy
echo "Extracting Tantivy..."
tar -xzf tantivy_store.tar.gz
rm tantivy_store.tar.gz
echo "Tantivy ready"

# Download Database
echo "Downloading database (~160MB)..."
if command -v wget &> /dev/null; then
    wget -c $DATABASE_URL -O foia_ai_db.tar.gz
elif command -v curl &> /dev/null; then
    curl -L -C - $DATABASE_URL -o foia_ai_db.tar.gz
else
    echo "Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Extract Database
echo "Extracting database..."
tar -xzf foia_ai_db.tar.gz
rm foia_ai_db.tar.gz
echo "Database ready"

echo ""
echo "All components downloaded and extracted!"
echo "  - LanceDB (vector search)"
echo "  - Tantivy (keyword search)"
echo "  - Database (documents & pages)"
echo ""
echo "You can now run: python scripts/wiki_web.py"

