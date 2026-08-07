#!/bin/bash

# Start search service with proper environment setup
# Usage: ./scripts/start_search_service.sh [bm25|dense] [port]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

export CUDA_VISIBLE_DEVICES=${3:-7}
# Set GPU_IDS to 0 because CUDA_VISIBLE_DEVICES restricts the view to only the selected GPUs, 
# so the application sees them starting from index 0.
export GPU_IDS=0

# Parameters
SEARCHER_TYPE="${1:-dense}"
PORT="${2:-8000}"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Starting Search Service${NC}"
echo -e "${GREEN}================================${NC}"
echo "Searcher Type: ${SEARCHER_TYPE}"
echo "Port: ${PORT}"
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Please run ./setup.sh first${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Verify we're using the right Python
PYTHON_VERSION=$(python --version)
echo "Using: $PYTHON_VERSION"
echo "Python path: $(which python)"
echo ""

# Set common environment variables
export LUCENE_EXTRA_DIR="${PROJECT_ROOT}/tevatron"
export CORPUS_PARQUET_PATH="${PROJECT_ROOT}/Tevatron/browsecomp-plus-corpus/data/*.parquet"

echo "LUCENE_EXTRA_DIR: ${LUCENE_EXTRA_DIR}"
echo "CORPUS_PARQUET_PATH: ${CORPUS_PARQUET_PATH}"

# Check if Lucene JARs exist
if [ ! -f "${LUCENE_EXTRA_DIR}/lucene-highlighter-9.9.1.jar" ]; then
    echo -e "${RED}Error: Lucene JARs not found in ${LUCENE_EXTRA_DIR}${NC}"
    echo "Please run ./setup.sh to download them"
    exit 1
fi

# Check if corpus exists
CORPUS_COUNT=$(ls ${PROJECT_ROOT}/Tevatron/browsecomp-plus-corpus/data/*.parquet 2>/dev/null | wc -l)
if [ "$CORPUS_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: Corpus not found${NC}"
    echo "Please run ./setup.sh to download the corpus"
    exit 1
fi

# Configure searcher-specific settings
if [ "$SEARCHER_TYPE" = "bm25" ]; then
    export LUCENE_INDEX_DIR="${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/bm25"
    export SEARCHER_TYPE="bm25"

    if [ ! -d "$LUCENE_INDEX_DIR" ]; then
        echo -e "${RED}Error: BM25 index not found at $LUCENE_INDEX_DIR${NC}"
        echo "Please run ./setup.sh to download the index"
        exit 1
    fi

    echo "LUCENE_INDEX_DIR: ${LUCENE_INDEX_DIR}"
    echo "SEARCHER_TYPE: ${SEARCHER_TYPE}"

elif [ "$SEARCHER_TYPE" = "dense" ]; then
    export DENSE_INDEX_PATH="${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl"
    export DENSE_MODEL_NAME="Qwen/Qwen3-Embedding-8B"
    export SEARCHER_TYPE="dense"

    # Check if index files exist
    INDEX_FILES=$(ls ${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl 2>/dev/null | wc -l)
    if [ "$INDEX_FILES" -eq 0 ]; then
        echo -e "${RED}Error: Dense index not found${NC}"
        echo "Please run ./setup.sh to download the index"
        exit 1
    fi

    echo "DENSE_INDEX_PATH: ${DENSE_INDEX_PATH}"
    echo "DENSE_MODEL_NAME: ${DENSE_MODEL_NAME}"
    echo "SEARCHER_TYPE: ${SEARCHER_TYPE}"

else
    echo -e "${RED}Error: Invalid searcher type. Use 'bm25' or 'dense'${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Starting uvicorn server...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Start uvicorn
uvicorn scripts.deploy_search_service:app --host 0.0.0.0 --port ${PORT}
