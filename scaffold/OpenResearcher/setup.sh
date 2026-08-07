#!/bin/bash

set -e  # Exit on error

echo "================================"
echo "GPT-OSS-DeepResearch-Eval Setup"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if uv is installed
echo -e "\n${YELLOW}[1/6] Checking uv installation...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv is not installed. Please install it first:${NC}"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo -e "${GREEN}✓ uv is installed${NC}"

# Create virtual environment with Python 3.12
echo -e "\n${YELLOW}[2/6] Creating Python 3.12 virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
else
    uv venv --python 3.12
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}[3/6] Installing Python packages...${NC}"
source .venv/bin/activate

# Install Python dependencies
uv pip install -e .
echo -e "${GREEN}✓ Python packages installed${NC}"

# Install OpenJDK 21
echo -e "\n${YELLOW}[4/6] Installing OpenJDK 21...${NC}"
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
    echo "Java is already installed: $JAVA_VERSION"
    echo -e "${YELLOW}If you need OpenJDK 21, please install it manually:${NC}"
    echo "sudo apt install -y openjdk-21-jdk"
else
    echo -e "${YELLOW}Installing OpenJDK 21 (requires sudo)...${NC}"
    sudo apt install -y openjdk-21-jdk
    echo -e "${GREEN}✓ OpenJDK 21 installed${NC}"
fi

# Clone and install tevatron
echo -e "\n${YELLOW}[5/6] Installing tevatron...${NC}"
if [ -d "tevatron" ]; then
    echo -e "${YELLOW}tevatron directory already exists. Skipping clone.${NC}"
    cd tevatron
    uv pip install -e .
    cd ..
else
    git clone https://github.com/texttron/tevatron.git
    cd tevatron
    uv pip install -e .
    cd ..
    echo -e "${GREEN}✓ tevatron installed${NC}"
fi

# Check Lucene JARs (already in tevatron/)
echo -e "\n${YELLOW}[6/9] Checking Lucene highlighter JARs...${NC}"
if [ -f "tevatron/lucene-highlighter-9.9.1.jar" ]; then
    echo -e "${GREEN}✓ Lucene JARs found in tevatron/${NC}"
else
    echo -e "${YELLOW}Downloading Lucene highlighter JARs to tevatron/...${NC}"
    cd tevatron
    LUCENE_VERSION="9.9.1"
    wget -q "https://repo1.maven.org/maven2/org/apache/lucene/lucene-highlighter/${LUCENE_VERSION}/lucene-highlighter-${LUCENE_VERSION}.jar"
    wget -q "https://repo1.maven.org/maven2/org/apache/lucene/lucene-queries/${LUCENE_VERSION}/lucene-queries-${LUCENE_VERSION}.jar"
    wget -q "https://repo1.maven.org/maven2/org/apache/lucene/lucene-memory/${LUCENE_VERSION}/lucene-memory-${LUCENE_VERSION}.jar"
    cd ..
    echo -e "${GREEN}✓ Lucene JARs downloaded${NC}"
fi

# Check huggingface-cli installation
echo -e "\n${YELLOW}[7/9] Checking huggingface-cli installation...${NC}"
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}huggingface-cli not found. Installing...${NC}"
    uv pip install huggingface_hub[cli]
    echo -e "${GREEN}✓ huggingface-cli installed${NC}"
else
    echo -e "${GREEN}✓ huggingface-cli is already installed${NC}"
fi

# Download test dataset (queries and answers)
echo -e "\n${YELLOW}[8/11] Downloading test dataset from Hugging Face...${NC}"
if [ -d "Tevatron/browsecomp-plus" ]; then
    echo -e "${YELLOW}Test dataset already exists. Skipping download.${NC}"
else
    mkdir -p Tevatron
    echo -e "${YELLOW}Downloading Tevatron/browsecomp-plus (test queries and answers)...${NC}"
    huggingface-cli download Tevatron/browsecomp-plus --repo-type=dataset --local-dir ./Tevatron/browsecomp-plus
    echo -e "${GREEN}✓ Test dataset downloaded${NC}"
fi

# Download corpus
echo -e "\n${YELLOW}[9/11] Downloading corpus from Hugging Face...${NC}"
if [ -d "Tevatron/browsecomp-plus-corpus" ]; then
    echo -e "${YELLOW}Corpus already exists. Skipping download.${NC}"
else
    mkdir -p Tevatron
    echo -e "${YELLOW}Downloading Tevatron/browsecomp-plus-corpus...${NC}"
    huggingface-cli download Tevatron/browsecomp-plus-corpus --repo-type=dataset --local-dir ./Tevatron/browsecomp-plus-corpus
    echo -e "${GREEN}✓ Corpus downloaded${NC}"
fi

# Download indexes
echo -e "\n${YELLOW}[10/11] Downloading BM25 index from Hugging Face...${NC}"
if [ -d "Tevatron/browsecomp-plus-indexes/bm25" ]; then
    echo -e "${YELLOW}BM25 index already exists. Skipping.${NC}"
else
    mkdir -p Tevatron/browsecomp-plus-indexes
    echo -e "${YELLOW}Downloading BM25 index (~2.1GB)...${NC}"
    huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="bm25/*" --local-dir ./Tevatron/browsecomp-plus-indexes
    echo -e "${GREEN}✓ BM25 index downloaded${NC}"
fi

# Download Qwen3-Embedding-8B index
echo -e "\n${YELLOW}[11/11] Downloading Qwen3-Embedding-8B index from Hugging Face...${NC}"
if [ -d "Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b" ]; then
    echo -e "${YELLOW}Qwen3-Embedding-8B index already exists. Skipping.${NC}"
else
    mkdir -p Tevatron/browsecomp-plus-indexes
    echo -e "${YELLOW}Downloading Qwen3-Embedding-8B index (~1.6GB, this may take a while)...${NC}"
    huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-8b/*" --local-dir ./Tevatron/browsecomp-plus-indexes
    echo -e "${GREEN}✓ Qwen3-Embedding-8B index downloaded${NC}"
fi
