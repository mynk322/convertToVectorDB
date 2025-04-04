#!/bin/bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/facebookresearch/faiss"  # Example repository
API_HOST="127.0.0.1"
API_PORT="5000"
API_URL="http://${API_HOST}:${API_PORT}"
MODEL="all-MiniLM-L6-v2"
QUERY="vector similarity search algorithm"
OUTPUT_DIR="repo_data"
CHUNK_SIZE=1000
MAX_RETRIES=3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --api-port)
      API_PORT="$2"
      API_URL="http://${API_HOST}:${API_PORT}"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --query)
      QUERY="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --repo-url URL     GitHub repository URL to process (default: $REPO_URL)"
      echo "  --api-port PORT    Port for the mock API server (default: $API_PORT)"
      echo "  --model NAME       Sentence transformer model to use (default: $MODEL)"
      echo "  --query TEXT       Query to search for (default: $QUERY)"
      echo "  --output-dir DIR   Directory to store temporary data (default: $OUTPUT_DIR)"
      echo "  --chunk-size SIZE  Size of text chunks for processing (default: $CHUNK_SIZE)"
      echo "  --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}=== GitHub Repository to Vector Database Demo ===${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Repository: ${REPO_URL}"
echo -e "  API URL: ${API_URL}"
echo -e "  Model: ${MODEL}"
echo -e "  Output directory: ${OUTPUT_DIR}"
echo -e "  Chunk size: ${CHUNK_SIZE}"
echo -e "  Query: \"${QUERY}\""
echo

# Check if required tools are installed
echo -e "${BLUE}Checking dependencies...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed${NC}"
    exit 1
fi

# Function to check if server is running
check_server() {
    local url="$1"
    local endpoints=("/health" "/stats" "health" "stats")
    
    for endpoint in "${endpoints[@]}"; do
        if curl -s "${url}${endpoint}" &> /dev/null; then
            return 0  # Server is running
        fi
    done
    
    return 1  # Server is not running
}

# Check if the API server is already running
if check_server "${API_URL}/"; then
    echo -e "${YELLOW}Warning: A server is already running at ${API_URL}${NC}"
    read -p "Do you want to continue using the existing server? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborting demo${NC}"
        exit 1
    fi
    SERVER_STARTED=false
else
    # Step 1: Start the mock vector database API server
    echo -e "${GREEN}Starting mock vector database API server...${NC}"
    echo "Press Ctrl+C to stop the server when done"
    python mock_vector_db_api.py --host $API_HOST --port $API_PORT --debug &
    SERVER_PID=$!
    SERVER_STARTED=true

    # Give the server time to start
    echo "Waiting for server to start..."
    for i in {1..10}; do
        if check_server "${API_URL}/"; then
            echo -e "${GREEN}Server started successfully${NC}"
            break
        fi
        if [ $i -eq 10 ]; then
            echo -e "${RED}Error: Server failed to start${NC}"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
        echo "Waiting... ($i/10)"
        sleep 1
    done
    echo
fi

# Step 2: Convert a GitHub repository to vector database
echo -e "${GREEN}Converting GitHub repository to vector database...${NC}"
echo "Repository: $REPO_URL"
echo "API URL: $API_URL"

# Try conversion with retries
for i in $(seq 1 $MAX_RETRIES); do
    echo -e "Attempt $i of $MAX_RETRIES"
    python repo_to_vector.py --repo-url $REPO_URL --api-url $API_URL --model $MODEL --output-dir $OUTPUT_DIR --chunk-size $CHUNK_SIZE
    CONVERT_STATUS=$?
    
    if [ $CONVERT_STATUS -eq 0 ]; then
        echo -e "${GREEN}Repository conversion successful${NC}"
        break
    else
        echo -e "${YELLOW}Repository conversion failed (attempt $i of $MAX_RETRIES)${NC}"
        
        if [ $i -eq $MAX_RETRIES ]; then
            echo -e "${RED}Error: Repository conversion failed after $MAX_RETRIES attempts${NC}"
            if [ "$SERVER_STARTED" = true ]; then
                echo -e "${GREEN}Stopping server...${NC}"
                kill $SERVER_PID 2>/dev/null || true
            fi
            exit 1
        fi
        
        echo "Retrying in 3 seconds..."
        sleep 3
    fi
done
echo

# Step 3: Query the vector database
echo -e "${GREEN}Querying the vector database...${NC}"
echo "Query: \"$QUERY\""

# Try query with retries
for i in $(seq 1 $MAX_RETRIES); do
    echo -e "Attempt $i of $MAX_RETRIES"
    python query_vector_db.py --query "$QUERY" --api-url $API_URL --model $MODEL --verbose
    QUERY_STATUS=$?
    
    if [ $QUERY_STATUS -eq 0 ]; then
        break
    else
        echo -e "${YELLOW}Query failed (attempt $i of $MAX_RETRIES)${NC}"
        
        if [ $i -eq $MAX_RETRIES ]; then
            echo -e "${RED}Error: Query failed after $MAX_RETRIES attempts${NC}"
            break
        fi
        
        echo "Retrying in 3 seconds..."
        sleep 3
    fi
done
echo

# Step 4: Show database statistics
echo -e "${GREEN}Vector database statistics:${NC}"
curl -s $API_URL/stats | python -m json.tool
echo

# Cleanup
if [ "$SERVER_STARTED" = true ]; then
    echo -e "${GREEN}Demo completed. Stopping server...${NC}"
    kill $SERVER_PID 2>/dev/null || true
else
    echo -e "${GREEN}Demo completed. Server was already running, not stopping it.${NC}"
fi

echo -e "${BLUE}=== Demo Finished ===${NC}"
