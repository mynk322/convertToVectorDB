#!/bin/bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/facebookresearch/faiss"  # Example repository
API_HOST="127.0.0.1"
API_PORT="5000"
API_URL="http://${API_HOST}:${API_PORT}"
MODEL="all-MiniLM-L6-v2"
QUERY="vector similarity search algorithm"

echo -e "${BLUE}=== GitHub Repository to Vector Database Demo ===${NC}"
echo

# Step 1: Start the mock vector database API server
echo -e "${GREEN}Starting mock vector database API server...${NC}"
echo "Press Ctrl+C to stop the server when done"
python mock_vector_db_api.py --host $API_HOST --port $API_PORT &
SERVER_PID=$!

# Give the server time to start
sleep 2
echo

# Step 2: Convert a GitHub repository to vector database
echo -e "${GREEN}Converting GitHub repository to vector database...${NC}"
echo "Repository: $REPO_URL"
echo "API URL: $API_URL"
python repo_to_vector.py --repo-url $REPO_URL --api-url $API_URL --model $MODEL
echo

# Step 3: Query the vector database
echo -e "${GREEN}Querying the vector database...${NC}"
echo "Query: \"$QUERY\""
python query_vector_db.py --query "$QUERY" --api-url $API_URL --model $MODEL
echo

# Step 4: Show database statistics
echo -e "${GREEN}Vector database statistics:${NC}"
curl -s $API_URL/stats | python -m json.tool
echo

# Cleanup
echo -e "${GREEN}Demo completed. Stopping server...${NC}"
kill $SERVER_PID

echo -e "${BLUE}=== Demo Finished ===${NC}"
