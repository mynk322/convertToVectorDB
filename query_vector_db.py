import argparse
import requests
import json
import time
import sys
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('query_vector_db.log')
    ]
)
logger = logging.getLogger(__name__)

def check_api_connection(api_url: str) -> bool:
    """
    Check if the vector database API is accessible.
    
    Args:
        api_url: URL of the vector database API
        
    Returns:
        True if the API is accessible, False otherwise
    """
    try:
        logger.info(f"Checking connection to vector database API at {api_url}")
        
        # Try different endpoint paths for flexibility
        endpoints = ["/stats", "stats", "/health", "health"]
        
        for endpoint in endpoints:
            try:
                # Construct the URL carefully to handle different formats
                url = api_url
                if not url.endswith('/') and not endpoint.startswith('/'):
                    url += '/'
                url += endpoint
                
                logger.debug(f"Trying to connect to: {url}")
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    # Try to get document count if available
                    try:
                        stats = response.json()
                        total_docs = stats.get("total_documents", stats.get("documents_count", 0))
                        logger.info(f"Successfully connected to vector database API at {url}")
                        logger.info(f"Database contains {total_docs} documents")
                    except:
                        logger.info(f"Successfully connected to vector database API at {url}")
                    
                    return True
            except requests.exceptions.RequestException:
                continue
        
        # If we get here, all connection attempts failed
        logger.error(f"Failed to connect to vector database API at {api_url}")
        logger.error("Please ensure the vector database API server is running and the URL is correct.")
        logger.error("You can start the mock API server with: python mock_vector_db_api.py")
        return False
        
    except Exception as e:
        logger.error(f"Error connecting to vector database API: {str(e)}")
        logger.error("Please ensure the vector database API server is running and the URL is correct.")
        logger.error("You can start the mock API server with: python mock_vector_db_api.py")
        return False

def query_vector_database(query_text: str, api_url: str, model_name: str = "all-MiniLM-L6-v2", top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the vector database with a text query.
    
    Args:
        query_text: The text to search for
        api_url: URL of the vector database API
        model_name: Name of the sentence-transformer model (must match the one used for conversion)
        top_k: Number of results to return
        
    Returns:
        List of matching documents
    """
    # Validate inputs
    if not query_text or not query_text.strip():
        logger.error("Query text cannot be empty")
        return []
        
    if not api_url or not api_url.strip():
        logger.error("API URL cannot be empty")
        return []
        
    if top_k <= 0:
        logger.warning(f"Invalid top_k value: {top_k}, using default of 5")
        top_k = 5
    
    # Check API connection first
    if not check_api_connection(api_url):
        logger.error("Cannot proceed without vector database API connection")
        logger.error(f"Please ensure the API server is running at: {api_url}")
        logger.error("You can start the mock API server with: python mock_vector_db_api.py --host 127.0.0.1 --port <port>")
        return []
    
    start_time = time.time()
    logger.info(f"Loading model: {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        model_load_time = time.time() - start_time
        logger.info(f"Model loaded in {model_load_time:.2f} seconds")
        
        # Generate embedding for the query
        embedding_start = time.time()
        query_embedding = model.encode(query_text, convert_to_tensor=False).tolist()
        embedding_time = time.time() - embedding_start
        logger.info(f"Query embedding generated in {embedding_time:.2f} seconds")
        
        # Send query to the vector database API
        logger.info(f"Querying vector database at {api_url}")
        query_start = time.time()
        
        try:
            # Construct the URL carefully
            query_url = api_url
            if not query_url.endswith('/'):
                query_url += '/'
            query_url += 'query'
            
            logger.debug(f"Sending query to: {query_url}")
            response = requests.post(
                query_url,
                json={
                    "query_embedding": query_embedding,
                    "top_k": top_k
                },
                timeout=10  # 10 second timeout
            )
            
            query_time = time.time() - query_start
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                logger.info(f"Found {len(results)} results in {query_time:.2f} seconds")
                
                # Log some stats about the results
                if results:
                    avg_score = sum(r.get('score', 0) for r in results) / len(results)
                    logger.info(f"Average similarity score: {avg_score:.4f}")
                    
                    # Count unique files in results
                    unique_files = len(set(r.get('path', '') for r in results))
                    logger.info(f"Results come from {unique_files} unique files")
                
                return results
            else:
                logger.error(f"Error: {response.status_code} - {response.text}")
                return []
                
        except requests.exceptions.Timeout:
            logger.error("Timeout querying vector database")
            return []
        except requests.exceptions.ConnectionError:
            logger.error("Connection error querying vector database")
            return []
        except Exception as e:
            logger.error(f"Error querying vector database: {str(e)}")
            return []
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return []

def display_results(results: List[Dict[str, Any]], query_text: str) -> None:
    """
    Display the query results in a readable format.
    
    Args:
        results: List of result documents from the vector database
        query_text: The original query text
    """
    if not results:
        logger.info("No results found.")
        print("\n" + "="*80)
        print("SEARCH RESULTS FOR: " + query_text)
        print("="*80)
        print("\nNo results found.")
        return
    
    logger.info(f"Displaying {len(results)} results")
    print("\n" + "="*80)
    print(f"SEARCH RESULTS FOR: {query_text}")
    print("="*80)
    
    for i, result in enumerate(results):
        score = result.get('score', 0)
        path = result.get('path', 'Unknown path')
        chunk_index = result.get('chunk_index', 0) + 1
        total_chunks = result.get('total_chunks', 1)
        content = result.get('content', 'No content available')
        
        # Truncate content if too long
        if len(content) > 500:
            content = content[:500] + "..."
        
        print(f"\nResult #{i+1} (Score: {score:.4f})")
        print(f"File: {path}")
        print(f"Chunk: {chunk_index} of {total_chunks}")
        print("-"*80)
        print(content)
        print("-"*80)

def main():
    parser = argparse.ArgumentParser(description='Query the vector database')
    parser.add_argument('--query', required=True, help='Text to search for')
    parser.add_argument('--api-url', required=True, help='URL of the vector database API')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Name of the sentence-transformer model')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    logger.info(f"Starting query: '{args.query}'")
    start_time = time.time()
    
    try:
        results = query_vector_database(
            query_text=args.query,
            api_url=args.api_url,
            model_name=args.model,
            top_k=args.top_k
        )
        
        total_time = time.time() - start_time
        logger.info(f"Query completed in {total_time:.2f} seconds")
        
        display_results(results, args.query)
        
    except KeyboardInterrupt:
        logger.info("Query interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
