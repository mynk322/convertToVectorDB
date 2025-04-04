import argparse
import requests
import json
from sentence_transformers import SentenceTransformer

def query_vector_database(query_text, api_url, model_name="all-MiniLM-L6-v2", top_k=5):
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
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embedding for the query
    query_embedding = model.encode(query_text, convert_to_tensor=False).tolist()
    
    # Send query to the vector database API
    print(f"Querying vector database at {api_url}")
    try:
        response = requests.post(
            f"{api_url}/query",
            json={
                "query_embedding": query_embedding,
                "top_k": top_k
            }
        )
        
        if response.status_code == 200:
            results = response.json().get("results", [])
            print(f"Found {len(results)} results")
            return results
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"Error querying vector database: {str(e)}")
        return []

def display_results(results):
    """
    Display the query results in a readable format.
    
    Args:
        results: List of result documents from the vector database
    """
    if not results:
        print("No results found.")
        return
    
    print("\n" + "="*80)
    print("SEARCH RESULTS")
    print("="*80)
    
    for i, result in enumerate(results):
        print(f"\nResult #{i+1} (Score: {result.get('score', 'N/A'):.4f})")
        print(f"File: {result.get('path', 'Unknown path')}")
        print(f"Chunk: {result.get('chunk_index', 0)+1} of {result.get('total_chunks', 1)}")
        print("-"*80)
        print(result.get('content', 'No content available')[:500] + "..." if len(result.get('content', '')) > 500 else result.get('content', 'No content available'))
        print("-"*80)

def main():
    parser = argparse.ArgumentParser(description='Query the vector database')
    parser.add_argument('--query', required=True, help='Text to search for')
    parser.add_argument('--api-url', required=True, help='URL of the vector database API')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Name of the sentence-transformer model')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    results = query_vector_database(
        query_text=args.query,
        api_url=args.api_url,
        model_name=args.model,
        top_k=args.top_k
    )
    
    display_results(results)

if __name__ == "__main__":
    main()
