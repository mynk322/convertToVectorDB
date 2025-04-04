import os
import json
import time
import argparse
import numpy as np
import logging
from flask import Flask, request, jsonify, Response
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vector_db_api.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure Flask logging to use our logger
app.logger.handlers = logger.handlers
app.logger.setLevel(logger.level)

# In-memory storage for documents and embeddings
documents = []
embeddings = []

# Directory to save/load the database
DB_DIR = Path("vector_db_data")
DB_FILE = DB_DIR / "documents.json"
BACKUP_DIR = DB_DIR / "backups"

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def save_database() -> bool:
    """
    Save the current database to disk.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        start_time = time.time()
        os.makedirs(DB_DIR, exist_ok=True)
        
        # Create a backup of the existing database if it exists
        if os.path.exists(DB_FILE):
            os.makedirs(BACKUP_DIR, exist_ok=True)
            backup_file = BACKUP_DIR / f"documents_{int(time.time())}.json.bak"
            try:
                import shutil
                shutil.copy2(DB_FILE, backup_file)
                logger.info(f"Created backup at {backup_file}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {str(e)}")
        
        # Save the database
        with open(DB_FILE, 'w') as f:
            json.dump(documents, f)
            
        save_time = time.time() - start_time
        logger.info(f"Database saved to {DB_FILE} in {save_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Error saving database: {str(e)}")
        return False

def load_database() -> Tuple[int, int]:
    """
    Load the database from disk if it exists.
    
    Returns:
        Tuple of (number of documents loaded, number of embeddings loaded)
    """
    global documents, embeddings
    
    if os.path.exists(DB_FILE):
        try:
            start_time = time.time()
            file_size = os.path.getsize(DB_FILE) / (1024 * 1024)  # Size in MB
            
            with open(DB_FILE, 'r') as f:
                documents = json.load(f)
                
            # Validate documents
            valid_docs = []
            for doc in documents:
                if not isinstance(doc, dict):
                    logger.warning(f"Skipping invalid document: not a dictionary")
                    continue
                    
                if 'embedding' not in doc:
                    logger.warning(f"Skipping document without embedding: {doc.get('path', 'unknown')}")
                    continue
                    
                valid_docs.append(doc)
                
            documents = valid_docs
                
            # Extract embeddings from documents
            embeddings = []
            for doc in documents:
                try:
                    embeddings.append(np.array(doc['embedding']))
                except Exception as e:
                    logger.warning(f"Error converting embedding for {doc.get('path', 'unknown')}: {str(e)}")
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {len(documents)} documents ({file_size:.2f} MB) in {load_time:.2f} seconds")
            
            # Log some stats about the database
            if documents:
                extensions = {}
                for doc in documents:
                    ext = doc.get('extension', 'unknown')
                    extensions[ext] = extensions.get(ext, 0) + 1
                
                logger.info(f"Database contains documents with extensions: {extensions}")
                
            return len(documents), len(embeddings)
        except json.JSONDecodeError:
            logger.error(f"Error decoding database file: {DB_FILE}")
            return 0, 0
        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
            return 0, 0
    else:
        logger.info("No existing database found.")
        return 0, 0

@app.route('/add_documents', methods=['POST'])
def add_documents() -> Response:
    """
    Add documents to the vector database.
    
    Returns:
        Flask response with JSON data
    """
    global documents, embeddings
    
    start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            logger.error("Request does not contain JSON data")
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        new_docs = data.get('documents', [])
        
        if not new_docs:
            logger.warning("No documents provided in request")
            return jsonify({"error": "No documents provided"}), 400
        
        logger.info(f"Received request to add {len(new_docs)} documents")
        
        # Validate documents
        valid_docs = []
        invalid_count = 0
        
        for doc in new_docs:
            if not isinstance(doc, dict):
                invalid_count += 1
                continue
                
            if 'embedding' not in doc or 'content' not in doc:
                invalid_count += 1
                continue
                
            valid_docs.append(doc)
        
        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid documents")
        
        if not valid_docs:
            logger.error("No valid documents to add")
            return jsonify({"error": "No valid documents provided"}), 400
        
        # Add documents to the database
        documents.extend(valid_docs)
        
        # Extract embeddings
        new_embeddings = []
        for doc in valid_docs:
            try:
                new_embeddings.append(np.array(doc['embedding']))
            except Exception as e:
                logger.warning(f"Error converting embedding: {str(e)}")
                # Remove the corresponding document
                documents.pop()
        
        embeddings.extend(new_embeddings)
        
        # Save the updated database
        save_success = save_database()
        
        process_time = time.time() - start_time
        logger.info(f"Added {len(new_embeddings)} documents in {process_time:.2f} seconds")
        
        return jsonify({
            "success": save_success,
            "message": f"Added {len(new_embeddings)} documents to the database",
            "total_documents": len(documents),
            "processing_time": process_time
        })
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query() -> Response:
    """
    Query the vector database with an embedding.
    
    Returns:
        Flask response with JSON data containing search results
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            logger.error("Query request does not contain JSON data")
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        query_embedding = data.get('query_embedding')
        top_k = data.get('top_k', 5)
        
        if not query_embedding:
            logger.error("No query embedding provided")
            return jsonify({"error": "No query embedding provided"}), 400
        
        if not isinstance(top_k, int) or top_k <= 0:
            logger.warning(f"Invalid top_k value: {top_k}, using default of 5")
            top_k = 5
        
        if not documents:
            logger.warning("Query attempted on empty database")
            return jsonify({"results": []}), 200
        
        # Convert query embedding to numpy array
        try:
            query_vector = np.array(query_embedding)
        except Exception as e:
            logger.error(f"Error converting query embedding: {str(e)}")
            return jsonify({"error": "Invalid query embedding format"}), 400
        
        # Check if dimensions match
        if len(query_vector) != len(embeddings[0]):
            logger.error(f"Query embedding dimension ({len(query_vector)}) does not match database embeddings ({len(embeddings[0])})")
            return jsonify({"error": "Query embedding dimension mismatch"}), 400
        
        # Calculate similarities
        logger.info(f"Calculating similarities for {len(embeddings)} documents")
        similarity_start = time.time()
        similarities = [cosine_similarity(query_vector, emb) for emb in embeddings]
        similarity_time = time.time() - similarity_start
        logger.info(f"Similarity calculation completed in {similarity_time:.2f} seconds")
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = documents[idx].copy()
            score = float(similarities[idx])
            
            # Skip results with very low similarity
            if score < 0.1:  # Threshold can be adjusted
                logger.debug(f"Skipping low similarity result: {score:.4f}")
                continue
                
            doc['score'] = score
            
            # Remove embedding from result to reduce response size
            if 'embedding' in doc:
                del doc['embedding']
                
            results.append(doc)
        
        query_time = time.time() - start_time
        logger.info(f"Query returned {len(results)} results in {query_time:.2f} seconds")
        
        return jsonify({
            "results": results,
            "query_time": query_time,
            "total_documents_searched": len(documents)
        })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats() -> Response:
    """
    Get statistics about the vector database.
    
    Returns:
        Flask response with JSON data containing database statistics
    """
    try:
        start_time = time.time()
        
        if not documents:
            logger.info("Stats requested for empty database")
            return jsonify({
                "total_documents": 0,
                "extensions": {},
                "database_file_exists": os.path.exists(DB_FILE)
            })
        
        # Calculate statistics
        extensions = {}
        paths = set()
        total_content_size = 0
        chunk_counts = {}
        
        for doc in documents:
            # Count by extension
            ext = doc.get('extension', 'unknown')
            extensions[ext] = extensions.get(ext, 0) + 1
            
            # Count unique paths
            path = doc.get('path', '')
            if path:
                paths.add(path)
                
            # Sum content size
            content = doc.get('content', '')
            total_content_size += len(content)
            
            # Count chunks per file
            if path:
                chunk_index = doc.get('chunk_index', 0)
                total_chunks = doc.get('total_chunks', 1)
                chunk_counts[path] = total_chunks
        
        # Calculate average chunks per file
        avg_chunks = sum(chunk_counts.values()) / len(chunk_counts) if chunk_counts else 0
        
        stats_time = time.time() - start_time
        logger.info(f"Generated stats in {stats_time:.2f} seconds")
        
        return jsonify({
            "total_documents": len(documents),
            "unique_files": len(paths),
            "extensions": extensions,
            "total_content_size_kb": total_content_size / 1024,
            "avg_chunks_per_file": avg_chunks,
            "database_file_exists": os.path.exists(DB_FILE),
            "database_file_size_mb": os.path.getsize(DB_FILE) / (1024 * 1024) if os.path.exists(DB_FILE) else 0
        })
    except Exception as e:
        logger.error(f"Error generating stats: {str(e)}")
        return jsonify({
            "error": str(e),
            "total_documents": len(documents)
        }), 500

@app.route('/clear', methods=['POST'])
def clear() -> Response:
    """
    Clear the vector database.
    
    Returns:
        Flask response with JSON data
    """
    global documents, embeddings
    
    try:
        logger.info("Clearing vector database")
        
        # Create a backup before clearing if database exists
        if os.path.exists(DB_FILE) and documents:
            os.makedirs(BACKUP_DIR, exist_ok=True)
            backup_file = BACKUP_DIR / f"documents_before_clear_{int(time.time())}.json.bak"
            try:
                import shutil
                shutil.copy2(DB_FILE, backup_file)
                logger.info(f"Created backup before clearing at {backup_file}")
            except Exception as e:
                logger.warning(f"Failed to create backup before clearing: {str(e)}")
        
        # Store counts for logging
        doc_count = len(documents)
        
        # Clear the database
        documents.clear()
        embeddings.clear()
        
        # Remove the database file
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            logger.info(f"Removed database file: {DB_FILE}")
        
        logger.info(f"Database cleared: removed {doc_count} documents")
        
        return jsonify({
            "success": True,
            "message": f"Database cleared: removed {doc_count} documents",
            "backup_created": os.path.exists(backup_file) if 'backup_file' in locals() else False
        })
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check() -> Response:
    """
    Health check endpoint.
    
    Returns:
        Flask response with health status
    """
    return jsonify({
        "status": "healthy",
        "documents_count": len(documents),
        "embeddings_count": len(embeddings),
        "database_file_exists": os.path.exists(DB_FILE)
    })

def main():
    parser = argparse.ArgumentParser(description='Run a mock vector database API server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Load existing database if available
    doc_count, emb_count = load_database()
    
    logger.info(f"Starting mock vector database API server at http://{args.host}:{args.port}")
    logger.info(f"Loaded {doc_count} documents and {emb_count} embeddings")
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
