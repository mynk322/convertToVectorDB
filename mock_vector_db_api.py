import os
import json
import argparse
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path

app = Flask(__name__)

# In-memory storage for documents and embeddings
documents = []
embeddings = []

# Directory to save/load the database
DB_DIR = Path("vector_db_data")
DB_FILE = DB_DIR / "documents.json"

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def save_database():
    """Save the current database to disk."""
    os.makedirs(DB_DIR, exist_ok=True)
    with open(DB_FILE, 'w') as f:
        json.dump(documents, f)
    print(f"Database saved to {DB_FILE}")

def load_database():
    """Load the database from disk if it exists."""
    global documents, embeddings
    
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            documents = json.load(f)
            
        # Extract embeddings from documents
        embeddings = [np.array(doc['embedding']) for doc in documents]
        print(f"Loaded {len(documents)} documents from {DB_FILE}")
    else:
        print("No existing database found.")

@app.route('/add_documents', methods=['POST'])
def add_documents():
    """Add documents to the vector database."""
    global documents, embeddings
    
    data = request.json
    new_docs = data.get('documents', [])
    
    if not new_docs:
        return jsonify({"error": "No documents provided"}), 400
    
    # Add documents to the database
    documents.extend(new_docs)
    
    # Extract embeddings
    for doc in new_docs:
        embeddings.append(np.array(doc['embedding']))
    
    # Save the updated database
    save_database()
    
    return jsonify({
        "success": True,
        "message": f"Added {len(new_docs)} documents to the database",
        "total_documents": len(documents)
    })

@app.route('/query', methods=['POST'])
def query():
    """Query the vector database with an embedding."""
    data = request.json
    query_embedding = data.get('query_embedding')
    top_k = data.get('top_k', 5)
    
    if not query_embedding:
        return jsonify({"error": "No query embedding provided"}), 400
    
    if not documents:
        return jsonify({"results": []}), 200
    
    # Convert query embedding to numpy array
    query_vector = np.array(query_embedding)
    
    # Calculate similarities
    similarities = [cosine_similarity(query_vector, emb) for emb in embeddings]
    
    # Get top-k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        doc = documents[idx].copy()
        doc['score'] = float(similarities[idx])
        results.append(doc)
    
    return jsonify({"results": results})

@app.route('/stats', methods=['GET'])
def stats():
    """Get statistics about the vector database."""
    extensions = {}
    for doc in documents:
        ext = doc.get('extension', 'unknown')
        extensions[ext] = extensions.get(ext, 0) + 1
    
    return jsonify({
        "total_documents": len(documents),
        "extensions": extensions
    })

@app.route('/clear', methods=['POST'])
def clear():
    """Clear the vector database."""
    global documents, embeddings
    documents = []
    embeddings = []
    
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    
    return jsonify({
        "success": True,
        "message": "Database cleared"
    })

def main():
    parser = argparse.ArgumentParser(description='Run a mock vector database API server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    
    args = parser.parse_args()
    
    # Load existing database if available
    load_database()
    
    print(f"Starting mock vector database API server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
