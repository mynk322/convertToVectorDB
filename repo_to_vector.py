import os
import sys
import git
import json
import time
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('repo_to_vector.log')
    ]
)
logger = logging.getLogger(__name__)

class RepoToVectorConverter:
    def __init__(self, 
                 repo_url: str, 
                 vector_db_api_url: str,
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 file_extensions: List[str] = None,
                 output_dir: str = "repo_data"):
        """
        Initialize the converter with the repository URL and vector database API URL.
        
        Args:
            repo_url: URL of the GitHub repository to clone
            vector_db_api_url: URL of the local vector database API
            model_name: Name of the sentence-transformer model to use for embeddings
            chunk_size: Size of text chunks for processing
            file_extensions: List of file extensions to process (default: common code files)
            output_dir: Directory to store temporary data
        """
        # Validate inputs
        if not repo_url or not repo_url.strip():
            raise ValueError("Repository URL cannot be empty")
        
        if not vector_db_api_url or not vector_db_api_url.strip():
            raise ValueError("Vector database API URL cannot be empty")
            
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        self.repo_url = repo_url
        self.vector_db_api_url = vector_db_api_url
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        self.metrics = {
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_created": 0,
            "embedding_time": 0,
            "total_time": 0
        }
        
        # Default file extensions to process if none provided
        self.file_extensions = file_extensions or [
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", 
            ".hpp", ".cs", ".go", ".rb", ".php", ".html", ".css", ".md", ".txt",
            ".json", ".yml", ".yaml", ".toml", ".rs", ".swift", ".kt", ".scala"
        ]
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def check_api_connection(self) -> bool:
        """
        Check if the vector database API is accessible.
        
        Returns:
            True if the API is accessible, False otherwise
        """
        try:
            logger.info(f"Checking connection to vector database API at {self.vector_db_api_url}")
            
            # Try different endpoint paths for flexibility
            endpoints = ["/stats", "stats", "/health", "health"]
            
            for endpoint in endpoints:
                try:
                    # Construct the URL carefully to handle different formats
                    url = self.vector_db_api_url
                    if not url.endswith('/') and not endpoint.startswith('/'):
                        url += '/'
                    url += endpoint
                    
                    logger.debug(f"Trying to connect to: {url}")
                    response = requests.get(url, timeout=5)
                    
                    if response.status_code == 200:
                        logger.info(f"Successfully connected to vector database API at {url}")
                        return True
                except requests.exceptions.RequestException:
                    continue
            
            # If we get here, all connection attempts failed
            logger.error(f"Failed to connect to vector database API at {self.vector_db_api_url}")
            logger.error("Please ensure the vector database API server is running and the URL is correct.")
            logger.error("You can start the mock API server with: python mock_vector_db_api.py")
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to vector database API: {str(e)}")
            logger.error("Please ensure the vector database API server is running and the URL is correct.")
            logger.error("You can start the mock API server with: python mock_vector_db_api.py")
            return False
            
    def clone_repository(self, target_dir: Optional[str] = None) -> str:
        """
        Clone the GitHub repository to a local directory.
        
        Args:
            target_dir: Optional directory to clone into (default: repo name)
            
        Returns:
            Path to the cloned repository
        """
        # Validate repo URL format
        if not (self.repo_url.startswith("https://") or self.repo_url.startswith("git@")):
            logger.warning(f"Repository URL format may be invalid: {self.repo_url}")
        if target_dir is None:
            # Extract repo name from URL
            repo_name = self.repo_url.split('/')[-1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
            target_dir = self.output_dir / repo_name
        else:
            target_dir = Path(target_dir)
        
        # Clone the repository if it doesn't exist
        if not os.path.exists(target_dir):
            logger.info(f"Cloning repository {self.repo_url} to {target_dir}")
            clone_start = time.time()
            try:
                git.Repo.clone_from(self.repo_url, target_dir)
                clone_time = time.time() - clone_start
                logger.info(f"Repository cloned successfully in {clone_time:.2f} seconds")
            except git.GitCommandError as e:
                logger.error(f"Git clone failed: {str(e)}")
                raise
        else:
            logger.info(f"Repository already exists at {target_dir}")
            # Check if it's a valid git repository
            try:
                repo = git.Repo(target_dir)
                logger.info(f"Found valid git repository at {target_dir}")
            except git.InvalidGitRepositoryError:
                logger.warning(f"Directory exists but is not a valid git repository: {target_dir}")
            
        return str(target_dir)
    
    def process_files(self, repo_path: str) -> List[Dict[str, Any]]:
        """
        Process files in the repository and extract content.
        
        Args:
            repo_path: Path to the cloned repository
            
        Returns:
            List of dictionaries containing file information and content
        """
        if not os.path.exists(repo_path):
            logger.error(f"Repository path does not exist: {repo_path}")
            return []
            
        logger.info(f"Processing files in {repo_path}")
        processed_files = []
        total_files = 0
        skipped_files = 0
        total_size = 0
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                total_files += 1
                file_path = os.path.join(root, file)
                
                # Skip hidden files and directories
                if any(part.startswith('.') for part in file_path.split(os.sep)):
                    skipped_files += 1
                    continue
                
                # Check if file extension is in the list to process
                _, ext = os.path.splitext(file)
                if ext not in self.file_extensions:
                    skipped_files += 1
                    logger.debug(f"Skipping file with unsupported extension: {file_path}")
                    continue
                
                # Skip files larger than 10MB to prevent memory issues
                file_size = os.path.getsize(file_path)
                if file_size > 10 * 1024 * 1024:  # 10MB
                    skipped_files += 1
                    logger.warning(f"Skipping large file ({file_size / 1024 / 1024:.2f} MB): {file_path}")
                    continue
                
                # Get relative path from repo root
                rel_path = os.path.relpath(file_path, repo_path)
                
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Skip empty files
                    if not content.strip():
                        skipped_files += 1
                        logger.debug(f"Skipping empty file: {file_path}")
                        continue
                    
                    file_size = len(content)
                    total_size += file_size
                    
                    processed_files.append({
                        'path': rel_path,
                        'content': content,
                        'extension': ext,
                        'size': file_size
                    })
                    
                    # Log progress periodically
                    if len(processed_files) % 100 == 0:
                        logger.info(f"Processed {len(processed_files)} files so far...")
                    
                except UnicodeDecodeError:
                    skipped_files += 1
                    logger.warning(f"Skipping binary file: {file_path}")
                    continue
                except Exception as e:
                    skipped_files += 1
                    logger.warning(f"Error processing file {file_path}: {str(e)}")
        
        self.metrics["files_processed"] = len(processed_files)
        self.metrics["files_skipped"] = skipped_files
        
        logger.info(f"Processed {len(processed_files)} files, skipped {skipped_files} files")
        logger.info(f"Total content size: {total_size / 1024 / 1024:.2f} MB")
        return processed_files
    
    def chunk_text(self, text: str, overlap: int = 200) -> List[str]:
        """
        Split text into chunks of specified size with overlap.
        
        Args:
            text: Text to split into chunks
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            logger.warning("Attempted to chunk empty text")
            return []
            
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a good breaking point (newline or space)
            if end < len(text):
                # Look for newline first
                newline_pos = text.rfind('\n', start, end)
                if newline_pos > start + self.chunk_size // 2:
                    end = newline_pos + 1
                else:
                    # Look for space
                    space_pos = text.rfind(' ', start + self.chunk_size // 2, end)
                    if space_pos != -1:
                        end = space_pos + 1
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Log chunk statistics
            logger.debug(f"Created chunk: {len(chunk)} characters, {chunk.count(' ')} words")
            
            start = end - overlap if end - overlap > start else end
            
            # Break if we've reached the end
            if start >= len(text):
                break
        
        self.metrics["chunks_created"] += len(chunks)
        logger.debug(f"Split text into {len(chunks)} chunks with {overlap} character overlap")
        return chunks
    
    def generate_embeddings(self, processed_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for the processed files.
        
        Args:
            processed_files: List of processed file dictionaries
            
        Returns:
            List of dictionaries with file information and embeddings
        """
        if not processed_files:
            logger.warning("No files to generate embeddings for")
            return []
            
        logger.info(f"Generating embeddings for {len(processed_files)} processed files")
        documents_with_embeddings = []
        total_chunks = 0
        embedding_start_time = time.time()
        
        # Process files in batches to show progress
        batch_size = max(1, len(processed_files) // 10)  # Show progress in ~10 steps
        
        for i, file_info in enumerate(processed_files):
            content = file_info['content']
            chunks = self.chunk_text(content)
            total_chunks += len(chunks)
            
            # Show progress periodically
            if (i + 1) % batch_size == 0 or i == len(processed_files) - 1:
                progress = (i + 1) / len(processed_files) * 100
                logger.info(f"Embedding progress: {progress:.1f}% ({i + 1}/{len(processed_files)} files)")
            
            try:
                for i, chunk in enumerate(chunks):
                    # Generate embedding for the chunk
                    embedding = self.model.encode(chunk, convert_to_tensor=False).tolist()
                    
                    # Validate embedding dimensions
                    if not embedding or len(embedding) == 0:
                        logger.warning(f"Empty embedding generated for {file_info['path']}, chunk {i}")
                        continue
                    
                    # Create document with metadata and embedding
                    document = {
                        'path': file_info['path'],
                        'extension': file_info['extension'],
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'content': chunk,
                        'embedding': embedding,
                        'timestamp': time.time()
                    }
                    
                    documents_with_embeddings.append(document)
            except Exception as e:
                logger.error(f"Error generating embedding for {file_info['path']}: {str(e)}")
        
        embedding_time = time.time() - embedding_start_time
        self.metrics["embedding_time"] = embedding_time
        
        logger.info(f"Generated {len(documents_with_embeddings)} embeddings from {total_chunks} chunks")
        logger.info(f"Embedding generation took {embedding_time:.2f} seconds")
        
        if len(documents_with_embeddings) < total_chunks:
            logger.warning(f"Some chunks failed to generate embeddings: {total_chunks - len(documents_with_embeddings)} failures")
            
        return documents_with_embeddings
    
    def store_in_vector_db(self, documents: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Store the documents with embeddings in the vector database.
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            Tuple of (success boolean, response data)
        """
        if not documents:
            logger.warning("No documents to store in vector database")
            return False, {"error": "No documents to store"}
            
        logger.info(f"Storing {len(documents)} documents in vector database at {self.vector_db_api_url}")
        
        # Store documents in batches to avoid overwhelming the API
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        success = True
        response_data = {}
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Sending batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                # Make a POST request to the vector database API
                start_time = time.time()
                response = requests.post(
                    f"{self.vector_db_api_url}/add_documents",
                    json={'documents': batch},
                    timeout=30  # 30 second timeout
                )
                
                request_time = time.time() - start_time
                
                if response.status_code == 200:
                    logger.info(f"Successfully stored batch {batch_num}/{total_batches} in {request_time:.2f} seconds")
                    if i == 0:  # Save the first response
                        try:
                            response_data = response.json()
                        except:
                            response_data = {"message": "Success but could not parse response"}
                else:
                    logger.error(f"Failed to store batch {batch_num}: {response.status_code} - {response.text}")
                    success = False
                    try:
                        response_data = response.json()
                    except:
                        response_data = {"error": f"HTTP {response.status_code}: {response.text}"}
                    break
                    
            except requests.exceptions.Timeout:
                logger.error(f"Timeout storing batch {batch_num} in vector database")
                success = False
                response_data = {"error": "Request timeout"}
                break
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error storing batch {batch_num} in vector database")
                success = False
                response_data = {"error": "Connection error"}
                break
            except Exception as e:
                logger.error(f"Error storing batch {batch_num} in vector database: {str(e)}")
                success = False
                response_data = {"error": str(e)}
                break
        
        if success:
            logger.info(f"Successfully stored all {len(documents)} documents in vector database")
        
        return success, response_data
    
    def log_metrics(self) -> None:
        """Log performance metrics for the conversion process."""
        self.metrics["total_time"] = time.time() - self.start_time
        
        logger.info("=== Performance Metrics ===")
        logger.info(f"Total time: {self.metrics['total_time']:.2f} seconds")
        logger.info(f"Files processed: {self.metrics['files_processed']}")
        logger.info(f"Files skipped: {self.metrics['files_skipped']}")
        logger.info(f"Chunks created: {self.metrics['chunks_created']}")
        logger.info(f"Embedding time: {self.metrics['embedding_time']:.2f} seconds")
        
        if self.metrics['files_processed'] > 0:
            logger.info(f"Average time per file: {self.metrics['total_time'] / self.metrics['files_processed']:.2f} seconds")
        
        if self.metrics['chunks_created'] > 0:
            logger.info(f"Average time per chunk: {self.metrics['embedding_time'] / self.metrics['chunks_created']:.2f} seconds")
    
    def run(self) -> bool:
        """
        Run the full conversion process.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting conversion of {self.repo_url} to vector database")
            
            # Check API connection first
            if not self.check_api_connection():
                logger.error("Cannot proceed without vector database API connection")
                logger.error(f"Please ensure the API server is running at: {self.vector_db_api_url}")
                logger.error("You can start the mock API server with: python mock_vector_db_api.py --host 127.0.0.1 --port <port>")
                return False
            
            # Clone the repository
            repo_path = self.clone_repository()
            
            # Process the files
            processed_files = self.process_files(repo_path)
            if not processed_files:
                logger.warning("No files were processed, nothing to convert")
                return False
            
            # Generate embeddings
            documents_with_embeddings = self.generate_embeddings(processed_files)
            if not documents_with_embeddings:
                logger.warning("No embeddings were generated, nothing to store")
                return False
            
            # Store in vector database
            success, response = self.store_in_vector_db(documents_with_embeddings)
            
            # Log metrics
            self.log_metrics()
            
            if success:
                logger.info("Repository successfully converted to vector database")
                # Log response data if available
                if response and isinstance(response, dict):
                    if "total_documents" in response:
                        logger.info(f"Total documents in database: {response['total_documents']}")
            else:
                logger.error("Failed to convert repository to vector database")
            
            return success
            
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
            self.log_metrics()
            return False
        except Exception as e:
            logger.error(f"Error in conversion process: {str(e)}", exc_info=True)
            self.log_metrics()
            return False


def main():
    parser = argparse.ArgumentParser(description='Convert GitHub repository to vector database')
    parser.add_argument('--repo-url', required=True, help='URL of the GitHub repository')
    parser.add_argument('--api-url', required=True, help='URL of the vector database API')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Name of the sentence-transformer model')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Size of text chunks')
    parser.add_argument('--output-dir', default='repo_data', help='Directory to store temporary data')
    parser.add_argument('--extensions', nargs='+', help='File extensions to process')
    
    args = parser.parse_args()
    
    converter = RepoToVectorConverter(
        repo_url=args.repo_url,
        vector_db_api_url=args.api_url,
        model_name=args.model,
        chunk_size=args.chunk_size,
        file_extensions=args.extensions,
        output_dir=args.output_dir
    )
    
    success = converter.run()
    
    if success:
        logger.info("Repository successfully converted to vector database")
        return 0
    else:
        logger.error("Failed to convert repository to vector database")
        return 1


if __name__ == "__main__":
    sys.exit(main())
