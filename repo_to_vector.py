import os
import sys
import git
import json
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        self.repo_url = repo_url
        self.vector_db_api_url = vector_db_api_url
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir)
        
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
        
    def clone_repository(self, target_dir: Optional[str] = None) -> str:
        """
        Clone the GitHub repository to a local directory.
        
        Args:
            target_dir: Optional directory to clone into (default: repo name)
            
        Returns:
            Path to the cloned repository
        """
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
            git.Repo.clone_from(self.repo_url, target_dir)
        else:
            logger.info(f"Repository already exists at {target_dir}")
            
        return str(target_dir)
    
    def process_files(self, repo_path: str) -> List[Dict[str, Any]]:
        """
        Process files in the repository and extract content.
        
        Args:
            repo_path: Path to the cloned repository
            
        Returns:
            List of dictionaries containing file information and content
        """
        logger.info(f"Processing files in {repo_path}")
        processed_files = []
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip hidden files and directories
                if any(part.startswith('.') for part in file_path.split(os.sep)):
                    continue
                
                # Check if file extension is in the list to process
                _, ext = os.path.splitext(file)
                if ext not in self.file_extensions:
                    continue
                
                # Get relative path from repo root
                rel_path = os.path.relpath(file_path, repo_path)
                
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Skip empty files
                    if not content.strip():
                        continue
                    
                    processed_files.append({
                        'path': rel_path,
                        'content': content,
                        'extension': ext,
                        'size': len(content)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {str(e)}")
        
        logger.info(f"Processed {len(processed_files)} files")
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
            
            chunks.append(text[start:end])
            start = end - overlap if end - overlap > start else end
            
            # Break if we've reached the end
            if start >= len(text):
                break
        
        return chunks
    
    def generate_embeddings(self, processed_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for the processed files.
        
        Args:
            processed_files: List of processed file dictionaries
            
        Returns:
            List of dictionaries with file information and embeddings
        """
        logger.info("Generating embeddings for processed files")
        documents_with_embeddings = []
        
        for file_info in processed_files:
            content = file_info['content']
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                # Generate embedding for the chunk
                embedding = self.model.encode(chunk, convert_to_tensor=False).tolist()
                
                # Create document with metadata and embedding
                document = {
                    'path': file_info['path'],
                    'extension': file_info['extension'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'content': chunk,
                    'embedding': embedding
                }
                
                documents_with_embeddings.append(document)
        
        logger.info(f"Generated embeddings for {len(documents_with_embeddings)} chunks")
        return documents_with_embeddings
    
    def store_in_vector_db(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Store the documents with embeddings in the vector database.
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Storing {len(documents)} documents in vector database")
        
        try:
            # Make a POST request to the vector database API
            response = requests.post(
                f"{self.vector_db_api_url}/add_documents",
                json={'documents': documents}
            )
            
            if response.status_code == 200:
                logger.info("Successfully stored documents in vector database")
                return True
            else:
                logger.error(f"Failed to store documents: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing documents in vector database: {str(e)}")
            return False
    
    def run(self) -> bool:
        """
        Run the full conversion process.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clone the repository
            repo_path = self.clone_repository()
            
            # Process the files
            processed_files = self.process_files(repo_path)
            
            # Generate embeddings
            documents_with_embeddings = self.generate_embeddings(processed_files)
            
            # Store in vector database
            success = self.store_in_vector_db(documents_with_embeddings)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in conversion process: {str(e)}")
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
