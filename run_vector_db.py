#!/usr/bin/env python3
"""
Vector Database Wrapper Script

This script provides a simple interface to run the vector database tools
with consistent configuration.
"""

import argparse
import os
import subprocess
import sys
import time
import signal
import requests
from pathlib import Path

# Default configuration
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 5000
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OUTPUT_DIR = "repo_data"

def check_server_running(host, port):
    """Check if the vector database API server is running."""
    url = f"http://{host}:{port}"
    endpoints = ["/health", "/stats", "health", "stats"]
    
    for endpoint in endpoints:
        try:
            full_url = f"{url}/{endpoint}" if not endpoint.startswith('/') else f"{url}{endpoint}"
            response = requests.get(full_url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
    
    return False

def start_server(host, port, debug=False):
    """Start the vector database API server."""
    print(f"Starting vector database API server at http://{host}:{port}")
    
    # Build the command
    cmd = ["python", "mock_vector_db_api.py", "--host", host, "--port", str(port)]
    if debug:
        cmd.append("--debug")
    
    # Start the server process
    server_process = subprocess.Popen(cmd)
    
    # Wait for the server to start
    print("Waiting for server to start...")
    for i in range(10):
        if check_server_running(host, port):
            print(f"Server started successfully at http://{host}:{port}")
            return server_process
        time.sleep(1)
        print(f"Waiting... ({i+1}/10)")
    
    # If we get here, the server failed to start
    print("Error: Server failed to start")
    server_process.terminate()
    return None

def convert_repository(repo_url, api_host, api_port, model, chunk_size, output_dir):
    """Convert a GitHub repository to vector database."""
    print(f"Converting repository: {repo_url}")
    
    api_url = f"http://{api_host}:{api_port}"
    cmd = [
        "python", "repo_to_vector.py",
        "--repo-url", repo_url,
        "--api-url", api_url,
        "--model", model,
        "--chunk-size", str(chunk_size),
        "--output-dir", output_dir
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def query_database(query_text, api_host, api_port, model, top_k=5, verbose=False):
    """Query the vector database."""
    print(f"Querying database with: '{query_text}'")
    
    api_url = f"http://{api_host}:{api_port}"
    cmd = [
        "python", "query_vector_db.py",
        "--query", query_text,
        "--api-url", api_url,
        "--model", model,
        "--top-k", str(top_k)
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def show_stats(api_host, api_port):
    """Show vector database statistics."""
    api_url = f"http://{api_host}:{api_port}"
    try:
        response = requests.get(f"{api_url}/stats")
        if response.status_code == 200:
            import json
            stats = response.json()
            print("\nVector Database Statistics:")
            print(json.dumps(stats, indent=2))
            return True
    except:
        print("Error: Failed to get database statistics")
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Vector Database Wrapper Script")
    
    # Main command options
    parser.add_argument("command", choices=["start", "convert", "query", "stats", "demo"],
                        help="Command to execute")
    
    # Server options
    parser.add_argument("--host", default=DEFAULT_API_HOST,
                        help=f"API server host (default: {DEFAULT_API_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT,
                        help=f"API server port (default: {DEFAULT_API_PORT})")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode for the server")
    
    # Repository conversion options
    parser.add_argument("--repo-url", 
                        help="URL of the GitHub repository to convert")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Name of the sentence-transformer model (default: {DEFAULT_MODEL})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Size of text chunks for processing (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to store temporary data (default: {DEFAULT_OUTPUT_DIR})")
    
    # Query options
    parser.add_argument("--query", 
                        help="Text to search for")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nInterrupted by user. Exiting...")
        if server_process:
            print("Stopping server...")
            server_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    server_process = None
    
    try:
        if args.command == "start":
            # Start the server
            server_process = start_server(args.host, args.port, args.debug)
            if not server_process:
                return 1
            
            print("Server is running. Press Ctrl+C to stop.")
            server_process.wait()
            
        elif args.command == "convert":
            if not args.repo_url:
                print("Error: --repo-url is required for the convert command")
                return 1
            
            # Check if server is running
            if not check_server_running(args.host, args.port):
                print(f"No server running at http://{args.host}:{args.port}")
                print("Starting server...")
                server_process = start_server(args.host, args.port, args.debug)
                if not server_process:
                    return 1
            
            # Convert repository
            success = convert_repository(
                args.repo_url, args.host, args.port, 
                args.model, args.chunk_size, args.output_dir
            )
            
            if not success:
                print("Error: Repository conversion failed")
                return 1
            
        elif args.command == "query":
            if not args.query:
                print("Error: --query is required for the query command")
                return 1
            
            # Check if server is running
            if not check_server_running(args.host, args.port):
                print(f"No server running at http://{args.host}:{args.port}")
                print("Starting server...")
                server_process = start_server(args.host, args.port, args.debug)
                if not server_process:
                    return 1
            
            # Query database
            success = query_database(
                args.query, args.host, args.port, 
                args.model, args.top_k, args.verbose
            )
            
            if not success:
                print("Error: Query failed")
                return 1
            
        elif args.command == "stats":
            # Check if server is running
            if not check_server_running(args.host, args.port):
                print(f"No server running at http://{args.host}:{args.port}")
                print("Starting server...")
                server_process = start_server(args.host, args.port, args.debug)
                if not server_process:
                    return 1
            
            # Show stats
            success = show_stats(args.host, args.port)
            if not success:
                print("Error: Failed to get statistics")
                return 1
            
        elif args.command == "demo":
            # Run the full demo
            
            # Check if server is running
            server_started = False
            if not check_server_running(args.host, args.port):
                print(f"No server running at http://{args.host}:{args.port}")
                print("Starting server...")
                server_process = start_server(args.host, args.port, args.debug)
                if not server_process:
                    return 1
                server_started = True
            else:
                print(f"Using existing server at http://{args.host}:{args.port}")
            
            # Use default repo if not specified
            repo_url = args.repo_url or "https://github.com/facebookresearch/faiss"
            
            # Convert repository
            print("\n=== Step 1: Converting Repository ===")
            success = convert_repository(
                repo_url, args.host, args.port, 
                args.model, args.chunk_size, args.output_dir
            )
            
            if not success:
                print("Error: Repository conversion failed")
                return 1
            
            # Query database
            print("\n=== Step 2: Querying Database ===")
            query_text = args.query or "vector similarity search algorithm"
            success = query_database(
                query_text, args.host, args.port, 
                args.model, args.top_k, args.verbose
            )
            
            if not success:
                print("Error: Query failed")
                return 1
            
            # Show stats
            print("\n=== Step 3: Database Statistics ===")
            show_stats(args.host, args.port)
            
            print("\nDemo completed successfully!")
        
    finally:
        # Clean up server process if we started it
        if server_process and args.command != "start":
            print("Stopping server...")
            server_process.terminate()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
