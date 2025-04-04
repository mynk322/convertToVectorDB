# GitHub Repository to Vector Database Converter

This tool converts a GitHub repository into vector embeddings and stores them in a local vector database for efficient querying.

## Prerequisites

- Python 3.7+
- A local vector database with an API endpoint for storing documents
- Git installed on your system

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can use the system in two ways:

### 1. Using the Wrapper Script (Recommended)

The wrapper script provides a simple interface to run all the components with consistent configuration:

```bash
# Start the vector database server
python run_vector_db.py start

# Convert a GitHub repository to vector database
python run_vector_db.py convert --repo-url <github_repo_url>

# Query the vector database
python run_vector_db.py query --query "your search query"

# Show database statistics
python run_vector_db.py stats

# Run the full demo
python run_vector_db.py demo --repo-url <github_repo_url> --query "your search query"
```

### 2. Using Individual Scripts

Alternatively, you can run each component separately:

```bash
# Start the mock vector database API server
python mock_vector_db_api.py --host 127.0.0.1 --port 5000

# Convert a GitHub repository to vector database
python repo_to_vector.py --repo-url <github_repo_url> --api-url http://127.0.0.1:5000

# Query the vector database
python query_vector_db.py --query "your search query" --api-url http://127.0.0.1:5000
```

### 3. Using the Demo Script

You can also run the full demo with a single command:

```bash
./demo.sh --repo-url <github_repo_url> --query "your search query"
```

### Required Arguments

- `--repo-url`: URL of the GitHub repository to convert
- `--api-url`: URL of your local vector database API endpoint (default: http://127.0.0.1:5000)

### Optional Arguments

- `--model`: Name of the sentence-transformer model to use (default: "all-MiniLM-L6-v2")
- `--chunk-size`: Size of text chunks for processing (default: 1000)
- `--output-dir`: Directory to store temporary data (default: "repo_data")
- `--extensions`: List of file extensions to process (default: common code file extensions)
- `--port`: Port for the mock API server (default: 5000)
- `--host`: Host for the mock API server (default: 127.0.0.1)
- `--debug`: Enable debug mode for more detailed logging
- `--verbose`: Enable verbose output

## Example

```bash
# Using the wrapper script
python run_vector_db.py demo --repo-url https://github.com/username/repository --query "search query"

# Using individual scripts
python repo_to_vector.py --repo-url https://github.com/username/repository --api-url http://127.0.0.1:5000
python query_vector_db.py --query "search query" --api-url http://127.0.0.1:5000
```

## Example Queries and Time Estimation

For a comprehensive list of example queries and how to use the vector database effectively, see [example_queries.md](example_queries.md). This includes queries for:

- Estimating implementation time for features
- Finding technical requirements
- Identifying dependencies
- Locating similar features
- Finding code examples
- Identifying potential challenges

### Implementation Time Estimation

This project includes tools to help estimate implementation time for new features based on the repository context:

- [time_estimation_guide.md](time_estimation_guide.md): A detailed guide on how to use the vector database with LLMs to generate accurate time estimates
- [estimate_time.py](estimate_time.py): A script that automates the process of generating implementation time estimates

To estimate implementation time for a feature:

```bash
# 1. Convert a repository to vector database
./run_vector_db.py convert --repo-url <github_repo_url>

# 2. Create a function documentation file
echo "Implement user authentication with OAuth and 2FA" > function_doc.txt

# 3. Query the vector database
./run_vector_db.py query --query "$(cat function_doc.txt)" --top-k 10 > vector_results.txt

# 4. Generate time estimate
./estimate_time.py --function-doc function_doc.txt --vector-results vector_results.txt --output time_estimate_report.md
```

This will generate a detailed time estimate report based on the repository context and the function documentation.

## How It Works

1. The script clones the specified GitHub repository
2. It processes all files with the specified extensions
3. The content is split into chunks with overlap for better context preservation
4. Each chunk is converted to a vector embedding using the specified model
5. The embeddings and associated metadata are sent to your local vector database API

## API Expectations

The script expects your vector database API to have an endpoint at:
`<api-url>/add_documents`

The API should accept a POST request with a JSON payload in the following format:

```json
{
  "documents": [
    {
      "path": "relative/path/to/file.py",
      "extension": ".py",
      "chunk_index": 0,
      "total_chunks": 3,
      "content": "The actual text content of this chunk",
      "embedding": [0.1, 0.2, ..., 0.768]
    },
    ...
  ]
}
```

## Customization

You can modify the script to adapt it to your specific vector database API requirements or to add additional processing steps for the repository files.
