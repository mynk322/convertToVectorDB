# Example Queries for Vector Database

This document provides example queries that demonstrate how to use the vector database to extract useful information from a repository.

## Setup

First, make sure you have converted a repository to the vector database:

```bash
./run_vector_db.py convert --repo-url <github_repo_url>
```

Then you can run queries using either:

```bash
./run_vector_db.py query --query "your query here"
```

Or directly:

```bash
python query_vector_db.py --query "your query here" --api-url http://127.0.0.1:5000
```

## Example Queries

### 1. Estimating Implementation Time

To find information about how much time a software developer might take to implement a feature, you can use queries like:

```bash
./run_vector_db.py query --query "implementation time estimates for feature development"
```

```bash
./run_vector_db.py query --query "how long would it take to implement a user authentication system"
```

```bash
./run_vector_db.py query --query "time required to develop a REST API endpoint"
```

These queries will search the codebase for:
- Comments discussing time estimates
- Commit messages mentioning implementation duration
- Project planning documents with timelines
- Issues or tickets with time estimates
- Documentation about development processes and timelines

### 2. Finding Technical Requirements

To understand the technical requirements for implementing a feature:

```bash
./run_vector_db.py query --query "technical requirements for implementing search functionality"
```

### 3. Identifying Dependencies

To find dependencies needed for a specific feature:

```bash
./run_vector_db.py query --query "dependencies required for implementing payment processing"
```

### 4. Locating Similar Features

To find similar features that have already been implemented:

```bash
./run_vector_db.py query --query "existing implementation of user profile management"
```

### 5. Finding Code Examples

To find code examples for specific functionality:

```bash
./run_vector_db.py query --query "examples of API authentication implementation"
```

### 6. Identifying Potential Challenges

To identify potential challenges in implementing a feature:

```bash
./run_vector_db.py query --query "challenges in implementing real-time notifications"
```

## Advanced Query Techniques

For more accurate results, try to:

1. **Be specific**: Include technical details and context in your query
2. **Use technical terminology**: Use terms that would appear in code or documentation
3. **Combine approaches**: Run multiple related queries to gather comprehensive information
4. **Refine iteratively**: Start with a broad query, then refine based on initial results

## Interpreting Results

When analyzing results to estimate implementation time:

1. Look for explicit time mentions (e.g., "took 2 weeks to implement")
2. Examine complexity indicators in the code
3. Check for mentions of challenges or blockers
4. Look for similar features and their implementation history
5. Consider the context of the repository (team size, technology stack)

Remember that the vector database provides semantic search, so it will return results based on conceptual similarity rather than just keyword matching. This means you'll get relevant information even if it doesn't contain the exact words from your query.
