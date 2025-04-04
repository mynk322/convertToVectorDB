#!/usr/bin/env python3
"""
Time Estimation Script

This script uses the vector database results and an LLM to generate
implementation time estimates and complexity analysis.
"""

import argparse
import json
import os
import sys
import requests
from pathlib import Path

# Rabbithole API configuration
LLM_API_URL = "https://api.rabbithole.cred.club/v1/chat/completions"
LLM_API_KEY = os.environ.get("RABBITHOLE_API_KEY", "sk-E26rh594RiFea2Wz2dORaQ")

def read_function_doc(file_path):
    """Read the function documentation from a file."""
    with open(file_path, 'r') as f:
        return f.read()

def read_vector_results(file_path):
    """Read the vector database query results."""
    with open(file_path, 'r') as f:
        return f.read()

def generate_time_estimate(function_doc, vector_results, model="rabbit-7b"):
    """Generate time estimate using an LLM."""
    
    # Create the prompt for the LLM
    prompt = f"""
You are an expert software developer tasked with estimating implementation time.

# Function Documentation:
{function_doc}

# Relevant Code and Context from the Repository:
{vector_results}

Based on the function documentation and the repository context, please provide:

1. An estimated time range to implement this functionality (in hours or days)
2. A complexity assessment (Low, Medium, High)
3. Key factors that influence the time estimate
4. Potential challenges or risks
5. Breakdown of implementation tasks with sub-estimates

Format your response as JSON with the following structure:
{{
  "time_estimate": {{
    "min_hours": number,
    "max_hours": number,
    "confidence": "Low|Medium|High"
  }},
  "complexity": "Low|Medium|High",
  "key_factors": [
    "factor 1",
    "factor 2",
    ...
  ],
  "potential_challenges": [
    "challenge 1",
    "challenge 2",
    ...
  ],
  "tasks": [
    {{
      "name": "task 1",
      "hours": number,
      "description": "description"
    }},
    ...
  ]
}}
"""

    # Call the Rabbithole API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert software developer specializing in time estimation."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,  # Lower temperature for more consistent results
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling LLM API: {str(e)}")
        return None

def format_output(estimate_json):
    """Format the JSON output into a readable report."""
    try:
        data = json.loads(estimate_json)
        
        # Format the report
        report = []
        report.append("# Implementation Time Estimate Report\n")
        
        # Time estimate
        time_est = data["time_estimate"]
        report.append("## Time Estimate")
        report.append(f"- **Range**: {time_est['min_hours']} to {time_est['max_hours']} hours")
        report.append(f"- **Confidence**: {time_est['confidence']}")
        report.append("")
        
        # Complexity
        report.append("## Complexity")
        report.append(f"- **Assessment**: {data['complexity']}")
        report.append("")
        
        # Key factors
        report.append("## Key Factors")
        for factor in data["key_factors"]:
            report.append(f"- {factor}")
        report.append("")
        
        # Potential challenges
        report.append("## Potential Challenges")
        for challenge in data["potential_challenges"]:
            report.append(f"- {challenge}")
        report.append("")
        
        # Tasks breakdown
        report.append("## Implementation Tasks")
        total_hours = 0
        for task in data["tasks"]:
            report.append(f"### {task['name']} ({task['hours']} hours)")
            report.append(f"{task['description']}")
            report.append("")
            total_hours += task["hours"]
        
        report.append(f"**Total Hours (Sum of Tasks)**: {total_hours}")
        
        return "\n".join(report)
    except json.JSONDecodeError:
        return "Error: Could not parse LLM response as JSON.\n\nRaw response:\n" + estimate_json
    except KeyError as e:
        return f"Error: Missing key in JSON response: {str(e)}.\n\nRaw response:\n" + estimate_json

def main():
    parser = argparse.ArgumentParser(description="Generate implementation time estimates")
    parser.add_argument("--function-doc", required=True, help="Path to function documentation file")
    parser.add_argument("--vector-results", required=True, help="Path to vector database results file")
    parser.add_argument("--output", help="Path to output file (default: stdout)")
    parser.add_argument("--api-key", help="Rabbithole API key (default: RABBITHOLE_API_KEY environment variable or built-in key)")
    parser.add_argument("--model", default="rabbit-7b", help="LLM model to use (default: rabbit-7b)")
    
    args = parser.parse_args()
    
    # Set API key
    global LLM_API_KEY
    if args.api_key:
        LLM_API_KEY = args.api_key
    
    if not LLM_API_KEY:
        print("Error: No API key provided. Set RABBITHOLE_API_KEY environment variable or use --api-key")
        return 1
    
    # Read inputs
    function_doc = read_function_doc(args.function_doc)
    vector_results = read_vector_results(args.vector_results)
    
    # Generate estimate
    estimate_json = generate_time_estimate(function_doc, vector_results, args.model)
    if not estimate_json:
        print("Failed to generate time estimate")
        return 1
    
    # Format output
    report = format_output(estimate_json)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
