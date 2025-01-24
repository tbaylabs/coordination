"""
Functions for gathering and summarizing results from the data collection pipeline.
"""

import json
from pathlib import Path
from litellm import token_counter
import statistics

def gather_results(results_file_path, answer_extraction_file_path, n):
    """
    Processes an answer extraction file and generates a results JSON file.
    Transforms each valid entry from answer_extraction_log into a results entry.
    
    Args:
        results_file_path (str): Path to the results JSON file.
        answer_extraction_file_path (str): Path to the answer extraction JSON file.
        n (int): The maximum result_number to process (1 to n).
    """
    # Convert to absolute path relative to project root
    project_root = Path(__file__).parent.parent.parent
    answer_extraction_file_path = (project_root / Path(answer_extraction_file_path)).resolve()
    if not answer_extraction_file_path.exists():
        raise FileNotFoundError(f"The file {answer_extraction_file_path} does not exist.")

    with open(answer_extraction_file_path, 'r') as file:
        answer_extraction_data = json.load(file)

    answer_extraction_log = answer_extraction_data.get("answer_extraction_log", [])
    if not isinstance(answer_extraction_log, list):
        raise ValueError("The 'answer_extraction_log' must be a list in the answer extraction JSON file.")

    # Load or initialize the results file
    results_file_path = (project_root / Path(results_file_path)).resolve()
    if results_file_path.exists():
        with open(results_file_path, 'r') as file:
            results_data = json.load(file)
    else:
        results_data = {"results": []}

    results_list = results_data.setdefault("results", [])
    processed_results_numbers = {entry["results_number"] for entry in results_list}

    # Extract options_list from the overview
    overview = answer_extraction_data.get("overview", {})
    options_list = overview.get("options_list", [])
    if not isinstance(options_list, list):
        raise ValueError("The 'options_list' must be a list in the overview section of the JSON file.")

    def transform_to_results_entry(answer_extraction_entry):
        """
        Transforms an answer_extraction_log entry into a results dictionary.

        Args:
            answer_extraction_entry (dict): A dictionary entry from answer_extraction_log.

        Returns:
            dict: A dictionary for the results list.
        """
        extracted_answer = None
        extracted_by = None
        
        # Determine the extracted_answer and extracted_by
        if answer_extraction_entry.get("rule_extract") is not None:
            extracted_answer = answer_extraction_entry["rule_extract"]
            extracted_by = "rule"
        elif answer_extraction_entry.get("llm_extract") is not None:
            extracted_answer = answer_extraction_entry["llm_extract"]
            extracted_by = "llm"
        elif answer_extraction_entry.get("human_extract") == "not checked by a human":
            # Log the error and associated dictionary
            print(f"Error: Human extract not checked for result_number {answer_extraction_entry['result_number']}")
            print(f"Associated entry: {answer_extraction_entry}")
            return None
        elif answer_extraction_entry.get("human_extract") is not None:
            extracted_answer = answer_extraction_entry["human_extract"]
            extracted_by = "human"
        else:
            raise ValueError(f"Invalid extraction state for result_number {answer_extraction_entry['result_number']}")

        # Return the results entry
        # Get model name for token counting
        model_name = overview.get("model_name")
        if not model_name:
            raise ValueError("Model name not found in overview")
            
        # Get litellm model name from mapping
        with open(project_root / "model_mapping.json", "r") as f:
            model_mapping = json.load(f)
        litellm_model_name = model_mapping.get(model_name)
        if not litellm_model_name:
            raise ValueError(f"Model {model_name} not found in model_mapping.json")
            
        # Calculate token counts
        content = answer_extraction_entry.get("content_received", "")
        full_token_count = token_counter(model=litellm_model_name, text=content)
        
        # Calculate tokens before answer
        answer_pos = content.find(extracted_answer)
        content_before_answer = content[:answer_pos] if answer_pos != -1 else content
        before_answer_token_count = token_counter(model=litellm_model_name, text=content_before_answer)

        return {
            "results_number": answer_extraction_entry["result_number"],
            "extracted_answer": extracted_answer,
            "extracted_by": extracted_by,
            "extraction_attempt_id": answer_extraction_entry["extraction_attempt_id"],
            "call_id": answer_extraction_entry["call_id"],
            "content_received": content,
            "content_received_token_count": full_token_count,
            "content_before_answer_token_count": before_answer_token_count
        }

    # Process the answer_extraction_log
    for entry in answer_extraction_log:
        # Use the result_number as a string to match the format in processed_results_numbers
        result_number_str = entry["result_number"]

        # Skip if already processed or out of range
        if result_number_str in processed_results_numbers:
            continue
        if int(result_number_str) > n:
            continue

        # Transform the entry into a results entry
        result_entry = transform_to_results_entry(entry)
        if result_entry is None:
            # Skip processing if an error was logged
            continue

        results_list.append(result_entry)
        processed_results_numbers.add(result_number_str)

    # Generate results-summary
    options_list_with_unanswered = options_list + ["unanswered"]
    results_summary = {option: 0 for option in options_list_with_unanswered}
    
    # Initialize token count tracking
    token_counts = []
    before_answer_token_counts = []
    
    # Initialize extraction method counters
    extraction_counts = {
        "extracted_by_rule_count": 0,
        "extracted_by_llm_count": 0,
        "extracted_by_human_count": 0
    }

    for result in results_list:
        extracted_answer = result["extracted_answer"]
        if extracted_answer in results_summary:
            results_summary[extracted_answer] += 1
        else:
            raise ValueError(f"Unexpected extracted answer '{extracted_answer}' in results.")
        
        # Count extraction methods
        if result["extracted_by"] == "rule":
            extraction_counts["extracted_by_rule_count"] += 1
        elif result["extracted_by"] == "llm":
            extraction_counts["extracted_by_llm_count"] += 1
        elif result["extracted_by"] == "human":
            extraction_counts["extracted_by_human_count"] += 1
            
        # Track token counts
        if "content_received_token_count" in result:
            token_counts.append(result["content_received_token_count"])
        if "content_before_answer_token_count" in result:
            before_answer_token_counts.append(result["content_before_answer_token_count"])

    # Add token statistics to results-summary
    if token_counts:
        results_summary["token_statistics"] = {
            "average_token_count": statistics.mean(token_counts),
            "median_token_count": statistics.median(token_counts),
            "min_token_count": min(token_counts),
            "max_token_count": max(token_counts),
            "total_token_count": sum(token_counts)
        }
        
    if before_answer_token_counts:
        results_summary["token_statistics_before_answer"] = {
            "average_token_count": statistics.mean(before_answer_token_counts),
            "median_token_count": statistics.median(before_answer_token_counts),
            "min_token_count": min(before_answer_token_counts),
            "max_token_count": max(before_answer_token_counts),
            "total_token_count": sum(before_answer_token_counts)
        }

    # Create a new ordered dictionary with results-summary first
    ordered_data = {
        "pipeline-hash": results_data.get("pipeline-hash"),
        "overview": results_data.get("overview"),
        "pipeline-paths": results_data.get("pipeline-paths"),
        "results-summary": {**results_summary, **extraction_counts},
        "results": results_data.get("results", [])
    }

    # Save the updated results JSON file with ordered structure
    with open(results_file_path, 'w', encoding='utf-8') as file:
        json.dump(ordered_data, file, indent=4, ensure_ascii=False)

def refresh_results(results_file_path, n=120):
    """
    Refreshes the results by re-running gather_results using the associated answer extraction file.
    
    Args:
        results_file_path (str): Path to the results JSON file to refresh
        n (int): The maximum result_number to process (1 to n)
    """
    # Load the results file to get the answer extraction path
    project_root = Path(__file__).parent.parent.parent
    results_file_path = (project_root / Path(results_file_path)).resolve()
    
    if not results_file_path.exists():
        raise FileNotFoundError(f"The file {results_file_path} does not exist.")

    with open(results_file_path, 'r') as file:
        results_data = json.load(file)
    
    # Get the answer extraction path from pipeline-paths
    answer_extraction_path = results_data.get("pipeline-paths", {}).get("answer_extraction")
    if not answer_extraction_path:
        raise ValueError("Results file is missing required pipeline-paths.answer_extraction path")
    
    # Clear existing results and re-run gather_results
    results_data["results"] = []
    with open(results_file_path, 'w') as file:
        json.dump(results_data, file, indent=4)
    
    gather_results(results_file_path, answer_extraction_path, n)

def refresh_all_results(n=120):
    """
    Refreshes all results files in the 3_results folder that have exactly n results.
    
    Args:
        n (int): The expected number of results in each file (default 120)
    """
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "pipeline" / "3_results"
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    refreshed_count = 0
    skipped_count = 0
    
    for results_file in results_dir.glob("*.json"):
        try:
            with open(results_file, 'r') as file:
                results_data = json.load(file)
            
            current_results = len(results_data.get("results", []))
            
            if current_results != n:
                print(f"Warning: Skipping {results_file.name} - has {current_results} results (expected {n})")
                skipped_count += 1
                continue
                
            refresh_results(str(results_file), n)
            refreshed_count += 1
            print(f"Refreshed: {results_file.name}")
            
        except Exception as e:
            print(f"Error processing {results_file.name}: {str(e)}")
            skipped_count += 1
    
    print(f"\nRefresh complete:")
    print(f"  Files refreshed: {refreshed_count}")
    print(f"  Files skipped: {skipped_count}")
