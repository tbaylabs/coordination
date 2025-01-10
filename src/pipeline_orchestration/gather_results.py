"""
Functions for gathering and summarizing results from the data collection pipeline.
"""

import json
from pathlib import Path

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
        # Determine the extracted_answer
        if answer_extraction_entry.get("rule_extract") is not None:
            extracted_answer = answer_extraction_entry["rule_extract"]
        elif answer_extraction_entry.get("llm_extract") is not None:
            extracted_answer = answer_extraction_entry["llm_extract"]
        elif answer_extraction_entry.get("human_extract") == "not checked by a human":
            # Log the error and associated dictionary
            print(f"Error: Human extract not checked for result_number {answer_extraction_entry['result_number']}")
            print(f"Associated entry: {answer_extraction_entry}")
            return None
        elif answer_extraction_entry.get("human_extract") is not None:
            extracted_answer = answer_extraction_entry["human_extract"]
        else:
            raise ValueError(f"Invalid extraction state for result_number {answer_extraction_entry['result_number']}")

        # Return the results entry
        return {
            "results_number": answer_extraction_entry["result_number"],
            "extracted_answer": extracted_answer,
            "extraction_attempt_id": answer_extraction_entry["extraction_attempt_id"],
            "call_id": answer_extraction_entry["call_id"]
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

    for result in results_list:
        extracted_answer = result["extracted_answer"]
        if extracted_answer in results_summary:
            results_summary[extracted_answer] += 1
        else:
            raise ValueError(f"Unexpected extracted answer '{extracted_answer}' in results.")

    # Add results-summary to the JSON structure
    results_data["results-summary"] = results_summary

    # Save the updated results JSON file
    with open(results_file_path, 'w') as file:
        json.dump(results_data, file, indent=4)
