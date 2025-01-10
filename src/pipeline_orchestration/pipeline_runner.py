"""
Functions for running the complete data collection pipeline.
"""

import json
from pathlib import Path

from .data_collector import collect_data
from .extract_answers import extract_answers
from .gather_results import gather_results

def run_pipeline(data_collection_path, n, model_mapping_file="model_mapping.json"):
    """
    Orchestrates the full pipeline: data_collection, answer_extraction, and gather_results.
    Proceeds to each stage only if the previous one completed successfully.

    Args:
        data_collection_path (str): Path to the data collection JSON file.
        n (int): Maximum result_number to process (1 to n, n ≤ 120).

    Returns:
        dict: A summary of the entire pipeline execution.
    """
    if not (1 <= n <= 120):
        raise ValueError("The value of n must be between 1 and 120.")

    # Load the pipeline paths from the data_collection file
    with open(data_collection_path, 'r') as file:
        data = json.load(file)

    pipeline_paths = data.get("pipeline-paths")
    if not pipeline_paths:
        raise ValueError("No pipeline-paths found in the data collection file.")

    data_collection_file = pipeline_paths["data_collection"]
    answer_extraction_file = pipeline_paths["answer_extraction"]
    results_file = pipeline_paths["results"]

    try:
        # Step 1: Data Collection
        print("Starting data collection...")
        data_collection_summary = collect_data(file_path=data_collection_file, n=n, model_mapping_file=model_mapping_file)
        print(f"Data collection completed: {data_collection_summary}")

        if data_collection_summary["error_logs"]:
            raise RuntimeError("Data collection encountered errors. Halting pipeline.")

        # Step 2: Answer Extraction
        print("Starting answer extraction...")
        answer_extraction_summary = extract_answers(answer_extraction_file_path=answer_extraction_file, n=n, model_mapping_file=model_mapping_file)
        print(f"Answer extraction completed: {answer_extraction_summary}")

        if answer_extraction_summary["error_logs"]:
            raise RuntimeError("Answer extraction encountered errors. Halting pipeline.")

        # Step 3: Gather Results
        print("Starting results gathering...")
        gather_results(
            results_file_path=results_file,
            answer_extraction_file_path=answer_extraction_file,
            n=n
        )
        print("Results gathering completed successfully.")

        # Pipeline Summary
        return {
            "data_collection": data_collection_summary,
            "answer_extraction": answer_extraction_summary,
            "results": "completed successfully"
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Pipeline failed: {error_msg}")
        
        # Create error summary for data collection if it failed
        if not 'data_collection_summary' in locals():
            data_collection_summary = {
                "success_count": 0,
                "error_logs": [{
                    "error_message": error_msg,
                    "status": "error"
                }]
            }
            
        return {
            "data_collection": data_collection_summary,
            "answer_extraction": answer_extraction_summary if 'answer_extraction_summary' in locals() else None,
            "results": "failed",
            "error": error_msg
        }

def create_and_run_pipeline(prompt_conditions, model_parameters, n, base_folder=None, test_mode=False, model_mapping_file="model_mapping.json"):
    """
    Creates the pipeline for a single condition and then runs the entire pipeline.

    Args:
        prompt_conditions (dict): The prompt conditions for the pipeline.
        model_parameters (dict): The model parameters for the pipeline.
        n (int): Maximum result_number to process (1 ≤ n ≤ 120).
        base_folder (str, optional): Base folder for pipeline files.
        test_mode (bool): If True, uses test-specific folders.

    Returns:
        dict: A summary of the pipeline execution.
    """
    if not (1 <= n <= 120):
        raise ValueError("The value of n must be between 1 and 120.")

    from .create_pipeline_for_one import create_pipeline_for_one_condition

    # Create the pipeline files and get the data_collection file path
    data_collection_path = create_pipeline_for_one_condition(
        prompt_conditions, 
        model_parameters,
        base_folder=base_folder,
        test_mode=test_mode
    )

    # Convert data_collection_path to absolute path before running pipeline
    data_collection_path = Path(data_collection_path).resolve()
    pipeline_summary = run_pipeline(data_collection_path=data_collection_path, n=n, model_mapping_file=model_mapping_file)

    return pipeline_summary
