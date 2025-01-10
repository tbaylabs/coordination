"""
Module for processing answer extraction files and managing the extraction pipeline.
"""

from datetime import datetime, timezone
from pathlib import Path
import json

from ..utils import generate_permutations, get_permutation_pattern_from_result_number
from ..utils.answer_extraction import extract_answer_by_rule, extract_answer_by_llm

def extract_answers(answer_extraction_file_path, n, model_mapping_file="model_mapping.json"):
    """
    Process an answer extraction file and validate/update entries.
    
    Args:
        answer_extraction_file_path (str): Path to answer extraction file
        n (int): Maximum result number to process
        model_mapping_file (str): Path to model mapping file
        
    Returns:
        dict: Summary of extraction process
    """
    if not (1 <= n <= 120):
        raise ValueError("The value of n must be between 1 and 120.")

    # Convert paths to absolute paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    answer_extraction_file_path = (project_root / Path(answer_extraction_file_path)).resolve()
    model_mapping_path = (project_root / Path(model_mapping_file)).resolve()
    
    if not answer_extraction_file_path.exists():
        raise FileNotFoundError(f"The file {answer_extraction_file_path} does not exist.")

    with open(answer_extraction_file_path, 'r') as file:
        answer_extraction_data = json.load(file)
    print(f"Loaded answer extraction file: {answer_extraction_file_path}")

    pipeline_paths = answer_extraction_data.get("pipeline-paths", {})
    if "data_collection" not in pipeline_paths:
        raise ValueError("The 'data_collection' path is missing in the 'pipeline-paths' key.")
    print(f"Pipeline paths loaded: {pipeline_paths}")

    # Load the corresponding data collection file
    data_collection_file_path = (project_root / Path(pipeline_paths["data_collection"])).resolve()
    if not data_collection_file_path.exists():
        raise FileNotFoundError(f"The corresponding data collection file {data_collection_file_path} does not exist.")

    with open(data_collection_file_path, 'r') as file:
        data_collection_data = json.load(file)
    print(f"Loaded data collection file: {data_collection_file_path}")

    data_collection_log = data_collection_data.get("data_collection_log", [])
    if not isinstance(data_collection_log, list):
        raise ValueError("The 'data_collection_log' must be a list in the data collection JSON file.")
    print(f"Loaded data collection log with {len(data_collection_log)} entries")

    # Check data collection log for successful entries
    successful_data_collection_nums = {entry["result_number"] for entry in data_collection_log if entry["status"] == "success"}
    missing_data_collection = [num for num in range(1, n + 1) if num not in successful_data_collection_nums]

    if missing_data_collection:
        raise ValueError(f"Missing successful data collection entries for result_numbers: {missing_data_collection}")
    else:
        print(f"Success entries found for all result_numbers 1 to {n} in the data collection log.")

    # Process answer extraction log
    answer_extraction_log = answer_extraction_data.setdefault("answer_extraction_log", [])
    print(f"Loaded answer extraction log with {len(answer_extraction_log)} existing entries.")

    existing_successes = {int(item["result_number"]) for item in answer_extraction_log if item.get("status") == "success"}
    print(f"Existing successful result_numbers in answer_extraction_log: {existing_successes}")
    
    to_process = [num for num in range(1, n + 1) if num not in existing_successes]
    print(f"Result numbers to process: {to_process}")

    # Prepare extraction parameters
    pipeline_hash = answer_extraction_data.get("pipeline-hash", "nohash")
    overview = answer_extraction_data.get("overview", {})
    options_list = overview.get("options_list", [])
    filename_base = answer_extraction_file_path.stem

    extraction_attempt_number = max(
        (item.get("extraction_attempt_number", 0) for item in answer_extraction_log if isinstance(item.get("extraction_attempt_number"), int)),
        default=0
    ) + 1

    error_logs = []
    new_success_count = 0

    # Create lookup for successful data collection entries
    data_collection_success_map = {
        entry["result_number"]: entry 
        for entry in data_collection_log 
        if entry["status"] == "success"
    }

    # Process each missing result number
    for result_number in to_process:
        try:
            # Get corresponding data collection entry
            data_entry = data_collection_success_map[result_number]
            
            # Setup extraction attempt metadata
            extraction_attempt_timestamp = datetime.now(timezone.utc).isoformat()
            extraction_attempt_id = f"{extraction_attempt_number}_{pipeline_hash[:6]}_{filename_base}"

            # Initialize result dictionary
            result_dict = {
                "extraction_attempt_number": extraction_attempt_number,
                "result_number": str(result_number),
                "options_list": options_list,
                "content_received": data_entry.get("content_received", ""),
                "rule_extract": None,
                "llm_extract": None,
                "llm_extract_chat_history": None,
                "llm_extract_model": None,
                "llm_extract_error": None,
                "human_extract": "not checked by a human",
                "extraction_attempt_id": extraction_attempt_id,
                "extraction_attempt_timestamp": extraction_attempt_timestamp,
                "call_id": data_entry.get("call_id", None),
                "status": None,
                "call_number": data_entry.get("call_number")
            }

            # Attempt rule-based extraction
            print(f"Attempting rule-based extraction for result_number {result_number}.")
            rule_extract = extract_answer_by_rule(result_dict["content_received"], result_dict["options_list"])
            result_dict["rule_extract"] = rule_extract

            if isinstance(rule_extract, str):
                print(f"Rule-based extraction succeeded for result_number {result_number}.")
                result_dict["status"] = "success"
            else:
                # Attempt LLM-based extraction
                print(f"Rule-based extraction failed for result_number {result_number}. Attempting LLM-based extraction.")
                permutation_pattern = get_permutation_pattern_from_result_number(
                    generate_permutations(), 
                    result_number
                )
                llm_result = extract_answer_by_llm(
                    content_received=result_dict["content_received"],
                    options_list=result_dict["options_list"],
                    permutation_pattern=permutation_pattern,
                    model_mapping_file=str(model_mapping_path)
                )

                result_dict["llm_extract"] = llm_result.get("llm_extract")
                result_dict["llm_extract_chat_history"] = llm_result.get("llm_extract_chat_history")
                result_dict["llm_extract_error"] = llm_result.get("llm_extract_error")
                result_dict["llm_extract_model"] = llm_result.get("llm_extract_model")

                if result_dict["llm_extract_error"]:
                    result_dict["status"] = "fail"
                    print(f"LLM-based extraction failed for result_number {result_number}: {result_dict['llm_extract_error']}")
                elif result_dict["llm_extract"] is None:
                    result_dict["status"] = "human check needed"
                    print(f"LLM-based extraction yielded no result for result_number {result_number}.")
                else:
                    result_dict["status"] = "success"
                    print(f"LLM-based extraction succeeded for result_number {result_number}.")

            answer_extraction_log.append(result_dict)
            if result_dict["status"] == "success":
                new_success_count += 1

        except Exception as e:
            print(f"Error during extraction for result_number {result_number}: {str(e)}")
            error_logs.append({
                "extraction_attempt_number": extraction_attempt_number,
                "result_number": result_number,
                "status": "error",
                "error_message": str(e),
            })
        finally:
            extraction_attempt_number += 1

    print(f"Saving updated answer_extraction_log with {len(answer_extraction_log)} total entries.")
    with open(answer_extraction_file_path, 'w') as file:
        json.dump(answer_extraction_data, file, indent=4)

    print(f"Extraction complete. New successes: {new_success_count}, Errors: {len(error_logs)}")
    return {
        "success_count": new_success_count,
        "error_logs": error_logs
    }
