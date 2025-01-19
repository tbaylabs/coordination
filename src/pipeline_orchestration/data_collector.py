"""
Functions for collecting data from LLMs and managing the collection process.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
from litellm import completion

from ..utils import get_model_mapping, generate_permutations, get_permutation_pattern_from_result_number
from ..utils.prompt_utils import build_prompt

def collect_data(file_path, n, model_mapping_file="model_mapping.json"):
    """
    Collects data from LLMs by making calls and storing responses.

    Args:
        file_path (str): Path to the data collection JSON file.
        n (int): Maximum result number to process (1 to n).
        model_mapping_file (str): Path to the model mapping file.

    Returns:
        dict: Summary of collection process.
    """
    if not (1 <= n <= 120):
        raise ValueError("The value of n must be between 1 and 120.")
    
    # Convert to absolute path relative to project root
    project_root = Path(__file__).parent.parent.parent
    file_path = (project_root / Path(file_path)).resolve()
    print(f"Attempting to load data collection file from: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        data = json.load(file)

    data_collection_log = data.setdefault("data_collection_log", [])
    if not isinstance(data_collection_log, list):
        raise ValueError("The 'data_collection_log' must be a list in the JSON file.")

    # Identify missing successes
    successful_results = {entry["result_number"] for entry in data_collection_log if entry["status"] == "success"}
    missing_result_numbers = [result_number for result_number in range(1, n + 1) 
                            if result_number not in successful_results]

    error_logs = []
    call_number = len(data_collection_log) + 1
    hash_key = data.get("pipeline-hash", "")
    file_name = file_path.stem

    if missing_result_numbers:
        print(f"Processing missing result numbers: {missing_result_numbers}")

        # Load overview details
        overview = data.get("overview", {})
        if not overview:
            raise ValueError("The 'overview' key is missing or empty in the JSON file.")

        # Convert model mapping path to absolute path relative to project root
        model_mapping_path = (project_root / model_mapping_file).resolve()
        model_mapping = get_model_mapping(model_mapping_path)
        model_name = overview.get("model_name", "fail")
        if model_name not in model_mapping:
            raise ValueError(f"Model name '{model_name}' not found in the model mapping file.")
        model_address = model_mapping[model_name]

        temperature = overview.get("temperature", "default")
        task_key = overview.get("task_instruction_component_key", "")
        options_key = overview.get("options_lists_key", "")
        reasoning_key = overview.get("reasoning_instruction_component_key", "")
        return_xml = overview.get("return_xml", False)

        permutations_list = generate_permutations()

        for result_number in missing_result_numbers:
            try:
                # Generate permutation pattern and prompt
                permutation_pattern = get_permutation_pattern_from_result_number(
                    permutations_list, 
                    result_number
                )
                prompt = build_prompt(
                    task_key,
                    options_key,
                    reasoning_key,
                    permutation_pattern,
                    return_xml
                )

                # Call LLM
                completion_params = {
                    "model": model_address,
                    "messages": [{"content": prompt, "role": "user"}],
                    "stream": False,
                }
                if temperature != "default":
                    completion_params["temperature"] = float(temperature)

                response = completion(**completion_params)
                
                # Check response structure
                if "choices" not in response or not response["choices"]:
                    raise ValueError("The LLM response is missing 'choices' or is empty.")
                if "message" not in response["choices"][0]:
                    raise ValueError("The LLM response is missing 'message' in the first choice.")

                content_received = response["choices"][0]["message"]["content"]

                # Extract only needed response details for o1/o1-mini models
                response_details = {}
                if model_name in ["o1", "o1-mini"]:
                    if hasattr(response, 'usage'):
                        # Safely extract token details
                        token_details = {}
                        if hasattr(response.usage, 'completion_tokens_details'):
                            try:
                                token_details = {
                                    "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens,
                                    "accepted_prediction_tokens": response.usage.completion_tokens_details.accepted_prediction_tokens,
                                    "rejected_prediction_tokens": response.usage.completion_tokens_details.rejected_prediction_tokens
                                }
                            except AttributeError:
                                token_details = {
                                    "reasoning_tokens": 0,
                                    "accepted_prediction_tokens": 0,
                                    "rejected_prediction_tokens": 0
                                }
                        
                        response_details = {
                            "id": response.id,
                            "created": response.created,
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "completion_tokens_details": token_details
                            }
                        }

                # Log successful response
                log_entry = {
                    "call_number": call_number,
                    "result_number": result_number,
                    "prompt_as_sent": prompt,
                    "content_received": content_received,
                    "status": "success",
                    "error_message": None,
                    "call_id": f"{call_number}_{hash_key[:6]}_{file_name}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "response": response_details
                }
                print(f"Successfully collected data for result_number {result_number}")

            except Exception as e:
                # Log error
                log_entry = {
                    "call_number": call_number,
                    "result_number": result_number,
                    "prompt_as_sent": None,
                    "content_received": None,
                    "status": "error",
                    "error_message": str(e),
                    "call_id": f"{call_number}_{hash_key[:6]}_{file_name}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "response": response

                }
                error_logs.append(log_entry)
                print(f"Error encountered for result_number {result_number}: {e}")
            finally:
                data_collection_log.append(log_entry)
                call_number += 1

            # Save after each collection to preserve progress
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

    # Return the summary
    success_count = len([entry for entry in data_collection_log if entry["status"] == "success"])
    return {
        "success_count": success_count,
        "error_logs": error_logs
    }
