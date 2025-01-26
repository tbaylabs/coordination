from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import time
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

        # Rate limiting for Gemini models
        gemini_rate_limit = 5  # Max requests per minute
        gemini_last_call_time = None
        gemini_call_count = 0

        for result_number in missing_result_numbers:
            # Handle rate limiting for Gemini models
            if "gemini" in model_name.lower():
                current_time = datetime.now()
                if gemini_last_call_time and (current_time - gemini_last_call_time) < timedelta(minutes=1):
                    gemini_call_count += 1
                    if gemini_call_count >= gemini_rate_limit:
                        # Calculate sleep time until next minute window
                        sleep_time = 60 - (current_time - gemini_last_call_time).seconds
                        print(f"Gemini rate limit reached. Sleeping for {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        gemini_call_count = 0
                        gemini_last_call_time = datetime.now()
                else:
                    gemini_last_call_time = current_time
                    gemini_call_count = 1

            # Initialize log_entry with basic info
            log_entry = {
                "call_number": call_number,
                "result_number": result_number,
                "prompt_as_sent": None,
                "content_received": None,
                "status": "error",  # Default to error, will update if successful
                "error_message": None,
                "call_id": f"{call_number}_{hash_key[:6]}_{file_name}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response": {}
            }
            
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

                # Extract response details based on model type
                response_details = {}
                if model_name in ["o1", "o1-mini", "o1-preview"]:
                    if hasattr(response, 'usage'):
                        # Safely extract token details for o1 family
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
                elif model_name == "deepseek-r1":
                    # Extract reasoning content for deepseek-r1
                    try:
                        message = response.choices[0].message
                        
                        # Try multiple ways to get reasoning content
                        reasoning_content = None
                        
                        # Try provider_specific_fields first (newer LiteLLM versions)
                        if hasattr(message, 'provider_specific_fields'):
                            reasoning_content = message.provider_specific_fields.get('reasoning_content')
                        
                        # Try direct access next (original DeepSeek format)
                        if reasoning_content is None and hasattr(message, 'reasoning_content'):
                            reasoning_content = message.reasoning_content
                            
                        # Try dictionary access (in case message is a dict)
                        if reasoning_content is None and isinstance(message, dict):
                            reasoning_content = message.get('reasoning_content')
                            
                        print(f"Debug - Reasoning tokens found: {len(str(reasoning_content)) if reasoning_content else 0}")
                            
                        response_details = {
                            "id": response.id if hasattr(response, 'id') else None,
                            "created": response.created if hasattr(response, 'created') else None,
                            "message": {
                                "role": message.get("role", "assistant") if isinstance(message, dict) else getattr(message, "role", "assistant"),
                                "content": message.get("content", "") if isinstance(message, dict) else getattr(message, "content", ""),
                                "reasoning_content": reasoning_content
                            }
                        }
                        
                        # Handle usage info safely
                        if hasattr(response, 'usage'):
                            usage_dict = {}
                            if hasattr(response.usage, 'prompt_tokens'):
                                usage_dict['prompt_tokens'] = response.usage.prompt_tokens
                            if hasattr(response.usage, 'completion_tokens'):
                                usage_dict['completion_tokens'] = response.usage.completion_tokens
                            if hasattr(response.usage, 'total_tokens'):
                                usage_dict['total_tokens'] = response.usage.total_tokens
                            response_details['usage'] = usage_dict
                            
                    except Exception as e:
                        print(f"Warning: Error processing DeepSeek response: {str(e)}")
                        response_details = {
                            "message": {
                                "role": "assistant",
                                "content": content_received
                            }
                        }

                # Update log entry for success
                log_entry.update({
                    "prompt_as_sent": prompt,
                    "content_received": content_received,
                    "status": "success",
                    "response": response_details
                })
                print(f"Successfully collected data for result_number {result_number}")

            except Exception as e:
                # Update log entry for error
                log_entry.update({
                    "error_message": str(e),
                    "response": {}
                })
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