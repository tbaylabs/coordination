"""
Functions for setting up and managing the data collection pipeline structure.
"""

import os
import json
from pathlib import Path

from ..utils.general import generate_hash
from ..utils.data_loading import load_prompt_components
from ..utils.prompt_utils import build_prompt

def create_pipeline_for_one_condition(prompt_conditions, model_parameters, base_folder=None, test_mode=False):
    """
    Creates a pipeline for one condition and generates three JSON files in 1_data_collection,
    2_answer_extraction, and 3_results folders under 'pipeline' if they do not already exist.

    Args:
        prompt_conditions (dict): Dictionary of prompt conditions.
        model_parameters (dict): Dictionary of model parameters.
        base_folder (str, optional): Base folder where the pipeline folders will be created.
                                   If None, uses 'pipeline' in project root.
        test_mode (bool): If True, uses test-specific folders to avoid interfering with real data.

    Returns:
        str: The path to the data_collection file that was created or confirmed.
    """
    # Get project root directory (where the src folder is)
    project_root = Path(__file__).parent.parent.parent
    
    # If base_folder not provided, use pipeline in root
    if base_folder is None:
        base_folder = project_root / "pipeline"
    else:
        # Convert relative path to absolute using project root
        base_folder = project_root / Path(base_folder)

    # If in test mode, modify folder names
    if test_mode:
        data_collection_folder = base_folder / "test_1_data_collection"
        answer_extraction_folder = base_folder / "test_2_answer_extraction"
        results_folder = base_folder / "test_3_results"
    else:
        data_collection_folder = base_folder / "1_data_collection"
        answer_extraction_folder = base_folder / "2_answer_extraction"
        results_folder = base_folder / "3_results"
    
    # Load components
    _, options_lists, _ = load_prompt_components()

    # Fetch the actual options_list
    options_list = options_lists.get(prompt_conditions["options_lists_key"])
    if options_list is None:
        raise ValueError(f"Options list not found for key: {prompt_conditions['options_lists_key']}")

    # Build example prompt
    example_prompt = build_prompt(
        task_instruction_component_key=prompt_conditions["task_instruction_component_key"],
        options_lists_key=prompt_conditions["options_lists_key"],
        reasoning_instruction_component_key=prompt_conditions["reasoning_instruction_component_key"],
        permutation_pattern="0123"
    )

    # Generate the filename
    temperature_suffix = ""
    if model_parameters["temperature"] != "default":
        temperature_str = str(model_parameters["temperature"]).replace(".", "")
        temperature_suffix = f"_temp{temperature_str}"
    
    xml_suffix = "_xml_prompt" if model_parameters.get("xml_prompt", False) else ""
    
    base_filename = (
        f"{prompt_conditions['task_instruction_component_key']}_"
        f"{prompt_conditions['options_lists_key']}_"
        f"{prompt_conditions['reasoning_instruction_component_key']}_"
        f"{model_parameters['model_name']}{temperature_suffix}{xml_suffix}.json"
    )
    
    # Define full paths for files
    data_collection_path = os.path.join(data_collection_folder, f"dc_{base_filename}")
    answer_extraction_path = os.path.join(answer_extraction_folder, f"ae_{base_filename}")
    results_path = os.path.join(results_folder, f"res_{base_filename}")

    # Create the overview object
    overview = {
        **model_parameters,
        **prompt_conditions,
        "options_list": options_list,
        "example_prompt": example_prompt
    }

    # Generate a deterministic hash
    overview_hash = generate_hash(overview)

    # Ensure the folders exist
    os.makedirs(data_collection_folder, exist_ok=True)
    os.makedirs(answer_extraction_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # Create the pipeline paths object
    pipeline_paths = {
        "data_collection": data_collection_path,
        "answer_extraction": answer_extraction_path,
        "results": results_path
    }

    base_file_content = {
        "pipeline-hash": overview_hash,
        "overview": overview,
        "pipeline-paths": pipeline_paths
    }

    def check_or_create_file(file_path, additional_content_key, additional_content_value):
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                existing_data = json.load(f)
            existing_hash = existing_data.get("pipeline-hash")
            if existing_hash != overview_hash:
                raise ValueError(
                    f"Hash mismatch for existing file: {file_path}\n"
                    f"Expected hash: {overview_hash}, Found: {existing_hash}"
                )
            else:
                print(f"File already exists and hash matches: {file_path}")
        else:
            new_file_content = {
                **base_file_content,
                additional_content_key: additional_content_value
            }
            with open(file_path, "w") as file:
                json.dump(new_file_content, file, indent=4)
            print(f"File created: {file_path}")

    # Check or create all pipeline files
    check_or_create_file(data_collection_path, "data_collection_log", [])
    check_or_create_file(answer_extraction_path, "answer_extraction_log", [])
    check_or_create_file(results_path, "results", [])

    print("Pipeline setup complete.")
    return data_collection_path
