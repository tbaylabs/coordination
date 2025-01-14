import itertools
from pathlib import Path
from .create_pipeline_for_one import create_pipeline_for_one_condition

def create_many_pipelines(prompt_conditions, model_parameters, base_folder=None, test_mode=False):
    """
    Creates multiple pipelines by iterating over lists of prompt conditions
    and model parameters, calling `create_pipeline_for_one_condition` for
    each permutation, but does *not* run them.
    
    Returns:
        list[str]: A list of data_collection paths (one per permutation).
    """
    # Unpack lists from prompt_conditions
    task_keys = prompt_conditions["task_instruction_component_keys"]
    options_keys = prompt_conditions["options_lists_keys"]
    reasoning_keys = prompt_conditions["reasoning_instruction_component_keys"]
    
    # Unpack lists from model_parameters
    model_names = model_parameters["model_names"]
    temperatures = model_parameters["temperatures"]
    xml_prompts = model_parameters["xml_prompts"]
    
    # Build all permutations
    all_permutations = list(itertools.product(
        task_keys,
        options_keys,
        reasoning_keys,
        model_names,
        temperatures,
        xml_prompts
    ))
    
    print(f"[create_many_pipelines] Generating {len(all_permutations)} pipeline(s)...")
    data_collection_paths = []
    
    for (task_key, options_key, reasoning_key, model_name, temperature, xml_prompt) in all_permutations:
        single_prompt_conditions = {
            "task_instruction_component_key": task_key,
            "options_lists_key": options_key,
            "reasoning_instruction_component_key": reasoning_key
        }
        single_model_parameters = {
            "model_name": model_name,
            "temperature": temperature,
            "xml_prompt": xml_prompt
        }
        
        # Create the pipeline (which writes 3 JSON files to disk)
        data_collection_path = create_pipeline_for_one_condition(
            prompt_conditions=single_prompt_conditions,
            model_parameters=single_model_parameters,
            base_folder=base_folder,
            test_mode=test_mode
        )
        # Convert to absolute path
        data_collection_paths.append(str(Path(data_collection_path).resolve()))
    
    return data_collection_paths