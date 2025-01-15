from .create_many_pipelines import create_many_pipelines
from .run_many_pipelines import run_many_pipelines

def create_and_run_many_pipelines(
    prompt_conditions,
    model_parameters,
    n,
    base_folder=None,
    test_mode=False,
    max_concurrent=5,
    model_mapping_file="model_mapping.json"
):
    """
    Convenience function that first calls `create_many_pipelines`
    to create all pipeline JSON files, then calls `run_many_pipelines`
    to execute them in parallel.
    
    Args:
        prompt_conditions (dict): The prompt conditions for pipeline permutations.
        model_parameters (dict): The model parameters for pipeline permutations.
        n (int): The maximum result_number to process (1 ≤ n ≤ 120).
        base_folder (str, optional): Base folder for pipeline files.
        test_mode (bool): If True, uses test-specific folders.
        max_concurrent (int): Maximum number of pipelines to run concurrently.
        model_mapping_file (str): Path to the JSON file mapping model IDs.
    
    Returns:
        list[tuple(str, dict)]: A list of (data_collection_path, pipeline_summary) results.
    """
    print("[create_and_run_many_pipelines] Creating all pipeline permutations...")
    data_collection_paths = create_many_pipelines(
        prompt_conditions=prompt_conditions,
        model_parameters=model_parameters,
        base_folder=base_folder,
        test_mode=test_mode
    )
    
    print("[create_and_run_many_pipelines] Running all pipelines...")
    results = run_many_pipelines(
        data_collection_paths=data_collection_paths,
        n=n,
        max_concurrent=max_concurrent,
        model_mapping_file=model_mapping_file
    )
    
    return results