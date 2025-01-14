from concurrent.futures import ThreadPoolExecutor, as_completed

def run_many_pipelines(data_collection_paths, n, max_concurrent=5, model_mapping_file="model_mapping.json"):
    """
    Runs the pipeline (data_collection, answer_extraction, results) for each path
    in `data_collection_paths`, with up to `max_concurrent` in parallel.
    
    Args:
        data_collection_paths (list[str]): Paths to pipeline data collection JSON files.
        n (int): The maximum result_number to process (1 ≤ n ≤ 120).
        max_concurrent (int): Maximum number of pipelines to run in parallel.
        model_mapping_file (str): Path to the JSON file mapping model IDs.
    
    Returns:
        list[tuple(str, dict)]: A list of tuples (data_collection_path, pipeline_summary).
                                `pipeline_summary` is the dict returned by `run_pipeline`.
    """
    from .pipeline_runner import run_pipeline  # Import your run_pipeline function
    
    # Validate n
    if not (1 <= n <= 120):
        raise ValueError("The value of n must be between 1 and 120.")
    
    results = []
    print(f"[run_many_pipelines] Starting execution of {len(data_collection_paths)} pipeline(s) with up to {max_concurrent} in parallel.")
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_path = {
            executor.submit(run_pipeline, path, n, model_mapping_file): path
            for path in data_collection_paths
        }
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                summary = future.result()  # run_pipeline returns a dict summary
                results.append((path, summary))
                print(f"[run_many_pipelines] Pipeline completed: {path}")
            except Exception as exc:
                print(f"[run_many_pipelines] ERROR running pipeline for {path}: {exc}")
                results.append((path, None))
    
    return results