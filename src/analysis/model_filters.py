def get_llama_family_filter():
    """
    Returns dictionaries of model family names mapped to their respective model identifiers.
    """
    MODEL_FAMILIES = {
        'llama_31': [
            "llama-31-8b",
            "llama-31-70b",
            "llama-31-405b"
        ],
        'llama_32': [
            "llama-32-1b",
            "llama-32-3b"
        ]
    }
    return MODEL_FAMILIES

def filter_by_model_family(df, family_name):
    """
    Filter DataFrame to include only models from a specific family and sort them
    according to the order defined in MODEL_FAMILIES.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'model_name' column
        family_name (str): Name of model family ('llama_31' or 'llama_32')
        
    Returns:
        pd.DataFrame: Filtered and sorted DataFrame containing only specified model family
    """
    MODEL_FAMILIES = get_llama_family_filter()
    
    if family_name not in MODEL_FAMILIES:
        raise ValueError(f"Family name must be one of {list(MODEL_FAMILIES.keys())}")
    
    # Create a dictionary mapping model names to their position in the original list
    order_dict = {model: idx for idx, model in enumerate(MODEL_FAMILIES[family_name])}
    
    # Filter the DataFrame and create a new column for sorting
    filtered_df = df[df['model_name'].isin(MODEL_FAMILIES[family_name])].copy()
    filtered_df['sort_order'] = filtered_df['model_name'].map(order_dict)
    
    # Sort by the order column and drop it
    result = filtered_df.sort_values('sort_order').drop('sort_order', axis=1)
    
    return result

def detect_model_family(models):
    """
    Detect which model family a set of models belongs to.
    
    Args:
        models (list-like): List of model names
    
    Returns:
        str or None: Name of the model family if detected, None otherwise
    """
    MODEL_FAMILIES = get_llama_family_filter()
    models_set = set(models)
    
    for family_name, family_models in MODEL_FAMILIES.items():
        if models_set.issubset(set(family_models)):
            return family_name
    return None

def get_model_order(models):
    """
    Get the correct order for a set of models, respecting family order if applicable.
    
    Args:
        models (list-like): List of model names
    
    Returns:
        list: Ordered list of model names
    """
    family = detect_model_family(models)
    if family:
        MODEL_FAMILIES = get_llama_family_filter()
        # Return only the models that are present in the input
        return [m for m in MODEL_FAMILIES[family] if m in models]
    else:
        # If no family detected, return sorted list
        return sorted(models)
