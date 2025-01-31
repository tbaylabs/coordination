import pandas as pd
from tabulate import tabulate

def build_benchmark_data(df, model_name):
    """
    Builds a benchmark dataset for a specific model by transforming task options
    and filtering results.
    
    Args:
        df (pd.DataFrame): DataFrame from aggregate_trial_results
        model_name (str): Name of the model to filter for
        
    Returns:
        pd.DataFrame: Transformed and filtered DataFrame
    """
    # Filter for specific model
    model_df = df[df['model_name'] == model_name].copy()
    
    # Remove letters task
    model_df = model_df[model_df['task_options'] != 'letters']
    
    def transform_task_options(task_option):
        """Transform a single task option into name and type."""
        if task_option.endswith('-text') or task_option.endswith('-english'):
            name = task_option.rsplit('-', 1)[0]
            type_val = 'text'
        elif task_option.endswith('-icon'):
            name = task_option.rsplit('-', 1)[0]
            type_val = 'symbol'
        else:
            name = task_option
            type_val = 'symbol'
        return pd.Series({'task_options_name': name, 'task_options_type': type_val})
    
    # Transform task options into name and type
    transformed = model_df['task_options'].apply(transform_task_options)
    model_df[['task_options_name', 'task_options_type']] = transformed
    
    # Remove original task_options column
    model_df = model_df.drop('task_options', axis=1)
    
    # Validate the transformation
    assert model_df['task_options_type'].isin(['symbol', 'text']).all(), \
        "Invalid task_options_type values found"
    
    assert not model_df['task_options_name'].isna().any() and \
           not model_df['task_options_type'].isna().any(), \
        "Missing values in transformed columns"
    
    # Check for pairs
    name_type_counts = model_df.groupby('task_options_name')['task_options_type'].nunique()
    assert (name_type_counts == 2).all(), \
        "Some task options don't have both symbol and text variants"
    
    # Print the results
    print(f"\nBenchmark data for model: {model_name}")
    print_nice_dataframe(model_df)
    
    return model_df

def print_nice_dataframe(df, max_rows=120, show_index=False):
    """Generic function for nicely printing any DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to display
        max_rows (int): Maximum number of rows to display
        show_index (bool): Whether to show the index in the output
    """
    if len(df) > max_rows:
        print(f"Displaying first {max_rows} rows (total: {len(df)}):\n")
        print(tabulate(df.head(max_rows), headers='keys', 
                     tablefmt='grid', showindex=show_index))
    else:
        print(tabulate(df, headers='keys', tablefmt='grid', 
                     showindex=show_index))
