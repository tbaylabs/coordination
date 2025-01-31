import sys
import pandas as pd
from pathlib import Path

def build_benchmark_data(df, model_name):
    """
    Builds a benchmark dataset for a specific model by transforming task options
    and filtering results. Saves results to a CSV file.
    
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
    
    # Create benchmark_results directory if it doesn't exist
    output_dir = Path(__file__).parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    
    # Create output file path
    output_file = output_dir / f"{model_name}.csv"
    
    # Save to CSV, overwriting if it exists
    model_df.to_csv(output_file, index=False)
    print(f"\nBenchmark data for model '{model_name}' saved to: {output_file}")
    
    # Print preview of the results
    print("\nPreview of saved data:")
    print_nice_dataframe(model_df.head(5))
    
    return model_df

def print_nice_dataframe(df, max_rows=120, show_index=False):
    """Generic function for nicely printing any DataFrame in a terminal-friendly format.
    
    Args:
        df (pd.DataFrame): DataFrame to display
        max_rows (int): Maximum number of rows to display
        show_index (bool): Whether to show the index in the output
    """
    # Set display options for better terminal output
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    if len(df) > max_rows:
        print(f"\nDisplaying first {max_rows} rows (total: {len(df)}):\n")
        display_df = df.head(max_rows)
    else:
        display_df = df
    
    if not show_index:
        print(display_df.to_string(index=False))
    else:
        print(display_df.to_string())
    
    # Reset display options to defaults
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

def main():
    """
    Command line interface for building benchmark data.
    Usage: python -m benchmark_builder.builder <model_name>
    """
    if len(sys.argv) != 2:
        print("Usage: python -m benchmark_builder.builder <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent
    
    # Load the aggregated results
    results_file = project_root / "pipeline" / "4_analysis" / "trial_results_aggregated.csv"
    
    if not results_file.exists():
        print(f"Error: Could not find results file at {results_file}")
        print("Please run aggregate_trial_results.py first")
        sys.exit(1)
        
    try:
        df = pd.read_csv(results_file)
        if model_name not in df['model_name'].unique():
            print(f"Error: Model '{model_name}' not found in results")
            print("Available models:", ", ".join(df['model_name'].unique()))
            sys.exit(1)
            
        build_benchmark_data(df, model_name)
        
    except Exception as e:
        print(f"Error processing results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
