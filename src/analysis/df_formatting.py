"""
Utility functions for managing DataFrame columns and formatting.
"""

import pandas as pd
from tabulate import tabulate
from src.analysis.visualization import simplify_column_names

def get_standard_column_exclusions(top_results_only=True):
    """Returns standard columns to exclude based on configuration."""
    default_exclude = [
        'file_name',
        'temperature',
        'xml_prompt',
        'answered_count',
        'unanswered_count',
        'total_count',
        'convergence_answered',
        'top_option_count',
        'second_option_count',
        'third_option_count',
        'fourth_option_count'
    ]
    
    if top_results_only:
        default_exclude.extend([
            'second_option_name', 'second_option_count',
            'third_option_name', 'third_option_count',
            'fourth_option_name', 'fourth_option_count',
            'second_prop_all', 'third_prop_all', 'fourth_prop_all',
            'second', 'third', 'fourth',
            'second_prop', 'third_prop', 'fourth_prop'
        ])
    
    return default_exclude

def get_standard_column_order(use_simple_names=True, top_results_only=True):
    """Returns standard column ordering based on configuration."""
    if use_simple_names:
        primary_cols = ['model', 'task_options', 'task_reasoning', 'task_instruction']
        metric_cols = ['convergence', 'top_prop', 'top']
    else:
        primary_cols = ['model_name', 'task_options', 'task_reasoning']
        metric_cols = ['convergence_all', 'first_prop_all', 'top_option_name']
    
    if not top_results_only:
        if use_simple_names:
            additional_pairs = [
                ('second_prop', 'second'),
                ('third_prop', 'third'),
                ('fourth_prop', 'fourth')
            ]
        else:
            additional_pairs = [
                ('second_prop_all', 'second_option_name'),
                ('third_prop_all', 'third_option_name'),
                ('fourth_prop_all', 'fourth_option_name')
            ]
        for prop, name in additional_pairs:
            metric_cols.extend([prop, name])
    
    final_cols = ['unanswered_prop']
    return primary_cols + metric_cols + final_cols

def prepare_df_for_display(df, use_simple_names=True, include_cols=None, 
                         top_results_only=True, sort=True):
    """
    Prepares DataFrame for display with standard formatting.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        use_simple_names (bool): Whether to use simplified column names
        include_cols (list): Columns to include that would normally be excluded
        top_results_only (bool): Whether to show only top results
        sort (bool): Whether to sort by standard columns
    
    Returns:
        pd.DataFrame: Prepared DataFrame
    """
    display_df = df.copy()
    
    if use_simple_names:
        display_df = simplify_column_names(display_df)
    
    # Handle exclusions
    exclude_cols = get_standard_column_exclusions(top_results_only)
    if include_cols:
        exclude_cols = [col for col in exclude_cols if col not in include_cols]
    
    # Get column order
    ordered_cols = get_standard_column_order(use_simple_names, top_results_only)
    display_cols = [col for col in ordered_cols if col in display_df.columns 
                   and col not in exclude_cols]
    
    # Sort if requested
    if sort:
        sort_cols = ['model_name' if not use_simple_names else 'model',
                    'task_options',
                    'task_reasoning']
        display_df = display_df.sort_values(by=sort_cols)
    
    return display_df[display_cols]

def print_nice_dataframe(df, max_rows=20, show_index=False):
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

def print_nice_aggregate_trial_results_table(df, max_rows=20):
    """Prints aggregate trial results with nice formatting and simplified column names.
    
    Args:
        df (pd.DataFrame): DataFrame containing trial results
        max_rows (int): Maximum number of rows to display
    """
    # Prepare DataFrame for display
    display_df = prepare_df_for_display(
        df, 
        use_simple_names=True,
        include_cols=None,  # You might want to specify default columns here
        top_results_only=True
    )
    
    print("Note: Using simplified column names for display\n")
    
    # Use the generic printing function
    print_nice_dataframe(display_df, max_rows=max_rows, show_index=False)

import pandas as pd

def make_long_df(df):
    """
    Prepares aggregate trial results for repeated measures analysis,
    including a 'control' column for both 'convergence' and 'top_prop'.
    
    Args:
        df (pd.DataFrame): Input DataFrame from aggregate_trial_results
        
    Returns:
        pd.DataFrame: Prepared DataFrame suitable for repeated measures analysis
    """
    # Get standard columns with minimal metrics
    prepared_df = prepare_df_for_display(
        df,
        use_simple_names=True,
        top_results_only=True
    )
    
    # Keep only convergence and top_prop metrics
    metric_cols = ['convergence', 'top_prop']
    keep_cols = ['model', 'task_options', 'task_reasoning', 'task_instruction'] + metric_cols
    prepared_df = prepared_df[keep_cols]
    
    # Remove duplicate rows with the same (model, task_options, task_reasoning, task_instruction)
    prepared_df = prepared_df.drop_duplicates(
        subset=['model', 'task_options', 'task_reasoning', 'task_instruction']
    )
    
    # We'll collect the pivoted DataFrames for each metric, then merge them at the end
    all_metrics_merged = None
    
    for metric in metric_cols:
        # --- 1. Subset for coordinate instructions: pivot on task_reasoning ---
        df_coordinate = prepared_df[prepared_df['task_instruction'] == 'coordinate'].copy()
        
        wide_coordinate = df_coordinate.pivot(
            index=['model', 'task_options'],
            columns='task_reasoning',
            values=metric
        ).reset_index()
        
        # rename columns from { 'none': '*_without_reasoning', 'step-by-step': '*_with_reasoning' }
        wide_coordinate = wide_coordinate.rename(
            columns={
                'none': f'{metric}_without_reasoning',
                'step-by-step': f'{metric}_with_reasoning'
            }
        )
        
        # Some pivot calls may not produce both columns if data is missing,
        # so fill them in if they don't exist
        for col_suffix in ['without_reasoning', 'with_reasoning']:
            col_name = f'{metric}_{col_suffix}'
            if col_name not in wide_coordinate.columns:
                wide_coordinate[col_name] = None
        
        # Keep just the columns we need
        keep_coordinate = [
            'model', 'task_options',
            f'{metric}_without_reasoning', f'{metric}_with_reasoning'
        ]
        wide_coordinate = wide_coordinate[keep_coordinate]
        
        # --- 2. Subset for control instructions (task_instruction='control', task_reasoning='none') ---
        df_control = prepared_df[
            (prepared_df['task_instruction'] == 'control') & 
            (prepared_df['task_reasoning'] == 'none')
        ].copy()
        
        wide_control = df_control.pivot(
            index=['model', 'task_options'],
            columns='task_reasoning',
            values=metric
        ).reset_index()
        
        # rename columns: 'none' → '*_control'
        wide_control = wide_control.rename(
            columns={
                'none': f'{metric}_control'
            }
        )
        
        # If pivot is empty or missing the column, ensure it exists
        if f'{metric}_control' not in wide_control.columns:
            wide_control[f'{metric}_control'] = None
        
        # Keep just the relevant columns
        keep_control = [
            'model', 'task_options',
            f'{metric}_control'
        ]
        wide_control = wide_control[keep_control]
        
        # --- 3. Merge the coordinate and control wide DataFrames for this metric ---
        wide_metric = pd.merge(
            wide_coordinate,
            wide_control,
            on=['model', 'task_options'],
            how='outer'  # or "inner", depending on your needs
        )
        
        # --- 4. Merge this metric-wide DataFrame with the (potentially) already-merged metrics ---
        if all_metrics_merged is None:
            # First metric – just use this as our starting point
            all_metrics_merged = wide_metric
        else:
            # Merge with existing data
            all_metrics_merged = pd.merge(
                all_metrics_merged,
                wide_metric,
                on=['model', 'task_options'],
                how='outer'
            )
    
    # all_metrics_merged now contains, for each metric, columns:
    #   * [metric]_without_reasoning
    #   * [metric]_with_reasoning
    #   * [metric]_control
    
    return all_metrics_merged

def print_repeated_measures_df(df, max_rows=20):
    """Specialized printing for repeated measures format."""
    try:
        if len(df) > max_rows:
            print(f"Displaying first {max_rows} rows (total: {len(df)}):\n")
            print(tabulate(df.head(max_rows), headers='keys', 
                         tablefmt='grid', showindex=False))
        else:
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    except ImportError:
        pd.set_option('display.max_rows', max_rows)
        print(df.to_string())
