from tabulate import tabulate
import pandas as pd
from prepare_graph_data import prepare_graph_data

## There should be two types of condition graphs created by seperate functions.
## Each should be able to take a list of models and then filter only for those models.
## One should be for reasoning models. In this case we are only comparing the difference between control and coordinate (none), but these should be labelled as control and coordinate
## The other should be for standard models. In this case we use the differences we already have below.
## The order the models from top to bottom on their performance on coordinate(COT) in the case of the standard models and coordinate in the case of the reasoning models
## Then in execution, For the reasoning models function, call the function with r1 and o1-mini only. 
## For the standard one, lets call it with llama 31 models, gpt 4o, and sonnet 
## Call the reasoning one with o1-mini and r1

def calculate_reasoning_differences(models):
    """Calculate differences between control and coordinate for reasoning models"""
    data = prepare_graph_data()
    metric_data = data['top_prop_all']
    
    # Filter for reasoning models
    metric_data = metric_data.loc[models]
    
    # Calculate differences (only control vs coordinate)
    differences = pd.DataFrame({
        'control_coordinate_diff': metric_data['coordinate'] - metric_data['control']
    })
    
    # Add model names and sort by coordinate performance
    differences['model'] = differences.index
    differences = differences.sort_values('control_coordinate_diff', ascending=False)
    
    return differences

def calculate_standard_differences(models):
    """Calculate differences between conditions for standard models"""
    data = prepare_graph_data()
    metric_data = data['top_prop_all']
    
    # Filter for standard models
    metric_data = metric_data.loc[models]
    
    # Calculate differences
    differences = pd.DataFrame({
        'control_coordinate_diff': metric_data['coordinate'] - metric_data['control'],
        'control_cot_diff': metric_data['coordinate-COT'] - metric_data['control']
    })
    
    # Add model names and sort by coordinate-COT performance
    differences['model'] = differences.index
    differences = differences.sort_values('control_cot_diff', ascending=False)
    
    return differences

def print_reasoning_differences():
    """Print formatted table for reasoning models"""
    reasoning_models = ['o1-mini', 'deepseek-r1']
    differences = calculate_reasoning_differences(reasoning_models)
    
    # Format as percentages
    formatted_diff = differences.copy()
    formatted_diff['control_coordinate_diff'] = formatted_diff['control_coordinate_diff'].apply(lambda x: f"{x:.1%}")
    
    print("\nReasoning Models - Differences (positive = coordinate better than control):")
    print_nice_dataframe(formatted_diff[['model', 'control_coordinate_diff']])

def print_standard_differences():
    """Print formatted table for standard models"""
    standard_models = [
        'llama-31-405b', 'llama-31-70b', 'llama-31-8b',
        'gpt-4o', 'claude-35-sonnet'
    ]
    differences = calculate_standard_differences(standard_models)
    
    # Format as percentages
    formatted_diff = differences.copy()
    formatted_diff['control_coordinate_diff'] = formatted_diff['control_coordinate_diff'].apply(lambda x: f"{x:.1%}")
    formatted_diff['control_cot_diff'] = formatted_diff['control_cot_diff'].apply(lambda x: f"{x:.1%}")
    
    print("\nStandard Models - Differences (positive = coordinate better than control):")
    print_nice_dataframe(formatted_diff[['model', 'control_coordinate_diff', 'control_cot_diff']])

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

if __name__ == "__main__":
    print_reasoning_differences()
    print_standard_differences()
