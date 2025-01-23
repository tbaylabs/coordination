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

##todo: include in the tables generated absolute differences (where you just deduct one percentage from the other) and proportional changes

def calculate_reasoning_differences(models):
    """Calculate differences between control and coordinate for reasoning models"""
    data = prepare_graph_data()
    metric_data = data['top_prop_all']
    
    # Filter for reasoning models
    metric_data = metric_data.loc[models]
    
    # Calculate absolute and proportional differences
    differences = pd.DataFrame({
        'control_value': metric_data['control'],
        'coordinate_value': metric_data['coordinate'],
        'absolute_diff': metric_data['coordinate'] - metric_data['control'],
        'proportional_change': (metric_data['coordinate'] - metric_data['control']) / metric_data['control']
    })
    
    # Add model names and sort by coordinate performance
    differences['model'] = differences.index
    differences = differences.sort_values('absolute_diff', ascending=False)
    
    return differences

def calculate_standard_differences(models):
    """Calculate differences between conditions for standard models"""
    data = prepare_graph_data()
    metric_data = data['top_prop_all']
    
    # Filter for standard models
    metric_data = metric_data.loc[models]
    
    # Calculate absolute and proportional differences
    differences = pd.DataFrame({
        'control_value': metric_data['control'],
        'coordinate_value': metric_data['coordinate'],
        'coordinate_cot_value': metric_data['coordinate-COT'],
        'absolute_diff_coordinate': metric_data['coordinate'] - metric_data['control'],
        'absolute_diff_cot': metric_data['coordinate-COT'] - metric_data['control'],
        'proportional_change_coordinate': (metric_data['coordinate'] - metric_data['control']) / metric_data['control'],
        'proportional_change_cot': (metric_data['coordinate-COT'] - metric_data['control']) / metric_data['control']
    })
    
    # Add model names and sort by coordinate-COT performance
    differences['model'] = differences.index
    differences = differences.sort_values('absolute_diff_cot', ascending=False)
    
    return differences

def print_reasoning_differences():
    """Print formatted table for reasoning models"""
    reasoning_models = ['o1-mini', 'deepseek-r1']
    differences = calculate_reasoning_differences(reasoning_models)
    
    # Format values
    formatted_diff = differences.copy()
    formatted_diff['control_value'] = formatted_diff['control_value'].apply(lambda x: f"{x:.1%}")
    formatted_diff['coordinate_value'] = formatted_diff['coordinate_value'].apply(lambda x: f"{x:.1%}")
    formatted_diff['absolute_diff'] = formatted_diff['absolute_diff'].apply(lambda x: f"{x:.1%}")
    formatted_diff['proportional_change'] = formatted_diff['proportional_change'].apply(lambda x: f"{x:.1%}")
    
    print("\nReasoning Models - Differences (positive = coordinate better than control):")
    print_nice_dataframe(formatted_diff[['model', 'control_value', 'coordinate_value', 
                                       'absolute_diff', 'proportional_change']])

def print_standard_differences():
    """Print formatted table for standard models"""
    standard_models = [
        'llama-31-405b', 'llama-31-70b', 'llama-31-8b',
        'gpt-4o', 'claude-35-sonnet'
    ]
    differences = calculate_standard_differences(standard_models)
    
    # Format values
    formatted_diff = differences.copy()
    formatted_diff['control_value'] = formatted_diff['control_value'].apply(lambda x: f"{x:.1%}")
    formatted_diff['coordinate_value'] = formatted_diff['coordinate_value'].apply(lambda x: f"{x:.1%}")
    formatted_diff['coordinate_cot_value'] = formatted_diff['coordinate_cot_value'].apply(lambda x: f"{x:.1%}")
    formatted_diff['absolute_diff_coordinate'] = formatted_diff['absolute_diff_coordinate'].apply(lambda x: f"{x:.1%}")
    formatted_diff['absolute_diff_cot'] = formatted_diff['absolute_diff_cot'].apply(lambda x: f"{x:.1%}")
    formatted_diff['proportional_change_coordinate'] = formatted_diff['proportional_change_coordinate'].apply(lambda x: f"{x:.1%}")
    formatted_diff['proportional_change_cot'] = formatted_diff['proportional_change_cot'].apply(lambda x: f"{x:.1%}")
    
    print("\nStandard Models - Differences (positive = coordinate better than control):")
    print_nice_dataframe(formatted_diff[['model', 'control_value', 'coordinate_value', 'coordinate_cot_value',
                                       'absolute_diff_coordinate', 'absolute_diff_cot',
                                       'proportional_change_coordinate', 'proportional_change_cot']])

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
