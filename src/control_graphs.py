from tabulate import tabulate
import pandas as pd
from src.prepare_graph_data import prepare_graph_data

def calculate_reasoning_differences(models):
    """Calculate differences between control and coordinate for reasoning models"""
    data = prepare_graph_data()
    
    # Calculate for both metrics
    results = {}
    for metric in ['top_prop_all', 'top_prop_answered']:
        metric_data = data[metric].loc[models]
        
        # Calculate absolute and proportional differences
        differences = pd.DataFrame({
            'control_value': metric_data['control'],
            'coordinate_value': metric_data['coordinate'],
            'raw_change': metric_data['coordinate'] - metric_data['control'],
            'relative_change': (metric_data['coordinate'] - metric_data['control']) / metric_data['control']
        })
        
        # Add model names and sort by coordinate performance
        differences['model'] = differences.index
        differences = differences.sort_values('absolute_diff', ascending=False)
        
        results[metric] = differences
    
    return results

def calculate_standard_differences(models):
    """Calculate differences between conditions for standard models"""
    data = prepare_graph_data()
    
    # Calculate for both metrics
    results = {}
    for metric in ['top_prop_all', 'top_prop_answered']:
        metric_data = data[metric].loc[models]
        
        # Calculate absolute and proportional differences
        differences = pd.DataFrame({
            'control_value': metric_data['control'],
            'coordinate_value': metric_data['coordinate'],
            'coordinate_cot_value': metric_data['coordinate-COT'],
            'raw_change_coordinate': metric_data['coordinate'] - metric_data['control'],
            'raw_change_cot': metric_data['coordinate-COT'] - metric_data['control'],
            'relative_change_coordinate': (metric_data['coordinate'] - metric_data['control']) / metric_data['control'],
            'relative_change_cot': (metric_data['coordinate-COT'] - metric_data['control']) / metric_data['control']
        })
        
        # Add model names and sort by coordinate-COT performance
        differences['model'] = differences.index
        differences = differences.sort_values('absolute_diff_cot', ascending=False)
        
        results[metric] = differences
    
    return results

def print_reasoning_differences():
    """Print formatted table for reasoning models"""
    reasoning_models = ['o1-mini', 'deepseek-r1']
    differences = calculate_reasoning_differences(reasoning_models)
    
    # Print both metrics
    for metric_name, metric_data in differences.items():
        # Format values
        formatted_diff = metric_data.copy()
        formatted_diff['control_value'] = formatted_diff['control_value'].apply(lambda x: f"{x:.1%}")
        formatted_diff['coordinate_value'] = formatted_diff['coordinate_value'].apply(lambda x: f"{x:.1%}")
        formatted_diff['raw_change'] = formatted_diff['raw_change'].apply(lambda x: f"{x:.1%}")
        formatted_diff['relative_change'] = formatted_diff['relative_change'].apply(lambda x: f"{x:.1%}")
        
        print(f"\nReasoning Models - {metric_name.replace('_', ' ').title()} Differences:")
        print("(positive = coordinate better than control)")
        print_nice_dataframe(formatted_diff[['model', 'control_value', 'coordinate_value', 
                                           'raw_change', 'relative_change']])

def print_standard_differences():
    """Print formatted table for standard models"""
    standard_models = [
        'llama-31-405b', 'llama-31-70b', 'llama-31-8b',
        'gpt-4o', 'claude-35-sonnet'
    ]
    differences = calculate_standard_differences(standard_models)
    
    # Print both metrics
    for metric_name, metric_data in differences.items():
        # Format values
        formatted_diff = metric_data.copy()
        formatted_diff['control_value'] = formatted_diff['control_value'].apply(lambda x: f"{x:.1%}")
        formatted_diff['coordinate_value'] = formatted_diff['coordinate_value'].apply(lambda x: f"{x:.1%}")
        formatted_diff['coordinate_cot_value'] = formatted_diff['coordinate_cot_value'].apply(lambda x: f"{x:.1%}")
        formatted_diff['raw_change_coordinate'] = formatted_diff['raw_change_coordinate'].apply(lambda x: f"{x:.1%}")
        formatted_diff['raw_change_cot'] = formatted_diff['raw_change_cot'].apply(lambda x: f"{x:.1%}")
        formatted_diff['relative_change_coordinate'] = formatted_diff['relative_change_coordinate'].apply(lambda x: f"{x:.1%}")
        formatted_diff['relative_change_cot'] = formatted_diff['relative_change_cot'].apply(lambda x: f"{x:.1%}")
        
        print(f"\nStandard Models - {metric_name.replace('_', ' ').title()} Differences:")
        print("(positive = coordinate better than control)")
        print_nice_dataframe(formatted_diff[['model', 'control_value', 'coordinate_value', 'coordinate_cot_value',
                                           'raw_change_coordinate', 'raw_change_cot',
                                           'relative_change_coordinate', 'relative_change_cot']])

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
