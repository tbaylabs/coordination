import tabulate
import pandas as pd
from prepare_graph_data import prepare_graph_data

def calculate_condition_differences():
    """Calculate and return differences between control and coordinate conditions"""
    data = prepare_graph_data()
    
    # We'll use top_prop_all as the main metric
    metric_data = data['top_prop_all']
    
    # Calculate differences
    differences = pd.DataFrame({
        'control_coordinate_diff': metric_data['coordinate'] - metric_data['control'],
        'control_cot_diff': metric_data['coordinate-COT'] - metric_data['control']
    })
    
    # Add model names as a column
    differences['model'] = differences.index
    
    # Sort by largest control_coordinate_diff first
    differences = differences.sort_values('control_coordinate_diff', ascending=False)
    
    return differences

def print_condition_differences():
    """Print a formatted table showing condition differences"""
    differences = calculate_condition_differences()
    
    # Format the differences as percentages
    formatted_diff = differences.copy()
    formatted_diff['control_coordinate_diff'] = formatted_diff['control_coordinate_diff'].apply(lambda x: f"{x:.1%}")
    formatted_diff['control_cot_diff'] = formatted_diff['control_cot_diff'].apply(lambda x: f"{x:.1%}")
    
    # Print with nice formatting
    print("\nDifferences between conditions (positive = coordinate better than control):")
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
    print_condition_differences()
