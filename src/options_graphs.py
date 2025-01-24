import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_graph_data import add_experiment_conditions

def calculate_task_deltas(df, metric='top_prop_all'):
    """
    Calculate deltas between conditions for each task.
    
    Args:
        df (pd.DataFrame): Raw experiment data
        metric (str): Metric to calculate deltas on ('top_prop_all' or 'top_prop_answered')
        
    Returns:
        pd.DataFrame: Dataframe with deltas for each task
    """
    # Add experiment conditions and filter
    df = add_experiment_conditions(df)
    
    # Group by task and experiment, calculate mean metric
    task_data = df.groupby(['task_options', 'experiment'], observed=True)[metric].mean().unstack()
    
    # Calculate deltas
    task_data['delta1'] = task_data['coordinate'] - task_data['control']
    task_data['delta2'] = task_data['coordinate-COT'] - task_data['coordinate']
    
    return task_data

def plot_delta_scatter(df, metric='top_prop_all'):
    """
    Create scatter plot of deltas between conditions.
    
    Args:
        df (pd.DataFrame): Raw experiment data
        metric (str): Metric to plot ('top_prop_all' or 'top_prop_answered')
    """
    # Calculate deltas
    deltas = calculate_task_deltas(df, metric)
    
    # Set up plot
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Create scatter plot
    ax = sns.scatterplot(
        x='delta1', 
        y='delta2', 
        data=deltas,
        s=100,
        alpha=0.8
    )
    
    # Add labels and title
    ax.set(
        xlabel='Δ1: Coordinate - Control',
        ylabel='Δ2: Coordinate-COT - Coordinate',
        title=f'Task Performance Deltas ({metric.replace("_", " ").title()})'
    )
    
    # Add quadrant lines
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add task labels
    for task, row in deltas.iterrows():
        plt.text(
            row['delta1'] + 0.005, 
            row['delta2'] + 0.005, 
            task,
            fontsize=8
        )
    
    plt.tight_layout()
    plt.show()

def main():
    # Read data
    df = pd.read_csv('pipeline/4_analysis/trial_results_aggregated.csv')
    
    # Filter out reasoning models
    reasoning_models = ['o1', 'o1-mini', 'deepseek-r1']
    df = df[~df['model_name'].isin(reasoning_models)]
    
    # Create plots for both metrics
    plot_delta_scatter(df, metric='top_prop_all')
    plot_delta_scatter(df, metric='top_prop_answered')

if __name__ == '__main__':
    main()


# Create a plot with:

# X-axis: Tasks (ordered by some metric)
# Y-axis: Performance
# Three lines for the conditions
# This shows where conditions diverge most
