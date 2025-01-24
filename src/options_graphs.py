import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.prepare_graph_data import add_experiment_conditions, get_filtered_data

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
        
    Returns:
        matplotlib.figure.Figure: The figure object containing the plot
    """
    # Calculate deltas
    deltas = calculate_task_deltas(df, metric)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Create scatter plot
    sns.scatterplot(
        x='delta1', 
        y='delta2', 
        data=deltas,
        s=100,
        alpha=0.8,
        ax=ax
    )
    
    # Add labels and title
    ax.set(
        xlabel='Δ1: Coordinate - Control',
        ylabel='Δ2: Coordinate-COT - Coordinate',
        title=f'Task Performance Deltas ({metric.replace("_", " ").title()})'
    )
    
    # Add quadrant lines
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add task labels
    for task, row in deltas.iterrows():
        ax.text(
            row['delta1'] + 0.005, 
            row['delta2'] + 0.005, 
            task,
            fontsize=8
        )
    
    plt.tight_layout()
    return fig

def plot_condition_task_interaction(df, metric='top_prop_all', model_name=None):
    """
    Create interaction plot showing condition effects across tasks.
    
    Args:
        df (pd.DataFrame): Raw experiment data
        metric (str): Metric to plot ('top_prop_all' or 'top_prop_answered')
        model_name (str, optional): Specific model to plot. If None, plots all models.
        
    Returns:
        matplotlib.figure.Figure: The figure object containing the plot
    """
    # Prepare data
    df = add_experiment_conditions(df)
    
    # Filter for specific model if provided
    if model_name:
        df = df[df['model_name'] == model_name]
    
    # Group by task, experiment, and optionally model
    group_cols = ['task_options', 'experiment']
    task_data = df.groupby(group_cols, observed=True)[metric].mean().unstack()
    
    # Sort tasks by control condition performance
    task_data = task_data.sort_values('control', ascending=False)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Plot lines for each condition
    for condition in ['control', 'coordinate', 'coordinate-COT']:
        ax.plot(
            task_data.index,
            task_data[condition],
            marker='o',
            label=condition
        )
    
    # Add labels and title
    title = 'Condition-Task Interaction Plot'
    if model_name:
        title = f'{title} - {model_name}'
    
    ax.set(
        xlabel='Tasks (ordered by control performance)',
        ylabel=f'{metric.replace("_", " ").title()}',
        title=title,
        xticks=range(len(task_data.index))
    )
    
    # Rotate x-labels for readability
    plt.xticks(rotation=90)
    
    # Add legend
    ax.legend(title='Condition')
    
    # Add grid lines
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_all_models_condition_task_interaction(df, metric='top_prop_all'):
    """
    Create condition-task interaction plots for each model.
    
    Args:
        df (pd.DataFrame): Raw experiment data
        metric (str): Metric to plot ('top_prop_all' or 'top_prop_answered')
        
    Returns:
        list: List of matplotlib.figure.Figure objects
    """
    # Get unique models
    models = df['model_name'].unique()
    
    # Create a plot for each model
    figures = []
    for model in models:
        fig = plot_condition_task_interaction(df, metric=metric, model_name=model)
        figures.append(fig)
    
    return figures

