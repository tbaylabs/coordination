from pathlib import Path
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
    
    # Plot lines for each condition using the correct column names
    for condition in ['control', 'coordinate', 'coordinate-COT']:
        if condition not in task_data.columns:
            print(f"Warning: Condition '{condition}' not found in data for model '{model_name}'. Skipping this condition.")
            continue
            
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
    # Filter to only show selected models
    selected_models = [
        'gpt-4o',
        'claude-35-sonnet'
    ]
    models = [model for model in selected_models if model in df['model_name'].unique()]
    
    # Create a plot for each model
    figures = []
    for model in models:
        fig = plot_condition_task_interaction(df, metric=metric, model_name=model)
        figures.append(fig)
    
    return figures

def plot_models_by_condition(df, metric='top_prop_all'):
    """
    Create three plots showing model performance across tasks for each condition.
    
    Args:
        df (pd.DataFrame): Raw experiment data
        metric (str): Metric to plot ('top_prop_all' or 'top_prop_answered')
        
    Returns:
        list: List of matplotlib.figure.Figure objects (one per condition)
    """
    # Prepare data
    df = add_experiment_conditions(df)
    
    # Filter to only show selected models
    selected_models = [
        'gpt-4o',
        'claude-35-sonnet'
    ]
    models = [model for model in selected_models if model in df['model_name'].unique()]
    
    # Create one plot per condition
    figures = []
    for condition in ['control', 'coordinate', 'coordinate-COT']:
        # Set up plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.set_style("whitegrid")
        
        # Group by task and model for this condition
        task_data = df[df['experiment'] == condition].groupby(
            ['task_options', 'model_name'], 
            observed=True
        )[metric].mean().unstack()
        
        # Load options lists
        import json
        with open(Path(__file__).parent / '..' / 'prompts' / 'options_lists.json') as f:
            options_lists = json.load(f)
        
        # Use fixed task ordering with proper prefixes
        task_order = [
            "letters",
            "colours-text",
            "shapes-3-text",
            "shapes-2-text",
            "shapes-1-text",
            "kanji-nature-english",
            "kanji-random-english",
            "emoji-3-text",
            "emoji-2-text",
            "emoji-1-text",
            "emoji-1",
            "emoji-2",
            "emoji-3",
            "kanji-random",
            "kanji-nature",
            "shapes-1-icon",
            "shapes-2-icon",
            "shapes-3-icon",
            "colours",
            "numbers"
        ]
        
        # Only include tasks that exist in the data
        task_order = [task for task in task_order if task.split()[-1] in task_data.index]
        
        # Create mapping from original task names to display names
        task_name_map = {}
        for task in task_data.index:
            if task in options_lists:
                if task == 'letters':
                    task_name_map[task] = ", ".join(options_lists[task]) + " " + task
                elif task == 'numbers':
                    task_name_map[task] = ", ".join(options_lists[task]) + " " + task
                elif task.endswith(('-text', '-english')):
                    # For text tasks, use original name
                    task_name_map[task] = task
                else:
                    # For icon tasks, use original name (we'll handle emojis separately)
                    task_name_map[task] = task
            else:
                task_name_map[task] = task
        
        # Create display names for all tasks in order
        display_order = []
        for task in task_order:
            if task in task_name_map:
                display_order.append(task_name_map[task])
        
        # Reindex using original task names but set display names
        task_data = task_data.reindex([t for t in task_order if t in task_data.index])
        task_data.index = [task_name_map[t] for t in task_data.index]
        
        # Plot each model's performance
        for model in models:
            if model in task_data.columns:
                ax.plot(
                    task_data.index,
                    task_data[model],
                    marker='o',
                    label=model
                )
        
        # Add labels and title
        ax.set(
            xlabel='Tasks (ordered by control performance)',
            ylabel=f'{metric.replace("_", " ").title()}',
            title=f'Model Performance Across Tasks - {condition} Condition',
            xticks=range(len(task_data.index))
        )
        
        # Rotate x-labels for readability
        plt.xticks(rotation=90)
        
        # Add legend
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid lines
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures

