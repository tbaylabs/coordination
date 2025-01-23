from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Define consistent color palette for all models
MODEL_COLORS = {
    'llama-31-405b': '#1f77b4',  # blue
    'llama-31-70b': '#2ca02c',   # green
    'llama-31-8b': '#d62728',    # red
    'llama-33-70b': '#9467bd',   # purple
    'gpt-4o': '#ff7f0e',         # orange
    'claude-35-sonnet': '#8c564b',  # brown
    'o1-mini': '#e377c2',        # pink
    'deepseek-r1': '#17becf'     # cyan
}

def prepare_graph_data():
    """Prepare and return the processed data for chart creation"""
    # Read and process data
    df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')

    # Create experiment condition mapping with ordering
    conditions = {
        ('control', 'none'): 'control',
        ('coordinate', 'none'): 'coordinate',
        ('coordinate', 'step-by-step'): 'coordinate-COT'
    }
    
    # Add experiment condition to dataframe
    df['experiment'] = df.apply(lambda row: conditions.get((row['task_instruction'], row['task_reasoning']), 'other'), axis=1)
    
    # Remove all rows with 'other' experiment
    df = df[df['experiment'] != 'other']
    
    # Create ordered category for experiment conditions
    experiment_order = ['control', 'coordinate', 'coordinate-COT']
    df['experiment'] = pd.Categorical(df['experiment'], categories=experiment_order, ordered=True)

    # Calculate top_prop_answered
    df['top_prop_answered'] = df['top_option_count'] / df['answered_count']

    # Prepare data for different metrics
    data = {
        'top_prop_answered': df.groupby(['model_name', 'experiment'])['top_prop_answered'].mean().unstack(),
        'top_prop_all': df.groupby(['model_name', 'experiment'])['top_prop_all'].mean().unstack(),
        'convergence_all': df.groupby(['model_name', 'experiment'])['convergence_all'].mean().unstack()
    }
    
    return data

def create_charts_1_and_2():
    """Create line charts for LLaMA models and GPT-4o"""
    print("Creating chart 1 and 2")
    data = prepare_graph_data()
    import numpy as np
    
    # Filter models - LLaMA models + GPT-4o
    selected_models = [
        'llama-31-405b', 'llama-31-70b', 'llama-31-8b',
        'llama-33-70b'
    ]
    
    # Prepare data for plotting
    metrics = {
        'top_prop_all': 'Top Option Proportion (All Responses)',
        'top_prop_answered': 'Top Option Proportion (Answered Responses)'
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot each metric
    for ax, (metric, title) in zip([ax1, ax2], metrics.items()):
        # Get the metric data for selected models
        metric_data = data[metric].loc[selected_models]
        
        # Remove coordinate-COT for o1-mini and deepseek-r1
        for model in ['o1-mini', 'deepseek-r1']:
            if model in metric_data.index:
                metric_data.loc[model, 'coordinate-COT'] = np.nan
        
        # Sort models by their performance in the coordinate condition (ascending)
        # So highest performance is at top of legend
        sorted_models = metric_data['coordinate'].sort_values(ascending=True).index.tolist()
                
        # Plot each model's line in sorted order
        for model in sorted_models:
            if model in selected_models and model in metric_data.index:
                line, = ax.plot(metric_data.columns, metric_data.loc[model], 
                              marker='o', label=model,
                              color=MODEL_COLORS[model])
                
                # Add horizontal dotted line for reasoning models
                if model in ['o1-mini', 'deepseek-r1']:
                    # Get the coordinate value
                    coord_value = metric_data.loc[model, 'coordinate']
                    if not pd.isna(coord_value):
                        # Draw dotted line from coordinate to coordinate-COT
                        ax.hlines(y=coord_value, 
                                 xmin=1, xmax=2,  # coordinate is index 1, coordinate-COT is index 2
                                 colors=line.get_color(), 
                                 linestyles='dotted')
        
        # Set plot properties
        ax.set_title(title)
        ax.set_xlabel('Condition')
        ax.set_ylabel('Proportion')
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_charts_3_and_4():
    """Create line charts for Sonnet, reasoning models and GPT-4o"""
    print("Creating chart 3 and 4")
    data = prepare_graph_data()
    import numpy as np
    
    # Filter models - Sonnet + reasoning models + GPT-4o
    selected_models = [
        'claude-35-sonnet', 'o1-mini', 'deepseek-r1',
        'gpt-4o'
    ]
    
    # Prepare data for plotting
    metrics = {
        'top_prop_all': 'Top Option Proportion (All Responses)',
        'top_prop_answered': 'Top Option Proportion (Answered Responses)'
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot each metric
    for ax, (metric, title) in zip([ax1, ax2], metrics.items()):
        # Get the metric data for selected models
        metric_data = data[metric].loc[selected_models]
        
        # Remove coordinate-COT for o1-mini and deepseek-r1
        for model in ['o1-mini', 'deepseek-r1']:
            if model in metric_data.index:
                metric_data.loc[model, 'coordinate-COT'] = np.nan
        
        # Sort models by their performance in the coordinate condition (descending)
        sorted_models = metric_data['coordinate'].sort_values(ascending=False).index.tolist()
                
        # Plot each model's line in sorted order
        for model in sorted_models:
            if model in selected_models and model in metric_data.index:
                line, = ax.plot(metric_data.columns, metric_data.loc[model], 
                              marker='o', label=model,
                              color=MODEL_COLORS[model])
                
                # Add horizontal dotted line for reasoning models
                if model in ['o1-mini', 'deepseek-r1']:
                    # Get the coordinate value
                    coord_value = metric_data.loc[model, 'coordinate']
                    if not pd.isna(coord_value):
                        # Draw dotted line from coordinate to coordinate-COT
                        ax.hlines(y=coord_value, 
                                 xmin=1, xmax=2,  # coordinate is index 1, coordinate-COT is index 2
                                 colors=line.get_color(), 
                                 linestyles='dotted')
        
        # Set plot properties
        ax.set_title(title)
        ax.set_xlabel('Condition')
        ax.set_ylabel('Proportion')
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_charts_5_and_6():
    """Create line charts for LLaMA models"""
    print("Creating chart 5 and 6")
    data = prepare_graph_data()
    import numpy as np
    
    # Filter models - LLaMA models only
    selected_models = [
        'llama-31-405b', 'llama-31-70b', 'llama-31-8b'
    ]
    
    # Prepare data for plotting
    metrics = {
        'top_prop_all': 'Top Option Proportion (All Responses)',
        'top_prop_answered': 'Top Option Proportion (Answered Responses)'
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot each metric
    for ax, (metric, title) in zip([ax1, ax2], metrics.items()):
        # Get the metric data for selected models
        metric_data = data[metric].loc[selected_models]
        
        # Plot each model's line
        for model in selected_models:
            if model in metric_data.index:
                line, = ax.plot(metric_data.columns, metric_data.loc[model], 
                              marker='o', label=model,
                              color=MODEL_COLORS[model])
                
                # Get the control value and draw horizontal dotted line across all conditions
                control_value = metric_data.loc[model, 'control']
                ax.hlines(y=control_value, 
                         xmin=0, xmax=2,  # From control (0) to coordinate-COT (2)
                         colors=line.get_color(), 
                         linestyles='dotted')
        
        # Set plot properties
        ax.set_title(title)
        ax.set_xlabel('Condition')
        ax.set_ylabel('Proportion')
        ax.set_ylim(0.3, 0.7)
        ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_charts_7_and_8():
    """Create line charts for LLaMA 33 70b, Sonnet and GPT-4o"""
    print("Creating chart 7 and 8")
    data = prepare_graph_data()
    import numpy as np
    
    # Filter models - LLaMA 33 70b, Sonnet and GPT-4o
    base_models = [
        'llama-33-70b', 'claude-35-sonnet', 'gpt-4o'
    ]
    
    # Sort models by their performance on coordinate condition
    selected_models = sorted(
        base_models,
        key=lambda model: data['top_prop_all'].loc[model, 'coordinate'],
        reverse=True  # Highest first
    )
    
    # Prepare data for plotting
    metrics = {
        'top_prop_all': 'Top Option Proportion (All Responses)',
        'top_prop_answered': 'Top Option Proportion (Answered Responses)'
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot each metric
    for ax, (metric, title) in zip([ax1, ax2], metrics.items()):
        # Get the metric data for selected models
        metric_data = data[metric].loc[selected_models]
        
        # Plot each model's line
        for model in selected_models:
            if model in metric_data.index:
                line, = ax.plot(metric_data.columns, metric_data.loc[model], 
                              marker='o', label=model,
                              color=MODEL_COLORS[model])
                
                # Get the control value and draw horizontal dotted line across all conditions
                control_value = metric_data.loc[model, 'control']
                ax.hlines(y=control_value, 
                         xmin=0, xmax=2,  # From control (0) to coordinate-COT (2)
                         colors=line.get_color(), 
                         linestyles='dotted')
        
        # Set plot properties
        ax.set_title(title)
        ax.set_xlabel('Condition')
        ax.set_ylabel('Proportion')
        ax.set_ylim(0.25, 1)
        ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig
