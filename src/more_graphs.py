from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def create_charts_1_and_2():
    """Create line charts for top_prop_all and top_prop_answered metrics"""
    # Prepare the data
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

    # Now create the charts
    """Create line charts for top_prop_all and top_prop_answered metrics"""
    import numpy as np
    
    # Filter models and conditions
    selected_models = [
        'llama-31-405b', 'llama-31-70b', 'llama-31-8b', 
        'claude-35-sonnet', 'o1-mini', 'deepseek-r1'
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
        
        # Plot each model's line
        for model in selected_models:
            if model in metric_data.index:
                line, = ax.plot(metric_data.columns, metric_data.loc[model], 
                              marker='o', label=model)
                
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


