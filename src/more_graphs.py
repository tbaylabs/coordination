from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.prepare_graph_data import prepare_graph_data

# Configure matplotlib to use a modern style
mpl.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'font.size': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.facecolor': 'white',
    'legend.edgecolor': 'black'
})

from src.plot_colors import MODEL_COLORS


# def create_chart_1(task_type='all'):
#     """Create line chart for LLaMA models - Top Option Proportion (All Responses)
    
#     Args:
#         task_type (str): Type of tasks included ('all', 'text_only', 'symbol_only')
#     """
#     data = prepare_graph_data(task_type=task_type)
#     import numpy as np
    
#     # Filter models - LLaMA models
#     selected_models = [
#         'llama-31-405b', 'llama-31-70b', 'llama-31-8b',
#     ]
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Get the metric data for selected models (using _mean suffix)
#     metric = 'top_prop_all'
#     metric_data = data[f'{metric}_mean'].loc[selected_models]
    
#     # Remove coordinate-COT for o1-mini and deepseek-r1
#     for model in ['o1-mini', 'deepseek-r1']:
#         if model in metric_data.index:
#             metric_data.loc[model, 'coordinate-COT'] = np.nan
    
#     # Sort models by their performance in the coordinate-CoT condition (descending)
#     sorted_models = metric_data['coordinate-COT'].sort_values(ascending=False).index.tolist()
            
#     # Plot each model's line in sorted order
#     for model in sorted_models:
#         if model in selected_models and model in metric_data.index:
#             # Get mean and SEM values
#             means = metric_data.loc[model]
#             sems = data[f'{metric}_sem'].loc[model]
            
#             # Plot with error bars
#             line, = ax.plot(means.index, means, 
#                           marker='o', label=model,
#                           color=MODEL_COLORS[model])
#             ax.errorbar(means.index, means, yerr=sems,
#                        fmt='none', ecolor=MODEL_COLORS[model],
#                        capsize=5, alpha=0.5)
            
#             # Add horizontal dotted line for reasoning models
#             if model in ['o1-mini', 'deepseek-r1']:
#                 # Get the coordinate value
#                 coord_value = metric_data.loc[model, 'coordinate']
#                 if not pd.isna(coord_value):
#                     # Draw dotted line from coordinate to coordinate-COT
#                     ax.hlines(y=coord_value, 
#                              xmin=1, xmax=2,  # coordinate is index 1, coordinate-COT is index 2
#                              colors=line.get_color(), 
#                              linestyles='dotted')
    
#     # Set plot properties
#     task_type_label = {
#         'all': 'All Task Variants',
#         'text_only': 'Text Task Variants',
#         'symbol_only': 'Symbol Task Variants'
#     }.get(task_type, 'All Task Variants')
    
#     ax.set_title(f"Mean Response Coordination of {task_type_label}\n(Top Response Proportion - All Responses)")
#     ax.set_xticks([0, 1, 2])
#     ax.set_xticklabels([
#         'control\n(No Coordination)', 
#         'coordinate\n(Elicit Answer Only)', 
#         'coordinate CoT\n(Elicit CoT)'
#     ], linespacing=1.5)
#     ax.set_xlabel('Condition and Context Type', labelpad=15)
#     ax.set_ylabel('Proportion')
#     ax.set_ylim(0, 1)
#     ax.grid(True)
#     # Add legend with note about deepseek-r1
#     legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     # Add note box below legend
#     ax.text(1.05, 0.85,  # Original position
#            '* deepseek-r1 does not support\nchain-of-thought prompting,\nso coordinate-CoT uses the\nsame value as coordinate\n\n† deepseek-v3 shows unusually\nlow coordination in the\ncoordinate condition',
#            transform=ax.transAxes,
#            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.9),
#            fontsize=11,
#            verticalalignment='top')
    
#     plt.tight_layout()
#     return fig

def create_chart_10(task_type='all'):
    """Create line chart comparing multiple models - Top Option Proportion (Answered Responses)
    
    Args:
        task_type (str): Type of tasks included ('all', 'text_only', 'symbol_only')
    """
    data = prepare_graph_data(task_type=task_type)
    import numpy as np
    
    # Filter models - expanded set
    base_models = [
        'gpt-4o', 'claude-35-sonnet', 'llama-31-8b'
    ]
    
    # Sort models by their performance on coordinate condition using _mean suffix
    selected_models = sorted(
        base_models,
        key=lambda model: data['top_prop_all_mean'].loc[model, 'coordinate'],
        reverse=True  # Highest first
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the metric data for selected models (using _mean suffix)
    metric = 'top_prop_all'
    metric_data = data[f'{metric}_mean'].loc[selected_models]
    
    # Plot each model's line
    for model in selected_models:
        if model in metric_data.index:
            # Get mean and SEM values
            means = metric_data.loc[model]
            sems = data[f'{metric}_sem'].loc[model]
            
            # Plot with error bars
            line, = ax.plot(means.index, means, 
                          marker='o', label=model,
                          color=MODEL_COLORS[model])
            ax.errorbar(means.index, means, yerr=sems,
                       fmt='none', ecolor=MODEL_COLORS[model],
                       capsize=5, alpha=0.5)
            
            # Add horizontal dotted line for GPT-4o in text_only condition
            if model == 'gpt-4o' and task_type == 'text_only':
                control_value = metric_data.loc[model, 'control']
                ax.hlines(y=control_value, 
                         xmin=0, xmax=2,  # From control (0) to coordinate-COT (2)
                         colors=MODEL_COLORS[model], 
                         linestyles='dotted')
            
            # Special handling for reasoning models (o1-mini and deepseek-r1)
            if model in ['o1-mini', 'deepseek-r1']:
                # Use coordinate value for coordinate-CoT point
                coord_value = metric_data.loc[model, 'coordinate']
                means[2] = coord_value  # coordinate-CoT is index 2
                
                # Plot with dotted line style
                line, = ax.plot(means.index, means, 
                              marker='o', label=f"{model}*",
                              color=MODEL_COLORS[model],
                              linestyle='dotted')
                ax.errorbar(means.index, means, yerr=sems,
                           fmt='none', ecolor=MODEL_COLORS[model],
                           capsize=5, alpha=0.5)
                
                # Add cross below deepseek-v3 point
                if model == 'deepseek-r1':
                    v3_value = metric_data.loc['deepseek-v3', 'coordinate']
                    ax.text(0.94, v3_value + 0.01, '†',
                           color=MODEL_COLORS['deepseek-v3'],
                           ha='center', va='top', fontsize=14)
    
    # Set plot properties
    task_type_label = {
        'all': 'All Task Variants',
        'text_only': 'Text Task Variants',
        'symbol_only': 'Symbol Task Variants'
    }.get(task_type, 'All Task Variants')
    
    ax.set_title(f"Model Comparison - {task_type_label}\n(Top Response Proportion - All Responses)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        'control\n(No Coordination)', 
        'coordinate\n(Elicit Answer Only)', 
        'coordinate-CoT\n(Elicit CoT)'
    ], linespacing=1.5)
    ax.set_xlabel('Condition and Context Type', labelpad=15)
    ax.set_ylabel('Proportion')
    ax.set_ylim(0.25, 1)
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add note box below legend
    ax.text(1.05, 0.4,  # Moved further down below legend
           '* no data for condition where\nreasoning models coordinate\nwithout CoT\n\n† deepseek-v3 shows unusually\nlow coordination in the\ncoordinate condition',
           transform=ax.transAxes,
           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.9),
           fontsize=11,
           verticalalignment='top')
    
    plt.tight_layout()
    return fig

def create_chart_9(task_type='all'):
    """Create line chart comparing GPT-4o and Claude Sonnet - Top Option Proportion (Answered Responses)
    
    Args:
        task_type (str): Type of tasks included ('all', 'text_only', 'symbol_only')
    """
    data = prepare_graph_data(task_type=task_type)
    import numpy as np
    
    # Filter models - only GPT-4o and Claude Sonnet
    base_models = [
        'gpt-4o', 'claude-35-sonnet'
    ]
    
    # Sort models by their performance on coordinate condition using _mean suffix
    selected_models = sorted(
        base_models,
        key=lambda model: data['top_prop_answered_mean'].loc[model, 'coordinate'],
        reverse=True  # Highest first
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the metric data for selected models (using _mean suffix)
    metric = 'top_prop_all'
    metric_data = data[f'{metric}_mean'].loc[selected_models]
    
    # Plot each model's line
    for model in selected_models:
        if model in metric_data.index:
            # Get mean and SEM values
            means = metric_data.loc[model]
            sems = data[f'{metric}_sem'].loc[model]
            
            # Plot with error bars
            line, = ax.plot(means.index, means, 
                          marker='o', label=model,
                          color=MODEL_COLORS[model])
            ax.errorbar(means.index, means, yerr=sems,
                       fmt='none', ecolor=MODEL_COLORS[model],
                       capsize=5, alpha=0.5)
            
            # Add horizontal dotted line for GPT-4o in text_only condition
            if model == 'gpt-4o' and task_type == 'text_only':
                control_value = metric_data.loc[model, 'control']
                ax.hlines(y=control_value, 
                         xmin=0, xmax=2,  # From control (0) to coordinate-COT (2)
                         colors=MODEL_COLORS[model], 
                         linestyles='dotted')
    
    # Set plot properties
    task_type_label = {
        'all': 'All Task Variants',
        'text_only': 'Text Task Variants',
        'symbol_only': 'Symbol Task Variants'
    }.get(task_type, 'All Task Variants')
    
    ax.set_title(f"GPT-4o vs Claude Sonnet - {task_type_label}\n(Top Response Proportion - Answered Responses)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        'control\n(No Coordination)', 
        'coordinate\n(Elicit Answer Only)', 
        'coordinate-CoT\n(Elicit CoT)'
    ], linespacing=1.5)
    ax.set_xlabel('Condition and Context Type', labelpad=15)
    ax.set_ylabel('Proportion')
    ax.set_ylim(0.25, 1)
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_chart_2(task_type='all'):
    """Create line chart for LLaMA models - Top Option Proportion (Answered Responses)
    
    Args:
        task_type (str): Type of tasks included ('all', 'text_only', 'symbol_only')
    """
    data = prepare_graph_data(task_type=task_type)
    import numpy as np
    
    # Filter models - LLaMA models
    selected_models = [
        'llama-31-405b', 'llama-31-70b', 'llama-31-8b',
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the metric data for selected models (using _mean suffix)
    metric = 'top_prop_answered'
    metric_data = data[f'{metric}_mean'].loc[selected_models]
    
    # Remove coordinate-COT for o1-mini and deepseek-r1
    for model in ['o1-mini', 'deepseek-r1']:
        if model in metric_data.index:
            metric_data.loc[model, 'coordinate-COT'] = np.nan
    
    # Sort models by their performance in the coordinate-CoT condition (descending)
    sorted_models = metric_data['coordinate-COT'].sort_values(ascending=False).index.tolist()
            
    # Plot each model's line in sorted order
    for model in sorted_models:
        if model in selected_models and model in metric_data.index:
            # Get mean and SEM values
            means = metric_data.loc[model]
            sems = data[f'{metric}_sem'].loc[model]
            
            # Plot with error bars
            line, = ax.plot(means.index, means, 
                          marker='o', label=model,
                          color=MODEL_COLORS[model])
            ax.errorbar(means.index, means, yerr=sems,
                       fmt='none', ecolor=MODEL_COLORS[model],
                       capsize=5, alpha=0.5)
            
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
    task_type_label = {
        'all': 'All Task Variants',
        'text_only': 'Text Task Variants',
        'symbol_only': 'Symbol Task Variants'
    }.get(task_type, 'All Task Variants')
    
    ax.set_title(f"Mean Response Coordination of {task_type_label}\n(Top Response Proportion - Answered Responses)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        'control\n(No Coordination)', 
        'coordinate\n(Elicit Answer Only)', 
        'coordinate-CoT\n(Elicit CoT)'
    ], linespacing=1.5)
    ax.set_xlabel('Condition and Context Type', labelpad=15)
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_charts_3_and_4(task_type='all'):
    """Create line charts for Sonnet, reasoning models and GPT-4o
    
    Args:
        task_type (str): Type of tasks included ('all', 'text_only', 'symbol_only')
    """
    data = prepare_graph_data(task_type=task_type)
    import numpy as np
    
    # Filter models - Sonnet + GPT-4o
    selected_models = [
        'claude-35-sonnet', 'gpt-4o'
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
        # Get the metric data for selected models (using _mean suffix)
        metric_data = data[f'{metric}_mean'].loc[selected_models]
        
        # Remove coordinate-COT for o1-mini and deepseek-r1
        for model in ['o1-mini', 'deepseek-r1']:
            if model in metric_data.index:
                metric_data.loc[model, 'coordinate-COT'] = np.nan
        
        # Sort models by their performance in the coordinate condition (descending)
        sorted_models = metric_data['coordinate'].sort_values(ascending=False).index.tolist()
                
        # Plot each model's line in sorted order
        for model in sorted_models:
            if model in selected_models and model in metric_data.index:
                # Get mean and SEM values
                means = metric_data.loc[model]
                sems = data[f'{metric}_sem'].loc[model]
                
                # Plot with error bars
                line, = ax.plot(means.index, means, 
                              marker='o', label=model,
                              color=MODEL_COLORS[model])
                ax.errorbar(means.index, means, yerr=sems,
                           fmt='none', ecolor=MODEL_COLORS[model],
                           capsize=5, alpha=0.5)
                
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
        # Add task type to title
        task_type_label = {
            'all': 'All Task Variants',
            'text_only': 'Text Task Variants',
            'symbol_only': 'Symbol Task Variants'
        }.get(task_type, 'All Task Variants')
        
        ax.set_title(f"Mean Response Coordination of {task_type_label}\n(Top Response Proportion)")
        # Set custom x-axis labels with line breaks and more spacing
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([
            'control\n(No Coordination)', 
            'coordinate\n(Elicit Answer Only)', 
            'coordinate-CoT\n(Elicit CoT)'
        ], linespacing=1.5)
        ax.set_xlabel('Condition and Context Type', labelpad=15)
        ax.set_ylabel('Proportion')
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_charts_5_and_6(task_type='all'):
    """Create line charts for LLaMA models
    
    Args:
        task_type (str): Type of tasks included ('all', 'text_only', 'symbol_only')
    """
    data = prepare_graph_data(task_type=task_type)
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
        # Get the metric data for selected models (using _mean suffix)
        metric_data = data[f'{metric}_mean'].loc[selected_models]
        
        # Plot each model's line
        for model in selected_models:
            if model in metric_data.index:
                # Get mean and SEM values
                means = metric_data.loc[model]
                sems = data[f'{metric}_sem'].loc[model]
                
                # Plot with error bars
                line, = ax.plot(means.index, means, 
                              marker='o', label=model,
                              color=MODEL_COLORS[model])
                ax.errorbar(means.index, means, yerr=sems,
                           fmt='none', ecolor=MODEL_COLORS[model],
                           capsize=5, alpha=0.5)
                
                # Just plot the line without horizontal dotted lines
                pass
        
        # Set plot properties
        # Add task type to title
        task_type_label = {
            'all': 'All Options',
            'text_only': 'Text Options',
            'symbol_only': 'Symbol Options'
        }.get(task_type, 'All Options')
        
        ax.set_title(f"{title} - {task_type_label}")
        # Set custom x-axis labels with line breaks and more spacing
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([
            'control\n(No Coordination)', 
            'coordinate\n(Elicit Answer Only)', 
            'coordinate-CoT\n(Elicit CoT)'
        ], linespacing=1.5)
        ax.set_xlabel('Condition and Context Type', labelpad=15)
        ax.set_ylabel('Proportion')
        ax.set_ylim(0.3, 0.7)
        ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # Add chart number in top right
        ax.text(1.02, 1.02, '#5' if metric == 'top_prop_all' else '#6', 
               transform=ax.transAxes, ha='left', va='bottom', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_chart_7(task_type='all'):
    """Create line chart for Deepseek V3, Sonnet and GPT-4o - Top Option Proportion (All Responses)
    
    Args:
        task_type (str): Type of tasks included ('all', 'text_only', 'symbol_only')
    """
    data = prepare_graph_data(task_type=task_type)
    import numpy as np
    
    # Filter models - Sonnet and GPT-4o
    base_models = [
        'claude-35-sonnet', 'gpt-4o'
    ]
    
    # Sort models by their performance on coordinate condition using _mean suffix
    selected_models = sorted(
        base_models,
        key=lambda model: data['top_prop_all_mean'].loc[model, 'coordinate'],
        reverse=True  # Highest first
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the metric data for selected models (using _mean suffix)
    metric = 'top_prop_all'
    metric_data = data[f'{metric}_mean'].loc[selected_models]
    
    # Plot each model's line
    for model in selected_models:
        if model in metric_data.index:
            # Get mean and SEM values
            means = metric_data.loc[model]
            sems = data[f'{metric}_sem'].loc[model]
            
            # Plot with error bars
            line, = ax.plot(means.index, means, 
                          marker='o', label=model,
                          color=MODEL_COLORS[model])
            ax.errorbar(means.index, means, yerr=sems,
                       fmt='none', ecolor=MODEL_COLORS[model],
                       capsize=5, alpha=0.5)
            
            # No horizontal dotted lines for this chart
            pass
    
    # Set plot properties
    task_type_label = {
        'all': 'All Task Variants',
        'text_only': 'Text Task Variants',
        'symbol_only': 'Symbol Task Variants'
    }.get(task_type, 'All Task Variants')
    
    ax.set_title(f"Mean Response Coordination of {task_type_label}\n(Top Response Proportion - All Responses)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        'control\n(No Coordination)', 
        'coordinate\n(Elicit Answer Only)', 
        'coordinate-CoT\n(Elicit CoT)'
    ], linespacing=1.5)
    ax.set_xlabel('Condition and Context Type', labelpad=15)
    ax.set_ylabel('Proportion')
    ax.set_ylim(0.25, 1)
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_chart_8(task_type='all'):
    """Create line chart for Deepseek V3, Sonnet and GPT-4o - Top Option Proportion (Answered Responses)
    
    Args:
        task_type (str): Type of tasks included ('all', 'text_only', 'symbol_only')
    """
    data = prepare_graph_data(task_type=task_type)
    import numpy as np
    
    # Filter models - Sonnet and GPT-4o
    base_models = [
        'claude-35-sonnet', 'gpt-4o'
    ]
    
    # Sort models by their performance on coordinate condition using _mean suffix
    selected_models = sorted(
        base_models,
        key=lambda model: data['top_prop_all_mean'].loc[model, 'coordinate'],
        reverse=True  # Highest first
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the metric data for selected models (using _mean suffix)
    metric = 'top_prop_answered'
    metric_data = data[f'{metric}_mean'].loc[selected_models]
    
    # Plot each model's line
    for model in selected_models:
        if model in metric_data.index:
            # Get mean and SEM values
            means = metric_data.loc[model]
            sems = data[f'{metric}_sem'].loc[model]
            
            # Plot with error bars
            line, = ax.plot(means.index, means, 
                          marker='o', label=model,
                          color=MODEL_COLORS[model])
            ax.errorbar(means.index, means, yerr=sems,
                       fmt='none', ecolor=MODEL_COLORS[model],
                       capsize=5, alpha=0.5)
            
            # No horizontal dotted lines for this chart
            pass
    
    # Set plot properties
    task_type_label = {
        'all': 'All Options',
        'text_only': 'Text Options',
        'symbol_only': 'Symbol Options'
    }.get(task_type, 'All Options')
    
    ax.set_title(f"Mean Response Coordination of {task_type_label}\n(Top Response Proportion - Answered Responses)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        'control\n(No Coordination)', 
        'coordinate\n(Elicit Answer Only)', 
        'coordinate-CoT\n(Elicit CoT)'
    ], linespacing=1.5)
    ax.set_xlabel('Condition and Context Type', labelpad=15)
    ax.set_ylabel('Proportion')
    ax.set_ylim(0.25, 1)
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig
