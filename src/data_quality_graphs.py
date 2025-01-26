from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from matplotlib import font_manager

# Configure matplotlib to use a font that supports emoji
font_path = font_manager.findfont(font_manager.FontProperties(family=['DejaVu Sans', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji']))
mpl.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
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

def create_llm_extract_chart():
    """Create chart showing LLM extract proportions across conditions"""
    # Read the raw data
    df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')
    
    # Create experiment condition mapping
    conditions = {
        ('control', 'none'): 'control',
        ('coordinate', 'none'): 'coordinate',
        ('coordinate', 'step-by-step'): 'coordinate-COT'
    }
    df['experiment'] = df.apply(lambda row: conditions.get((row['task_instruction'], row['task_reasoning']), 'other'), axis=1)
    
    # Remove 'other' experiments
    df = df[df['experiment'] != 'other']
    
    # Group by model and experiment, calculate mean LLM extract proportion
    llm_data = df.groupby(['model_name', 'experiment'])['extracted_by_llm_prop'].mean().unstack()
    
    # Sort models by their LLM extract proportion in the coordinate condition
    llm_data = llm_data.loc[llm_data['coordinate'].sort_values(ascending=False).index]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each model's data
    for model in llm_data.index:
        ax.plot(llm_data.columns, llm_data.loc[model], marker='o', label=model)
    
    # Set plot properties
    ax.set_title('LLM Extract Proportion by Model and Condition')
    ax.set_xlabel('Condition')
    ax.set_ylabel('LLM Extract Proportion')
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def get_llm_extract_data():
    """Return the processed LLM extract data for use in notebooks"""
    # Read the raw data
    df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')
    
    # Create experiment condition mapping
    conditions = {
        ('control', 'none'): 'control',
        ('coordinate', 'none'): 'coordinate',
        ('coordinate', 'step-by-step'): 'coordinate-COT'
    }
    df['experiment'] = df.apply(lambda row: conditions.get((row['task_instruction'], row['task_reasoning']), 'other'), axis=1)
    
    # Remove 'other' experiments
    df = df[df['experiment'] != 'other']
    
    # Group by model and experiment, calculate mean LLM extract proportion
    llm_data = df.groupby(['model_name', 'experiment'])['extracted_by_llm_prop'].mean().unstack()
    
    # Sort models by their LLM extract proportion in the coordinate condition
    llm_data = llm_data.loc[llm_data['coordinate'].sort_values(ascending=False).index]
    
    return llm_data

def create_llm_extract_chart():
    """Create and return figure showing LLM extract proportions across conditions"""
    llm_data = get_llm_extract_data()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each model's data
    for model in llm_data.index:
        ax.plot(llm_data.columns, llm_data.loc[model], marker='o', label=model)
    
    # Set plot properties
    ax.set_title('LLM Extract Proportion by Model and Condition')
    ax.set_xlabel('Condition')
    ax.set_ylabel('LLM Extract Proportion')
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def get_token_count_data(metric='avg_token_count'):
    """
    Return processed token count data for use in notebooks
    
    Args:
        metric (str): Which token metric to use ('avg_token_count' or 'median_token_count')
    """
    # Read the raw data
    df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')
    
    # Create experiment condition mapping
    conditions = {
        ('control', 'none'): 'control',
        ('coordinate', 'none'): 'coordinate',
        ('coordinate', 'step-by-step'): 'coordinate-COT'
    }
    df['experiment'] = df.apply(lambda row: conditions.get((row['task_instruction'], row['task_reasoning']), 'other'), axis=1)
    
    # Remove 'other' experiments
    df = df[df['experiment'] != 'other']
    
    # Group by model and experiment, calculate mean of the selected metric
    token_data = df.groupby(['model_name', 'experiment'])[metric].mean().unstack()
    
    # Sort models by their token count in the coordinate condition
    token_data = token_data.loc[token_data['coordinate'].sort_values(ascending=False).index]
    
    return token_data

def create_token_count_chart(metric='avg_token_count'):
    """
    Create and return figure showing token counts across models and conditions
    
    Args:
        metric (str): Which token metric to plot ('avg_token_count' or 'median_token_count')
    """
    token_data = get_token_count_data(metric)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each model's data
    for model in token_data.index:
        ax.plot(token_data.columns, token_data.loc[model], marker='o', label=model)
    
    # Set plot properties
    metric_name = 'Average' if metric == 'avg_token_count' else 'Median'
    ax.set_title(f'{metric_name} Token Count by Model and Condition')
    ax.set_xlabel('Condition')
    ax.set_ylabel(f'{metric_name} Token Count')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_before_answer_token_count_chart(metric='avg_before_answer_token_count'):
    """
    Create and return figure showing token counts before answer appears across models and conditions
    
    Args:
        metric (str): Which token metric to plot ('avg_before_answer_token_count' or 'median_before_answer_token_count')
    """
    # Read the raw data
    df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')
    
    # Create experiment condition mapping
    conditions = {
        ('control', 'none'): 'control',
        ('coordinate', 'none'): 'coordinate',
        ('coordinate', 'step-by-step'): 'coordinate-COT'
    }
    df['experiment'] = df.apply(lambda row: conditions.get((row['task_instruction'], row['task_reasoning']), 'other'), axis=1)
    
    # Remove 'other' experiments
    df = df[df['experiment'] != 'other']
    
    # Group by model and experiment, calculate mean of the selected metric
    token_data = df.groupby(['model_name', 'experiment'])[metric].mean().unstack()
    
    # Sort models by their token count in the coordinate condition
    token_data = token_data.loc[token_data['coordinate'].sort_values(ascending=False).index]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each model's data
    for model in token_data.index:
        ax.plot(token_data.columns, token_data.loc[model], marker='o', label=model)
    
    # Set plot properties
    metric_name = 'Average' if metric == 'avg_before_answer_token_count' else 'Median'
    ax.set_title(f'{metric_name} Token Count Before Answer by Model and Condition')
    ax.set_xlabel('Condition')
    ax.set_ylabel(f'{metric_name} Token Count Before Answer')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_task_difficulty_chart():
    """
    Create and return figure showing relative difficulty of different option tasks
    based on the change in top proportion between control and coordinate conditions
    for non-reasoning models.
    """
    # Read the raw data
    df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')
    
    # Define non-reasoning models
    non_reasoning_models = [
        "claude-3-haiku", "llama-31-405b", "llama-31-70b", "llama-31-8b", 
        "llama-32-3b", "llama-32-1b", "claude-35-haiku", "claude-35-sonnet", 
        "llama-33-70b", "gpt-4o", "gemini", "gemini-flash"
    ]
    
    # Filter for non-reasoning models
    df = df[df['model_name'].isin(non_reasoning_models)]
    
    # Create experiment condition mapping
    conditions = {
        ('control', 'none'): 'control',
        ('coordinate', 'none'): 'coordinate',
        ('coordinate', 'step-by-step'): 'coordinate-COT'
    }
    df['experiment'] = df.apply(lambda row: conditions.get((row['task_instruction'], row['task_reasoning']), 'other'), axis=1)
    
    # Remove 'other' experiments
    df = df[df['experiment'] != 'other']
    
    # Pivot to get control and coordinate conditions side by side
    pivot_df = df.pivot_table(
        index=['task_options', 'model_name'],
        columns='experiment',
        values='top_prop_all'
    ).reset_index()
    
    # Calculate delta between coordinate and control
    pivot_df['delta'] = pivot_df['coordinate'] - pivot_df['control']
    
    # Load options lists
    with open(Path(__file__).parent / '..' / 'prompts' / 'options_lists.json', 'r') as f:
        options_lists = json.load(f)
    
    # Group by task options and calculate mean delta
    task_difficulty = pivot_df.groupby('task_options')['delta'].mean().sort_values()
    
    # Create labels showing the options
    labels = []
    for task in task_difficulty.index:
        options = options_lists[task]
        # For text options, just show first option followed by "..."
        if any(c.isalpha() for c in ''.join(options)):  # Check if any option contains letters
            label = f"{task}\n({options[0]}...)"
        else:
            label = f"{task}\n({', '.join(options)})"
        labels.append(label)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create horizontal bar plot
    task_difficulty.plot(kind='barh', ax=ax, color='steelblue')
    
    # Set y-tick labels with options
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    
    # Set plot properties
    ax.set_title('Relative Difficulty of Option Tasks\n(Change in Top Proportion: Coordinate vs Control)')
    ax.set_xlabel('Average Change in Top Proportion\n(Coordinate - Control)')
    ax.set_ylabel('Task Options and Choices')
    ax.grid(True)
    
    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    return fig
from src.plot_colors import MODEL_COLORS
