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
