"""
Functions for creating visualizations of experimental results.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.analysis.model_filters import (
    get_model_order,
    detect_model_family,
)

def set_comparison_style():
    """Set consistent style for comparisons (with/without reasoning)."""
    plt.style.use('default')
    sns.set_context("talk")
    
    return {
        "without": "#20B2AA",  # Light sea green
        "with": "#DA70D6"      # Orchid
    }

def set_change_style():
    """Set consistent style for changes (increases/decreases)."""
    plt.style.use('default')
    sns.set_context("talk")
    
    return {
        "increase": "#2ECC71",  # Green for increases
        "decrease": "#E74C3C",  # Red for decreases
        "neutral": "#95A5A6"    # Gray for no change
    }

def get_formatted_family_name(family):
    """Convert internal family name to display name."""
    family_map = {
        'llama_31': 'Llama 3.1',
        'llama_32': 'Llama 3.2'
    }
    return family_map.get(family, family.upper() if family else '')
    """Set consistent style for comparisons (with/without reasoning)."""
    plt.style.use('default')
    sns.set_context("talk")
    
    return {
        "without": "#20B2AA",  # Light sea green
        "with": "#DA70D6"      # Orchid
    }

def set_change_style():
    """Set consistent style for changes (increases/decreases)."""
    plt.style.use('default')
    sns.set_context("talk")
    
    return {
        "increase": "#2ECC71",  # Green for increases
        "decrease": "#E74C3C",  # Red for decreases
        "neutral": "#95A5A6"    # Gray for no change
    }

def boxplot_model_performance_comparison(df, metric='top_prop', title=None):
    """
    Create a styled boxplot with stripplot for model comparison.
    
    Args:
        df (pd.DataFrame): DataFrame in repeated measures format
        metric (str): Either 'top_prop' or 'convergence'
        title (str): Plot title
    """
    # Set up the style
    reasoning_palette = set_comparison_style()
    
    # Convert data to long format for plotting
    plot_data = pd.melt(
        df,
        id_vars=['model', 'task_options'],
        value_vars=[f'{metric}_without_reasoning', f'{metric}_with_reasoning'],
        var_name='condition',
        value_name='value'
    )
    
    # Clean up condition names
    plot_data['condition'] = plot_data['condition'].map({
        f'{metric}_without_reasoning': 'without',
        f'{metric}_with_reasoning': 'with'
    })
    
    # Get model order
    model_order = get_model_order(df['model'].unique())
    model_labels = [m.split('-')[-1].upper() for m in model_order]
    
    # Create figure
    plt.figure(figsize=(12, 7), dpi=100, facecolor='white')
    
    # Create boxplot
    sns.boxplot(
        x='model',
        y='value',
        hue='condition',
        data=plot_data,
        order=model_order,
        palette=reasoning_palette,
        fliersize=0,
        linewidth=1.2,
        boxprops={
            'facecolor': 'none',
            'edgecolor': 'gray',
            'alpha': 0.8
        },
        medianprops={'color': 'black', 'linewidth': 1.5},
        whiskerprops={'color': 'gray', 'linewidth': 1.2},
        capprops={'color': 'gray', 'linewidth': 1.2}
    )
    
    # Add individual points
    sns.stripplot(
        x='model',
        y='value',
        hue='condition',
        data=plot_data,
        order=model_order,
        palette=reasoning_palette,
        dodge=True,
        alpha=0.4,
        size=5,
        edgecolor='none',
        jitter=0.2
    )
    
    # Add family indicator to title if applicable
    family = detect_model_family(df['model'].unique())
    family_name = get_formatted_family_name(family)
    if title is None:
        metric_name = 'Top Response Proportion' if metric == 'top_prop' else 'Convergence'
        title = f"Mean Performance Over All Task Variations Across Model Sizes ({metric_name} metric)\n{family_name}"
    plt.title(title, pad=20)
    
    plt.xlabel("Model Size", fontsize=12)
    plt.ylabel("Top Proportion" if metric == 'top_prop' else "Convergence", fontsize=12)
    
    # Set y-axis limits based on metric
    if metric == 'top_prop':
        plt.ylim(0.2, 1.0)  # 20% to 100%
    else:
        plt.ylim(0.2, 1.0)  # Changed to start at 0.2 for convergence as well
    
    # Update x-axis labels
    plt.xticks(range(len(model_order)), model_labels, rotation=0)
    
    # Clean up the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_handles = handles[:2]
    unique_labels = ['Without Reasoning', 'With Reasoning']
    plt.legend(
        unique_handles,
        unique_labels,
        title="Condition",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0
    )
    
    # Remove top and right spines and add grid
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_interaction_lines(df, metric='top_prop', title=None):
    """
    Create an interaction plot showing means for with/without reasoning across models.
    
    Args:
        df (pd.DataFrame): DataFrame in repeated measures format
        metric (str): Either 'top_prop' or 'convergence'
        title (str): Plot title
    """
    # Create figure with more space for title
    plt.figure(figsize=(10, 7))
    
    # Calculate means for each model and condition
    plot_data = pd.DataFrame({
        'model': df['model'].unique(),
        'without': df.groupby('model')[f'{metric}_without_reasoning'].mean(),
        'with': df.groupby('model')[f'{metric}_with_reasoning'].mean()
    }).reset_index(drop=True)
    
    # Extract model sizes for ordering
    unique_models = df['model'].unique()
    sizes = [int(model.split('-')[-1][:-1]) for model in unique_models]
    model_order = [x for _, x in sorted(zip(sizes, unique_models))]
    
    # Plot lines with error bars
    for i, model in enumerate(model_order):
        model_data = df[df['model'] == model]
        without_mean = model_data[f'{metric}_without_reasoning'].mean()
        with_mean = model_data[f'{metric}_with_reasoning'].mean()
        without_sem = model_data[f'{metric}_without_reasoning'].sem()
        with_sem = model_data[f'{metric}_with_reasoning'].sem()
        
        size = int(model.split('-')[-1][:-1])
        plt.errorbar([0, 1], [without_mean, with_mean],
                    yerr=[[without_sem, with_sem], [without_sem, with_sem]],
                    marker='o', markersize=8, linewidth=1.5,
                    label=f"{size}B")
    
    # Customize appearance
    plt.xticks([0, 1], ['None', 'CoT'])
    plt.xlabel('Reasoning')
    plt.ylabel(f"{'Top Response Proportion' if metric == 'top_prop' else 'Convergence Score'}")
    plt.ylim(0.2, 1.0)
    
    plt.legend(title="Model Size", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add title with more space
    family = detect_model_family(df['model'].unique())
    family_name = get_formatted_family_name(family)
    metric_name = 'Top Response Proportion' if metric == 'top_prop' else 'Convergence Score'
    if title is None:
        title = f"Mean Performance Over All Task Variations Across Reasoning Elicitation\n{family_name}"
    plt.title(title, pad=20)
    
    # Add more space above title
    plt.subplots_adjust(top=0.85)
    
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_paired_changes(df, metric='top_prop', title=None):
    """
    Create paired plots showing individual changes from without to with reasoning.
    """
    colors = set_change_style()
    
    # Extract and sort models by size
    models = df['model'].unique()
    sizes = [int(model.split('-')[-1][:-1]) for model in models]
    model_order = [x for _, x in sorted(zip(sizes, models))]
    n_models = len(model_order)
    
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5), sharey=True)

    
    # Create paired plots for each model
    legend_handles = []  # Store handles for legend
    for i, model in enumerate(model_order):
        model_data = df[df['model'] == model]
        
        # Get paired data
        without_data = model_data[f'{metric}_without_reasoning']
        with_data = model_data[f'{metric}_with_reasoning']
        
        # Plot individual lines with color coding
        decrease_plotted = increase_plotted = False
        for j in range(len(without_data)):
            change = with_data.iloc[j] - without_data.iloc[j]
            if change < 0:
                color = colors['decrease']  # Red for decreases
                label = 'Decrease' if not decrease_plotted else None
                decrease_plotted = True
            elif change > 0:
                color = colors['increase']  # Green for increases
                label = 'Increase' if not increase_plotted else None
                increase_plotted = True
            else:
                color = colors['neutral']
                label = None
                
            line = axes[i].plot([0, 1], [without_data.iloc[j], with_data.iloc[j]], 
                              color=color, alpha=1.0, linewidth=1, label=label)
            
            if label:
                legend_handles.append(line[0])
        
        # Customize appearance
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['None', 'CoT'])
        size = model.split('-')[-1]
        axes[i].set_title(f"{size} Parameters")
        
        if i == 0:
            metric_label = 'Individual Task Score' if metric == 'top_prop' else 'Individual Convergence Score'
            axes[i].set_ylabel(metric_label)
        
        axes[i].set_ylim(0.2, 1.0)
        if i == n_models - 1:
            axes[i].set_xlabel('Chain-of-Thought Prompting')
        
        sns.despine(ax=axes[i])
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add overall title with more space
    metric_name = 'Top Response Proportion' if metric == 'top_prop' else 'Convergence'
    title = f"Individual Task Performance Changes in {metric_name}\nby Model Size"
    fig.suptitle(title, y=1.08)
    
    # Add legend below the subplots
    fig.legend(handles=legend_handles, 
              labels=['Performance Decrease', 'Performance Increase'],
              bbox_to_anchor=(0.5, -0.1),
              loc='upper center',
              ncol=2,
              title="Changes with Chain-of-Thought")
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.85)

def plot_scores_by_model_size(df, metric='top_prop', title=None):
    """
    Create a line plot showing scores across model sizes for with/without reasoning.
    """
    print(df)
    colors = set_comparison_style()
    
    plt.figure(figsize=(10, 6))
    
    models = df['model'].unique()
    sizes = [int(model.split('-')[-1][:-1]) for model in models]
    model_order = [x for _, x in sorted(zip(sizes, models))]
    
    for condition, color_key in [('without', 'without'), ('with', 'with')]:
        means = []
        sems = []
        for model in model_order:
            model_data = df[df['model'] == model][f'{metric}_{condition}_reasoning']
            means.append(model_data.mean())
            sems.append(model_data.sem())
        
        x_pos = range(len(model_order))
        plt.errorbar(x_pos, means, yerr=sems, 
                    color=colors[color_key], 
                    marker='o', markersize=8, linewidth=2,
                    label=f'{"With" if condition == "with" else "Without"} Chain-of-Thought')
    
    plt.xticks(range(len(model_order)), [m.split('-')[-1].upper() for m in model_order])
    plt.xlabel('Model Size (Billion Parameters)', fontsize=11)
    
    metric_label = 'Mean Top Response Proportion' if metric == 'top_prop' else 'Mean Convergence Score'
    plt.ylabel(f'{metric_label} (± SEM)', fontsize=11)
    plt.ylim(0.2, 1.0)
    
    metric_name = 'Top Response Proportion' if metric == 'top_prop' else 'Convergence'
    title = f"Mean Response Coordination of all Task Variants ({metric_name}).\n" if title is None else title
    plt.title(title, pad=20)
    
    plt.legend(title="Prompting Condition")
    
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_model_comparison(df):
    """
    Create comparison plots for a model family.
    
    Args:
        df (pd.DataFrame): DataFrame containing model family data
    """
    # Plot top_prop
    boxplot_model_performance_comparison(
        df,
        metric='top_prop'
        # Title will be auto-generated with family name if detected
    )
    
    # Plot convergence
    plt.figure()
    boxplot_model_performance_comparison(
        df,
        metric='convergence'
    )
    
    plt.show()

def create_comprehensive_analysis(df):
    """
    Create a comprehensive set of visualizations for analyzing the reasoning effect.
    
    Args:
        df (pd.DataFrame): DataFrame containing model performance data
    """
    # Create scores by model size plots
    plot_scores_by_model_size(df, metric='top_prop')
    plt.figure()
    plot_scores_by_model_size(df, metric='convergence')
    
    # Create interaction plots
    plot_interaction_lines(df, metric='top_prop')
    plt.figure()
    plot_interaction_lines(df, metric='convergence')
    
    # Create paired change plots
    plot_paired_changes(df, metric='top_prop')
    plt.figure()
    plot_paired_changes(df, metric='convergence')
    
    plt.show()

def plot_control_reasoning_bars(df, metric='top_prop', title=None):
    """
    Creates a bar graph showing mean performance for each model on three conditions:
      - control
      - without reasoning
      - with reasoning
    for the specified metric ('top_prop' or 'convergence').
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Gather columns from repeated-measures data
    # e.g., top_prop_control, top_prop_without_reasoning, top_prop_with_reasoning
    control_col = f"{metric}_control"
    no_reason_col = f"{metric}_without_reasoning"
    with_reason_col = f"{metric}_with_reasoning"

    # Melt into long format if the control column exists
    melted = pd.melt(
        df,
        id_vars=['model', 'task_options'],
        value_vars=[c for c in [control_col, no_reason_col, with_reason_col] if c in df.columns],
        var_name='condition',
        value_name='score'
    )

    # Convert condition names to a friendlier label
    condition_labels = {
        control_col: 'Control',
        no_reason_col: 'Without Reasoning',
        with_reason_col: 'With Reasoning'
    }
    melted['condition'] = melted['condition'].map(condition_labels)

    # Calculate means (or we can just plot raw if desired)
    grouped = melted.groupby(['model', 'condition'])['score'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=grouped,
        x='model',
        y='score',
        hue='condition'
    )
    plt.ylim(0, 1)  # Adjust as needed
    if not title:
        title = f"Mean {metric.capitalize()} by Model and Condition"
    plt.title(title, pad=15)
    plt.legend(title="Condition")
    sns.despine()
    plt.tight_layout()
    plt.show()

def simplify_column_names(df):
    """
    Simplifies column names from the descriptive format to a more concise format for visualization.
    Does not rename the convergence_answered column.
    
    Args:
        df (pandas.DataFrame): The DataFrame with descriptive column names
    
    Returns:
        pandas.DataFrame: A copy of the DataFrame with simplified column names
    """
    return df.rename(columns={
        'model_name': 'model',
        'task_instruction_component_key': 'task_instruction',
        'reasoning_instruction_component_key': 'task_reasoning', 
        'options_lists_key': 'task_options',
        'top_option_name': 'top',
        'second_option_name': 'second',
        'third_option_name': 'third',
        'fourth_option_name': 'fourth',
        'top_prop_all': 'top_prop',
        'second_prop_all': 'second_prop',
        'third_prop_all': 'third_prop',
        'fourth_prop_all': 'fourth_prop',
        'convergence_all': 'convergence'
    })
