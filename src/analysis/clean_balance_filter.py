from src.analysis.model_filters import filter_by_model_family
from src.analysis.data_processing import (
    prune_high_unanswered,
    validate_experiment_data,
    repeated_measures_rebalance
)
from src.analysis.df_formatting import make_long_df, print_nice_dataframe

"""
Functions for running repeated measures ANOVA on experimental results.
"""
def clean_balance_filter(df, model_family, verbose=False):
    """
    Cleans, balances and filters data for analysis.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        model_family (str): Model family to filter for
        verbose (bool): If True, prints detailed preparation info
    
    Returns:
        pandas.DataFrame: Cleaned and balanced DataFrame
    """
    info = {
        'initial_rows': len(df),
        'validation_results': None,
        'pruning_results': None,
        'balancing_results': None
    }
    
    # Initial filtering
    df = filter_by_model_family(df, model_family)
    df = df[df['task_reasoning'].isin(['none', 'step-by-step', 'control'])]
    
    info['after_filtering_rows'] = len(df)
    
    if verbose:
        print("\nData after filtering by model family and reasoning conditions:")
        print_nice_dataframe(df)
    
    # Validate
    info['validation_results'] = validate_experiment_data(df, verbose=verbose)
    
    # Prune high unanswered
    df, pruning_info = prune_high_unanswered(df, verbose=verbose)
    info['pruning_results'] = pruning_info
    
    # Rebalance
    df, balancing_info = repeated_measures_rebalance(df, verbose=verbose)
    info['balancing_results'] = balancing_info
    
    info['final_rows'] = len(df)
    
    # Only print preparation summary if verbose
    if verbose:
        print("\n=== Data Preparation Summary ===")
        print(f"Initial rows: {info['initial_rows']}")
        print(f"After filtering: {info['after_filtering_rows']}")
        print(f"Final balanced rows: {info['final_rows']}")
        
        if info['pruning_results']['rows_removed'] > 0:
            print("\nRemoved options due to high unanswered rate:")
            for opt in info['pruning_results']['removed_options']:
                print(f"- {opt}")

        if info['balancing_results']['rows_removed'] > 0:
            print("\nRemoved options to ensure balanced design:")
            for opt in info['balancing_results']['removed_options']:
                print(f"- {opt}")
        print("=============================\n")
    elif info['pruning_results']['rows_removed'] > 0 or info['balancing_results']['rows_removed'] > 0:
        # Print single line summary if not verbose but there were removals
        print(f"Data preparation: removed {info['pruning_results']['rows_removed']} rows (unanswered) and {info['balancing_results']['rows_removed']} rows (balance)")
    
    return df

try:
    import pingouin as pg
except ImportError:
    raise ImportError(
        "The pingouin package is required for running repeated measures ANOVA. "
        "Please install it using: pip install pingouin"
    )

import pandas as pd
import numpy as np
from tabulate import tabulate

def format_value(value, decimal_places=3):
    """Safely formats a value with specified decimal places."""
    try:
        if pd.isna(value):
            return 'NA'
        if isinstance(value, (float, int)):
            return f"{value:.{decimal_places}f}"
        return str(value)
    except:
        return 'NA'

import warnings
from contextlib import contextmanager

@contextmanager
def suppress_warnings():
    """Context manager to suppress specific warnings"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', message='Data preparation:')
        yield

def run_repeated_measures_anova(df, verbose=False):
    """
    Runs repeated measures ANOVA on the prepared data testing for interaction
    between reasoning condition and model size.
    
    Args:
        df (pd.DataFrame): DataFrame in standard format (not repeated measures)
        verbose (bool): If True, prints detailed statistics and ANOVA results
            
    Returns:
        dict: Dictionary containing ANOVA results and summary statistics
    """
    # First prepare the data in repeated measures format
    # df = df.drop(columns=['top_prop_control', 'convergence_control'])
    """
    Runs repeated measures ANOVA on the prepared data testing for interaction
    between reasoning condition and model size.
    
    Args:
        df (pd.DataFrame): DataFrame in repeated measures format
        verbose (bool): If True, prints detailed statistics and ANOVA results
            
    Returns:
        dict: Dictionary containing ANOVA results and summary statistics
    """
    results = {}
    
    def analyze_metric(metric_name):
        # Extract model size from model name
        df['model_size'] = df['model'].str.extract(r'(\d+)b').astype(str) + 'B'
        
        # Prepare data in long format for pingouin
        long_data = pd.melt(
            df,
            id_vars=['model_size', 'task_options'],
            value_vars=[f'{metric_name}_without_reasoning', 
                       f'{metric_name}_with_reasoning'],
            var_name='condition',
            value_name='value'
        )
        
        # Clean up condition names for analysis
        long_data['condition'] = long_data['condition'].map({
            f'{metric_name}_without_reasoning': 'without',
            f'{metric_name}_with_reasoning': 'with'
        })
        
        # Calculate descriptive statistics by model size and condition
        desc_stats = long_data.groupby(['model_size', 'condition'])['value'].agg([
            'count', 'mean', 'std'
        ]).round(3)
        
        if verbose:
            print(f"\nDescriptive statistics for {metric_name}:")
            print(tabulate(desc_stats, headers='keys', tablefmt='grid'))
            print(f"\nRunning RM-ANOVA for {metric_name}...")
        
        with suppress_warnings():
            # Main effect of reasoning condition
            aov_condition = pg.rm_anova(
                data=long_data,
                dv='value',
                within='condition',
                subject='task_options',
                detailed=True
            )
            
            # Main effect of model size
            aov_model = pg.rm_anova(
                data=long_data,
                dv='value',
                within='model_size',
                subject='task_options',
                detailed=True
            )
            
            # Interaction effect
            aov_interaction = pg.rm_anova(
                data=long_data,
                dv='value',
                within=['condition', 'model_size'],
                subject='task_options',
                detailed=True
            )
        
        if verbose:
            print(f"\nANOVA results for {metric_name}:")
            print("\nMain effect of reasoning condition:")
            print(tabulate(aov_condition, headers='keys', tablefmt='grid'))
            print("\nMain effect of model size:")
            print(tabulate(aov_model, headers='keys', tablefmt='grid'))
            print("\nInteraction effect:")
            print(tabulate(aov_interaction, headers='keys', tablefmt='grid'))
        
        return {
            'descriptive_stats': desc_stats,
            'aov_condition': aov_condition,
            'aov_model': aov_model,
            'aov_interaction': aov_interaction
        }
    
    if verbose:
        print("\n=== Running Repeated Measures ANOVA ===")
    
    # Run analysis for both metrics
    for metric in ['top_prop', 'convergence']:
        results[metric] = analyze_metric(metric)
    
    if verbose:
        print("=======================================")
    
    return results
def print_rm_anova_summary(results):
    """
    Prints a detailed statistical summary of the RM-ANOVA results with interpretations.
    
    Args:
        results (dict): Results dictionary from run_repeated_measures_anova
    """
    print("\n=== Repeated Measures ANOVA Summary ===")
    
    def get_effect_interpretation(p_val, eta_squared):
        """Helper function to interpret statistical significance and effect size"""
        if p_val < 0.001:
            sig = "strong evidence was found"
        elif p_val < 0.01:
            sig = "evidence was found"
        elif p_val < 0.05:
            sig = "some evidence was found"
        else:
            sig = "no evidence was found"
            
        if eta_squared >= 0.14:
            effect = "large"
        elif eta_squared >= 0.06:
            effect = "medium"
        else:
            effect = "small"
            
        return sig, effect
    
    for metric in ['top_prop', 'convergence']:
        try:
            print(f"\n{metric.upper()}:")
            
            # Get all rows from interaction results
            aov = results[metric]['aov_interaction']
            
            # Extract correct rows for each effect
            cond_row = aov[aov['Source'] == 'condition'].iloc[0]
            model_row = aov[aov['Source'] == 'model_size'].iloc[0]
            int_row = aov[aov['Source'] == 'condition * model_size'].iloc[0]
            
            # Function to format F statistic
            def format_f_stat(row):
                return f"F({row['ddof1']},{row['ddof2']}) = {row['F']:.2f}"
            
            # Descriptive statistics
            print("\nDescriptive Statistics:")
            print(tabulate(results[metric]['descriptive_stats'], 
                         headers='keys', tablefmt='grid'))
            
            # Main effect of model size
            sig, effect = get_effect_interpretation(
                model_row['p-unc'], 
                model_row['ng2']
            )
            print("\nModel Size Effect:")
            print(f"{format_f_stat(model_row)}, p = {model_row['p-unc']:.2e}, η² = {model_row['ng2']:.3f}")
            print(f"→ {sig} for a {effect} effect of model size")
            
            # Main effect of reasoning
            sig, effect = get_effect_interpretation(
                cond_row['p-unc'],
                cond_row['ng2']
            )
            print("\nReasoning Effect:")
            print(f"{format_f_stat(cond_row)}, p = {cond_row['p-unc']:.3f}, η² = {cond_row['ng2']:.3f}")
            print(f"→ {sig} for an effect of reasoning condition")
            
            # Interaction effect
            sig, effect = get_effect_interpretation(
                int_row['p-unc'],
                int_row['ng2']
            )
            print("\nInteraction Effect:")
            print(f"{format_f_stat(int_row)}, p = {int_row['p-unc']:.3f}, η² = {int_row['ng2']:.3f}")
            print(f"→ {sig} for an interaction between model size and reasoning")
            
        except Exception as e:
            print(f"\n{metric.upper()}: Error formatting results - {str(e)}")
    
    print("\n=================================")
