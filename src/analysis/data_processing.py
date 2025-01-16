import pandas as pd

"""
Functions for processing, cleaning, and formatting data for analysis.
"""

def validate_experiment_data(df, unanswered_threshold=0.2, verbose=False):
    """
    Validates experimental data for analysis readiness.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        unanswered_threshold (float): Threshold for unanswered proportion warning
        verbose (bool): If True, prints detailed validation report
    
    Returns:
        dict: Summary of validation results containing:
            - status: 'pass' or 'warning' or 'error'
            - issues: list of found issues
            - metrics: validation metrics
    """
    issues = []
    results = {
        'status': 'pass',
        'issues': [],
        'metrics': {
            'rows_with_high_unanswered': 0,
            'rows_with_invalid_metrics': 0,
            'invalid_total_counts': 0
        }
    }
    
    # Check unanswered proportions
    high_unanswered = df[df['unanswered_prop'] > unanswered_threshold]
    if not high_unanswered.empty:
        results['status'] = 'warning'
        results['metrics']['rows_with_high_unanswered'] = len(high_unanswered)
        issues.append(f"Found {len(high_unanswered)} rows with unanswered_prop > {unanswered_threshold}")
        if verbose:
            print("\n=== Rows with unanswered_prop above threshold ===")
            print(high_unanswered[['file_name', 'model_name', 'task_instruction', 'task_options', 'unanswered_prop']])
    
    # Check metrics for invalid values
    metric_cols = ['top_prop_all', 'convergence_answered', 'convergence_all']
    for col in metric_cols:
        invalid_metrics = df[
            (df[col].isna()) | 
            (df[col] <= 0) | 
            (df[col] > 1)  # assuming these are proportions
        ]
        if not invalid_metrics.empty:
            results['status'] = 'error'
            results['metrics']['rows_with_invalid_metrics'] += len(invalid_metrics)
            issues.append(f"Found {len(invalid_metrics)} rows with invalid {col}")
            if verbose:
                print(f"\n=== Invalid metric rows for {col} ===")
                print(invalid_metrics[['file_name', 'model_name', 'task_instruction', 'task_options', col]])
    
    # Check total count
    invalid_totals = df[df['total_count'] != 120]
    if not invalid_totals.empty:
        results['status'] = 'error'
        results['metrics']['invalid_total_counts'] = len(invalid_totals)
        issues.append(f"Found {len(invalid_totals)} rows with total_count != 120")
        if verbose:
            print("\n=== Rows with invalid total_count !== 120 ===")
            print(invalid_totals[['file_name', 'model_name', 'task_instruction', 'task_options', 'total_count']])
    
    results['issues'] = issues
    
    # Only print report if verbose
    if verbose:
        print("\n=== Data Validation Report ===")
        if results['status'] == 'pass':
            print("✓ All validations passed. Checked: total_count, metrics validity, unanswered proportions")
        else:
            print(f"Status: {results['status'].upper()}")
            for issue in issues:
                print(f"- {issue}")
        print("===========================\n")
    elif results['status'] != 'pass':
        # Print single line summary if not verbose but there are issues
        print(f"Data validation {results['status']}: {len(issues)} issue(s) found")
    
    if results['status'] == 'error':
        raise ValueError("Data validation failed. Use verbose=True for details.")
        
    return results

def prune_high_unanswered(df, threshold=0.2, verbose=False):
    """
    Removes rows where unanswered_prop exceeds threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (float): Threshold for removal
    
    Returns:
        tuple: (pruned_df, dict of removed_info)
    """
    if threshold != 0.2:
        warnings.warn(f"Using non-standard threshold {threshold} for unanswered proportion pruning")
    
    high_unanswered = df[df['unanswered_prop'] > threshold]
    pruned_df = df[df['unanswered_prop'] <= threshold]
    
    info = {
        'rows_removed': len(high_unanswered),
        'removed_options': high_unanswered['task_options'].unique().tolist(),
        'details': high_unanswered[['file_name', 'model_name', 'task_instruction', 
                                  'task_options', 'unanswered_prop']].to_dict('records')
    }
    
    return pruned_df, info

def repeated_measures_rebalance(df, verbose=False):
    """
    Ensures balanced repeated measures by removing all instances
    of task_options that are missing for any condition.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        tuple: (balanced_df, dict of rebalancing_info)
    """
    # Create a cross product of all unique values
    models = pd.DataFrame({'model_name': df['model_name'].unique()})
    reasoning = pd.DataFrame({'task_reasoning': ['none', 'step-by-step']})
    options = pd.DataFrame({'task_options': df['task_options'].unique()})
    
    # Create all possible combinations using cross joins
    all_conditions = (
        models.assign(key=1)
        .merge(reasoning.assign(key=1), on='key')
        .merge(options.assign(key=1), on='key')
        .drop('key', axis=1)
    )
    
    # Find which task_options are missing for any condition
    actual_conditions = df.groupby(
        ['model_name', 'task_reasoning', 'task_options']
    ).size().reset_index()
    
    missing_conditions = pd.merge(
        all_conditions, actual_conditions,
        how='left',
        on=['model_name', 'task_reasoning', 'task_options']
    )
    
    incomplete_options = missing_conditions[
        missing_conditions[0].isna()
    ]['task_options'].unique()
    
    # Remove all rows with these task_options
    balanced_df = df[~df['task_options'].isin(incomplete_options)]
    
    info = {
        'rows_removed': len(df) - len(balanced_df),
        'removed_options': incomplete_options.tolist(),
        'remaining_options': balanced_df['task_options'].unique().tolist()
    }
    
    return balanced_df, info

