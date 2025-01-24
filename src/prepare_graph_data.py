from pathlib import Path
import pandas as pd

def add_experiment_conditions(df):
    """
    Add experiment conditions to the dataframe and filter out 'other' experiments.
    
    Args:
        df (pd.DataFrame): The raw experiment data
        
    Returns:
        pd.DataFrame: Dataframe with added 'experiment' column and filtered rows
    """
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
    
    return df

def prepare_graph_data(df=None):
    """
    Prepare and return the processed data for chart creation.
    
    Args:
        df (pd.DataFrame, optional): Preprocessed dataframe with experiment conditions.
                                     If None, reads from trial_results_aggregated.csv.
                                     
    Returns:
        dict: Dictionary of DataFrames with mean metrics for visualization
    """
    # Read data if not provided
    if df is None:
        df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')
        df = add_experiment_conditions(df)

    # Prepare data for different metrics
    data = {
        'top_prop_answered': df.groupby(['model_name', 'experiment'], observed=True)['top_prop_answered'].mean().unstack(),
        'top_prop_all': df.groupby(['model_name', 'experiment'], observed=True)['top_prop_all'].mean().unstack(),
        'convergence_all': df.groupby(['model_name', 'experiment'], observed=True)['convergence_all'].mean().unstack()
    }
    
    return data
