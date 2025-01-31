from pathlib import Path
import pandas as pd
from tabulate import tabulate

def get_filtered_data():
    """
    Get filtered experiment data without reasoning models.
    
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # Read data using consistent path handling
    df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')
    
    # Filter out reasoning models
    reasoning_models = ['o1', 'o1-mini', 'deepseek-r1']
    return df[~df['model_name'].isin(reasoning_models)]

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

def filter_by_task_type(df, task_type='all'):
    """
    Filter dataframe by task type.
    
    Args:
        df (pd.DataFrame): The raw experiment data
        task_type (str): Type of tasks to include. Options are:
            - 'all': Include all tasks (default)
            - 'text_only': Only tasks with 'text' or 'english' in name
            - 'symbol_only': Only symbol tasks (emoji, shapes, colors, kanji) without text/english
            - 'numbers': Only number-related tasks
            - 'letters': Only letter-related tasks
            
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if task_type == 'all':
        return df
    elif task_type == 'text_only':
        return df[df['task_options'].str.contains('text|english')]
    elif task_type == 'symbol_only':
        # Get all symbol tasks excluding text/english versions
        symbol_tasks = [
            'shapes-1-icon', 'shapes-2-icon', 'shapes-3-icon',
            'emoji-1', 'emoji-2', 'emoji-3',
            'kanji-nature', 'kanji-random',
            'colours'
        ]
        return df[df['task_options'].isin(symbol_tasks)]
    elif task_type == 'numbers':
        return df[df['task_options'].str.contains('numbers')]
    elif task_type == 'letters':
        return df[df['task_options'] == 'letters']
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Valid options are: 'all', 'text_only', 'symbol_only', 'numbers', 'letters'")

from pathlib import Path
import pandas as pd

def prepare_graph_data(df=None, task_type='all'):
    """
    Prepare and return the processed data for chart creation with SEM.
    
    Args:
        df (pd.DataFrame, optional): Preprocessed dataframe with experiment conditions.
                                     If None, reads from trial_results_aggregated.csv.
        task_type (str): Type of tasks to include. See filter_by_task_type() for options.
                                     
    Returns:
        dict: Dictionary of DataFrames with mean metrics and SEM for visualization
    """
    # Read data if not provided
    if df is None:
        df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')
        df = add_experiment_conditions(df)
    
    # Apply task type filter
    df = filter_by_task_type(df, task_type)

    # Prepare data for different metrics with mean and SEM
    data = {}
    metrics = ['top_prop_answered', 'top_prop_all', 'convergence_all']
    
    for metric in metrics:
        # Calculate mean across tasks
        mean_df = df.groupby(['model_name', 'experiment'], observed=True)[metric].mean().unstack()
        data[f'{metric}_mean'] = mean_df
        
        # Calculate standard error of the mean (SEM) across tasks
        sem_df = df.groupby(['model_name', 'experiment'], observed=True)[metric].sem().unstack()
        data[f'{metric}_sem'] = sem_df
    
    return data

def print_nice_dataframe(df, max_rows=120, show_index=False):
    """Generic function for nicely printing any DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to display
        max_rows (int): Maximum number of rows to display
        show_index (bool): Whether to show the index in the output
    """
    if len(df) > max_rows:
        print(f"Displaying first {max_rows} rows (total: {len(df)}):\n")
        print(tabulate(df.head(max_rows), headers='keys', 
                     tablefmt='grid', showindex=show_index))
    else:
        print(tabulate(df, headers='keys', tablefmt='grid', 
                     showindex=show_index))