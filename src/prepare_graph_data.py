from pathlib import Path
import pandas as pd

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
        'top_prop_answered': df.groupby(['model_name', 'experiment'], observed=True)['top_prop_answered'].mean().unstack(),
        'top_prop_all': df.groupby(['model_name', 'experiment'], observed=True)['top_prop_all'].mean().unstack(),
        'convergence_all': df.groupby(['model_name', 'experiment'], observed=True)['convergence_all'].mean().unstack()
    }
    
    return data
