from pathlib import Path
import pandas as pd

def prepare_graph_data():
    # Read CSV into DataFrame
    df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')

    # Create experiment condition mapping
    conditions = {
        ('control', 'step-by-step'): '0-control-COT',
        ('control', 'none'): '1-control',
        ('coordinate', 'none'): '2-coordinate-none',
        ('coordinate', 'step-by-step'): '3-coordinate-COT'
    }

    # Add experiment condition to dataframe
    df['experiment'] = df.apply(lambda row: conditions.get((row['task_instruction'], row['task_reasoning']), 'other'), axis=1)

    # Remove all rows with 'other' experiment and with '0-control-COT' experiment
    df = df[~df['experiment'].isin(['other', '0-control-COT'])]

    # Calculate top_prop_answered
    df['top_prop_answered'] = df['top_option_count'] / df['answered_count']

    # Prepare data for different metrics
    data = {
        'top_prop_answered': df.groupby(['model_name', 'experiment'])['top_prop_answered'].mean().unstack(),
        'top_prop_all': df.groupby(['model_name', 'experiment'])['top_prop_all'].mean().unstack(),
        'convergence_all': df.groupby(['model_name', 'experiment'])['convergence_all'].mean().unstack()
    }

    return data


if __name__ == '__main__':
    data = prepare_graph_data()
    # Data is now ready for any additional analysis or charting
