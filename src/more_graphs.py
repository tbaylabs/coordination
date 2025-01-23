from pathlib import Path
import pandas as pd

def prepare_graph_data():
    # Read CSV into DataFrame
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
        'top_prop_answered': df.groupby(['model_name', 'experiment'])['top_prop_answered'].mean().unstack(),
        'top_prop_all': df.groupby(['model_name', 'experiment'])['top_prop_all'].mean().unstack(),
        'convergence_all': df.groupby(['model_name', 'experiment'])['convergence_all'].mean().unstack()
    }

    return data

### For reference, here are the column names: file_name,model_name,temperature,xml_prompt,task_instruction,task_reasoning,task_options,top_option_name,top_option_count,second_option_name,second_option_count,third_option_name,third_option_count,fourth_option_name,fourth_option_count,unanswered_count,answered_count,total_count,unanswered_prop,top_prop_all,second_prop_all,third_prop_all,fourth_prop_all,convergence_answered,convergence_all,extracted_by_rule_count,extracted_by_llm_count,extracted_by_human_count,extracted_by_rule_prop,extracted_by_llm_prop,extracted_by_human_prop
### create a function to plot chart 1 and 2.
### It should contain only llama-31-405b, llama-31-70b, llama-31-8b, claude-35-sonnet, llama-33-70b, o1-mini, deepseek-r1 (those are model_name values)
### For o1-mini and deepseek-r1 it should only include the data for the control and coordinate conditions (not coordinate-COT)
### I want the first chart to be top_prop_all and the second to use top_prop_answered. For each condition, get the average of the model across all the task_options.
### Have a line graph where from left to right we have control, coordinate and coordinate-COT and a line showing the average performance on the metric for each model across the task options for that condition


if __name__ == '__main__':
    data = prepare_graph_data()
    # Data is now ready for any additional analysis or charting
