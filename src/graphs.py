from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Read CSV into DataFrame

# file_name,model_name,temperature,xml_prompt,task_instruction,task_reasoning,task_options,top_option_name,top_option_count,second_option_name,second_option_count,third_option_name,third_option_count,fourth_option_name,fourth_option_count,unanswered_count,answered_count,total_count,unanswered_prop,top_prop_all,second_prop_all,third_prop_all,fourth_prop_all,convergence_answered,convergence_all,extracted_by_rule_count,extracted_by_llm_count,extracted_by_human_count,extracted_by_rule_prop,extracted_by_llm_prop,extracted_by_human_prop
# res_coordinate_shapes-3-text_none_llama-33-70b.json,llama-33-70b,default,False,coordinate,none,shapes-3-text,filled diamond,57,filled triangle,35,hollow diamond,16,hollow triangle,12,0,120,120,0.0,0.475,0.2916666666666667,0.13333333333333333,0.1,0.33847222222222223,0.33847222222222223,17,103,0,0.14166666666666666,0.8583333333333333,0.0
# res_coordinate_numbers_none_llama-32-3b.json,llama-32-3b,default,False,coordinate,none,numbers,2,51,3,31,4,26,1,9,3,117,120,0.025,0.425,0.25833333333333336,0.21666666666666667,0.075,0.31550880268828985,0.3005555555555555,0,0,0,0.0,0.0,0.0
# res_coordinate_emoji-3-text_none_llama-31-8b.json,llama-31-8b,default,False,coordinate,none,emoji-3-text,bucket,53,mailbox,22,tennis ball,21,clock,20,4,116,120,0.03333333333333333,0.44166666666666665,0.18333333333333332,0.175,0.16666666666666666,0.30722354340071345,0.2881944444444445,86,34,0,0.7166666666666667,0.2833333333333333,0.0

df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')


# chart 1: (just a test chart) A line chart with the x axis as `task_instruction`, and the Y `convergence_all``

# Group by task_instruction and calculate mean convergence_all
grouped = df.groupby('task_instruction')['convergence_all'].mean().reset_index()

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(grouped['task_instruction'], grouped['convergence_all'], 
         marker='o', linestyle='-', color='b')

# Add labels and title
plt.xlabel('Task Instruction')
plt.ylabel('Average Convergence (All)')
plt.title('Average Convergence by Task Instruction')
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout and save plot
plt.tight_layout()
output_path = Path(__file__).parent / 'convergence_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to {output_path}")

# Chart 2: Model comparison across experiment conditions
plt.figure(figsize=(12, 6))

# Create experiment condition mapping
conditions = {
    ('control', 'none'): 'control',
    ('control', 'step-by-step'): 'control-COT',
    ('coordinate', 'none'): 'coordinate-none',
    ('coordinate', 'step-by-step'): 'coordinate-COT'
}

# Add experiment condition to dataframe
df['experiment'] = df.apply(lambda row: conditions.get((row['task_instruction'], row['task_reasoning']), 'other'), axis=1)

# Filter out control-COT if it has no data
if 'control-COT' not in df['experiment'].unique():
    conditions = {k:v for k,v in conditions.items() if v != 'control-COT'}

# Group by model and experiment, calculate mean convergence
model_comparison = df.groupby(['model_name', 'experiment'])['convergence_all'].mean().unstack()

# Plot each model's performance
for model in model_comparison.index:
    plt.plot(model_comparison.columns, model_comparison.loc[model], 
             marker='o', linestyle='-', label=model)

# Add labels and title
plt.xlabel('Experiment Condition')
plt.ylabel('Average Convergence (All)')
plt.title('Model Performance Comparison Across Experiment Conditions')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and save plot
plt.tight_layout()
output_path = Path(__file__).parent / 'model_comparison_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Model comparison chart saved to {output_path}")
