from pathlib import Path
import matplotlib
import pandas as pd

# Read CSV into DataFrame
df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')

# Print first few rows
print(df.head())

# file_name,model_name,temperature,xml_prompt,task_instruction,task_reasoning,task_options,top_option_name,top_option_count,second_option_name,second_option_count,third_option_name,third_option_count,fourth_option_name,fourth_option_count,unanswered_count,answered_count,total_count,unanswered_prop,top_prop_all,second_prop_all,third_prop_all,fourth_prop_all,convergence_answered,convergence_all,extracted_by_rule_count,extracted_by_llm_count,extracted_by_human_count,extracted_by_rule_prop,extracted_by_llm_prop,extracted_by_human_prop
# res_coordinate_shapes-3-text_none_llama-33-70b.json,llama-33-70b,default,False,coordinate,none,shapes-3-text,filled diamond,57,filled triangle,35,hollow diamond,16,hollow triangle,12,0,120,120,0.0,0.475,0.2916666666666667,0.13333333333333333,0.1,0.33847222222222223,0.33847222222222223,17,103,0,0.14166666666666666,0.8583333333333333,0.0
# res_coordinate_numbers_none_llama-32-3b.json,llama-32-3b,default,False,coordinate,none,numbers,2,51,3,31,4,26,1,9,3,117,120,0.025,0.425,0.25833333333333336,0.21666666666666667,0.075,0.31550880268828985,0.3005555555555555,0,0,0,0.0,0.0,0.0
# res_coordinate_emoji-3-text_none_llama-31-8b.json,llama-31-8b,default,False,coordinate,none,emoji-3-text,bucket,53,mailbox,22,tennis ball,21,clock,20,4,116,120,0.03333333333333333,0.44166666666666665,0.18333333333333332,0.175,0.16666666666666666,0.30722354340071345,0.2881944444444445,86,34,0,0.7166666666666667,0.2833333333333333,0.0

# we want to group by model name, and produce one chart for each of task_reasoning and task_instruction. task_options should be combined and averaged.

# chart 1: A line chart with the x axis as `task_instruction`, and the Y `convergence_all``

# ignore for now: 
# chart 2: eg. comparison of [claude-3-haiku, llama-33-70b, ..., gemini-flash] models, each should be a
# line on the chart, with the values as the averaged `convergence_all` over the various `task_option` values.
# 
#  