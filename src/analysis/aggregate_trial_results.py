"""
Functions for aggregating and analyzing results from multiple trials.
"""

import os
import json
import pandas as pd
from pathlib import Path
from .utils.metrics import convergence_metric

def aggregate_trial_results(results_folder_path=None):
    """
    Processes all JSON files in a results folder to aggregate trial data,
    including convergence metrics for answered and all options.

    Args:
        results_folder_path (str, optional): Path to the folder containing results JSON files.
                                           If None, uses pipeline/3_results in project root.

    Returns:
        pandas.DataFrame: A DataFrame containing the aggregated trial results with descriptive column names.
    """
    if results_folder_path is None:
        # Get the project root directory (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        results_folder = project_root / "pipeline" / "3_results"
    else:
        results_folder = Path(results_folder_path)

    # Ensure the folder exists or create it
    results_folder.mkdir(parents=True, exist_ok=True)

    if not results_folder.is_dir():
        raise NotADirectoryError(f"The path {results_folder_path} exists but is not a directory.")

    rows = []

    for file_path in results_folder.glob("*.json"):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)

            overview = data.get("overview", {})
            model_name = overview.get("model_name", "N/A")
            temperature = overview.get("temperature", "N/A")
            xml_prompt = overview.get("xml_prompt", False)
            task_instruction_component_key = overview.get("task_instruction_component_key", "N/A")
            reasoning_instruction_component_key = overview.get("reasoning_instruction_component_key", "N/A")
            options_lists_key = overview.get("options_lists_key", "N/A")

            results_summary = data.get("results-summary", {})
            options_list = overview.get("options_list", [])
            if not isinstance(options_list, list):
                raise ValueError(f"'options_list' must be a list in file {file_path.name}")

            ranked_options = [(option, results_summary.get(option, 0)) for option in options_list]
            ranked_options.sort(key=lambda x: (-x[1], options_list.index(x[0])))

            top_option_name, top_option_count = ("N/A", 0)
            second_option_name, second_option_count = ("N/A", 0)
            third_option_name, third_option_count = ("N/A", 0)
            fourth_option_name, fourth_option_count = ("N/A", 0)

            if len(ranked_options) > 0:
                top_option_name, top_option_count = ranked_options[0]
            if len(ranked_options) > 1:
                second_option_name, second_option_count = ranked_options[1]
            if len(ranked_options) > 2:
                third_option_name, third_option_count = ranked_options[2]
            if len(ranked_options) > 3:
                fourth_option_name, fourth_option_count = ranked_options[3]

            unanswered_count = results_summary.get("unanswered", 0)
            answered_count = top_option_count + second_option_count + third_option_count + fourth_option_count
            total_count = answered_count + unanswered_count

            first_prop_all = (top_option_count / total_count) if total_count > 0 else 0
            second_prop_all = (second_option_count / total_count) if total_count > 0 else 0
            third_prop_all = (third_option_count / total_count) if total_count > 0 else 0
            fourth_prop_all = (fourth_option_count / total_count) if total_count > 0 else 0
            unanswered_prop = (unanswered_count / total_count) if total_count > 0 else 0

            if answered_count > 0:
                p_values_answered = [
                    top_option_count / answered_count,
                    second_option_count / answered_count,
                    third_option_count / answered_count,
                    fourth_option_count / answered_count,
                ]
                convergence_answered = convergence_metric(p_values_answered)
            else:
                convergence_answered = None

            if total_count > 0:
                p_values_all = [
                    top_option_count / total_count,
                    second_option_count / total_count,
                    third_option_count / total_count,
                    fourth_option_count / total_count,
                ]
                convergence_all = convergence_metric(p_values_all)
            else:
                convergence_all = None

            # Get token statistics from results summary
            token_stats = results_summary.get("token_statistics", {})
            avg_token_count = token_stats.get("average_token_count", 0)
            median_token_count = token_stats.get("median_token_count", 0)
            min_token_count = token_stats.get("min_token_count", 0)
            max_token_count = token_stats.get("max_token_count", 0)
            total_token_count = token_stats.get("total_token_count", 0)

            # Get extraction counts from results summary
            extracted_by_rule_count = results_summary.get("extracted_by_rule_count", 0)
            extracted_by_llm_count = results_summary.get("extracted_by_llm_count", 0)
            extracted_by_human_count = results_summary.get("extracted_by_human_count", 0)
            
            # Calculate extraction proportions
            extracted_by_rule_prop = extracted_by_rule_count / total_count if total_count > 0 else 0
            extracted_by_llm_prop = extracted_by_llm_count / total_count if total_count > 0 else 0
            extracted_by_human_prop = extracted_by_human_count / total_count if total_count > 0 else 0

            rows.append({
                "file_name": file_path.name,
                "model_name": model_name,
                "temperature": temperature,
                "xml_prompt": xml_prompt,
                "task_instruction": task_instruction_component_key,
                "task_reasoning": reasoning_instruction_component_key,
                "task_options": options_lists_key,
                "top_option_name": top_option_name,
                "top_option_count": top_option_count,
                "second_option_name": second_option_name,
                "second_option_count": second_option_count,
                "third_option_name": third_option_name,
                "third_option_count": third_option_count,
                "fourth_option_name": fourth_option_name,
                "fourth_option_count": fourth_option_count,
                "unanswered_count": unanswered_count,
                "answered_count": answered_count,
                "total_count": total_count,
                "unanswered_prop": unanswered_prop,
                "top_prop_all": first_prop_all,
                "second_prop_all": second_prop_all,
                "third_prop_all": third_prop_all,
                "fourth_prop_all": fourth_prop_all,
                "convergence_answered": convergence_answered,
                "convergence_all": convergence_all,
                "extracted_by_rule_count": extracted_by_rule_count,
                "extracted_by_llm_count": extracted_by_llm_count,
                "extracted_by_human_count": extracted_by_human_count,
                "extracted_by_rule_prop": extracted_by_rule_prop,
                "extracted_by_llm_prop": extracted_by_llm_prop,
                "extracted_by_human_prop": extracted_by_human_prop,
                "avg_token_count": avg_token_count,
                "median_token_count": median_token_count,
                "min_token_count": min_token_count,
                "max_token_count": max_token_count,
                "total_token_count": total_token_count
            })

        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}")
            continue

    df = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file in the project root's pipeline/4_analysis folder
    if results_folder_path is None:
        output_dir = project_root / "pipeline" / "4_analysis"
    else:
        output_dir = Path(results_folder_path).parent / "4_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "trial_results_aggregated.csv"
    df.to_csv(output_file, index=False)

    print(f"Aggregated trial results saved to {output_file}")
    return df
