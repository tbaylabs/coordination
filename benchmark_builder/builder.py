# Create a function to...
# - take the df from aggregate_trial_results
# - For a given model (model_name given as function parameter)
# - Strip out all other results except that model.
# - Remove the row for the "letters" task option
# - 
# - If the model is a reasoning model, there should be 40 rows. 20 unique for each task. 
# Task Options Column Transformation Instructions

# ## Transformation Rules

# 1. The original column 'task_options' contains values like "numbers", "numbers-text", "shapes-1-icon", etc.
# 2. This needs to be split into two new columns:
#    - 'task_options_name': The base name of the task option
#    - 'task_options_type': Either "symbol" or "text"

# ## Mapping Rules

# For any value in the 'task_options' column, apply these transformations:

# ### Pattern 1: Base + "-text" or "-english"
# If the value ends with "-text" or "-english":
# - task_options_name: Remove the "-text" or "-english" suffix
# - task_options_type: "text"

# ### Pattern 2: Base + "-icon" or no suffix
# If the value ends with "-icon" or has no special suffix:
# - task_options_name: Remove the "-icon" suffix if present
# - task_options_type: "symbol"

# ## Complete Mapping Reference

# Original Value (task_options) -> New Values (task_options_name, task_options_type)

# ```
# "numbers" -> "numbers", "symbol"
# "numbers-text" -> "numbers", "text"
# "shapes-1-icon" -> "shapes-1", "symbol"
# "shapes-1-text" -> "shapes-1", "text"
# "shapes-2-icon" -> "shapes-2", "symbol"
# "shapes-2-text" -> "shapes-2", "text"
# "shapes-3-icon" -> "shapes-3", "symbol"
# "shapes-3-text" -> "shapes-3", "text"
# "emoji-1" -> "emoji-1", "symbol"
# "emoji-1-text" -> "emoji-1", "text"
# "emoji-2" -> "emoji-2", "symbol"
# "emoji-2-text" -> "emoji-2", "text"
# "emoji-3" -> "emoji-3", "symbol"
# "emoji-3-text" -> "emoji-3", "text"
# "kanji-nature" -> "kanji-nature", "symbol"
# "kanji-nature-english" -> "kanji-nature", "text"
# "kanji-random" -> "kanji-random", "symbol"
# "kanji-random-english" -> "kanji-random", "text"
# "colours" -> "colours", "symbol"
# "colours-text" -> "colours", "text"
# ```

# ## Implementation Notes

# 1. The transformation should be case-sensitive
# 2. No other columns should be modified
# 3. The order of rows should be preserved
# 4. The original 'task_options' column should be removed after the transformation
# 5. Skip any rows containing "letters" in the task_options column
# 6. Ensure proper handling of hyphenated names (e.g., "kanji-nature" should stay hyphenated)

# ## Validation Rules

# After transformation, verify:
# 1. Every row has both a task_options_name and task_options_type
# 2. task_options_type only contains "symbol" or "text"
# 3. Pairs of rows with the same task_options_name have different task_options_type values
# 4. No "letters" related entries are present in the transformed data

# Then print the df with print_nice_dataframe
# Then return the df.


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


# import tabulate correctly