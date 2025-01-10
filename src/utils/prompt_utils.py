"""
Functions for building and managing prompts.
"""

def build_prompt(
    task_instruction_component_key,
    options_lists_key,
    reasoning_instruction_component_key,
    permutation_pattern,
    return_xml=False
):
    """
    Build a prompt based on the selected task instruction, reasoning instruction, and options list.

    Args:
        task_instruction_key (str): Key to select a task instruction component.
        reasoning_instruction_key (str): Key to select a reasoning instruction component.
        options_lists_key (str): Key to select an options list.
        permutation_pattern (str): A string of digits representing the order of options.
        return_xml (bool): Whether to return the prompt as XML (True) or in bullet list format (False).

    Returns:
        str: A formatted prompt string in either XML or bullet list format.
    """
    # Load the components
    from .data_loading import load_prompt_components
    task_instruction_components, options_lists, reasoning_instruction_components = load_prompt_components()

    # Select the components based on the provided keys
    task_instruction = task_instruction_components.get(task_instruction_component_key, "")
    reasoning_instruction = reasoning_instruction_components.get(reasoning_instruction_component_key, "")
    options_list = options_lists.get(options_lists_key, [])

    # Validate and reorder the options based on the permutation pattern
    if len(options_list) != len(permutation_pattern):
        raise ValueError("The length of the options list must match the length of the permutation pattern.")

    try:
        reordered_options = [options_list[int(index)] for index in permutation_pattern]
    except (IndexError, ValueError):
        raise ValueError("The permutation pattern must be valid indices into the options list.")

    # Build the prompt based on the return_xml flag
    if return_xml:
        # Build XML format with new lines before and after <options>
        options_xml = '\n'.join([f"            <option>{option}</option>" for option in reordered_options])
        return f"""
<task>
    <instruction>
        {task_instruction}
        Choose from the following options:

        <options>
{options_xml}
        </options>

        {reasoning_instruction}
    </instruction>
</task>
        """
    else:
        # Build bullet list format
        options_list_str = '\n'.join([f"- {option}" for option in reordered_options])
        return f"""{task_instruction}
Choose from the following options:

{options_list_str}

{reasoning_instruction}
        """
