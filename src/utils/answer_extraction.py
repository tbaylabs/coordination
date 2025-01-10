"""
Functions for extracting answers from LLM responses.
"""

from xml.etree import ElementTree
from datetime import datetime, timezone
from pathlib import Path
import json
from litellm import completion

from .data_loading import get_model_mapping

def is_answer_valid(options_list: list[str], answer: str) -> bool:
    """
    Check if an answer is valid against a list of options.
    
    Args:
        options_list (list[str]): List of valid options
        answer (str): Answer to validate
        
    Returns:
        bool: True if answer is valid, False otherwise
    """
    # Add "unanswered" as a valid option
    valid_options = options_list + ["unanswered"]
    return answer in valid_options

def extract_answer_by_rule(content_received: str, options_list: list[str]) -> str:
    """
    Extract an answer from the content using rule-based matching.
    
    Args:
        content_received (str): The content to extract an answer from
        options_list (list[str]): List of valid options
    
    Returns:
        str or None: Extracted answer or None if no valid answer found
    """
    # Temporarily extend options_list with "unanswered"
    extended_options_list = options_list + ["unanswered"]

    def option_match(text: str) -> str:
        # Strip whitespace
        text = text.strip()

        # Helper function for wrapped match
        def is_wrapped_match(t: str, opt: str) -> bool:
            # Check exact match
            if t == opt:
                return True
            # Check wrapped forms
            wrapped_forms = [
                f"`{opt}`",
                f"\"{opt}\"",
                f"'{opt}'",
                f"<{opt}>"
            ]
            return t in wrapped_forms

        # Helper function for bullet prefix match
        def is_bullet_match(t: str, opt: str) -> bool:
            bullet_prefixes = ["- ", "* ", "• ", "-", "*", "•"]
            for prefix in bullet_prefixes:
                if t.startswith(prefix) and t[len(prefix):].strip() == opt:
                    return True
            return False

        # Helper function for "I choose" pattern match
        def is_choice_match(t: str, opt: str) -> bool:
            choice_patterns = ["I choose:", "I pick:", "My choice is:", "Answer:"]
            for pattern in choice_patterns:
                if pattern.lower() in t.lower():
                    remaining = t[t.lower().index(pattern.lower()) + len(pattern):].strip()
                    if remaining.lower() == opt.lower():
                        return True
            return False

        # Check for matches against each option
        for opt in extended_options_list:
            opt_lower = opt.lower()
            t_lower = text.lower()
            if (is_wrapped_match(t_lower, opt_lower) or 
                is_bullet_match(t_lower, opt_lower) or 
                is_choice_match(t_lower, opt_lower)):
                return opt  # Return original case option

        return None

    # 1. Direct match or wrapped/bullet match
    direct_match = option_match(content_received)
    if direct_match:
        return "unanswered" if direct_match == "unanswered" else direct_match

    # 2. Attempt XML parsing
    try:
        root = ElementTree.fromstring(content_received.strip())
    except ElementTree.ParseError:
        return None

    def single_tag(tag_name: str):
        # Check if root itself is this tag
        if root.tag == tag_name:
            # Find any other tags of the same type
            # If found, that means it's not a single tag scenario
            other_tags = [el for el in root.findall('.//' + tag_name) if el is not root]
            if not other_tags:
                return [root]
            return None
        else:
            # If root is not this tag, look for descendants
            tags = root.findall('.//' + tag_name)
            return tags if len(tags) == 1 else None

    # Check tags for matches
    def match_inside_tags(tag_name: str) -> str:
        tags = single_tag(tag_name)
        if tags:
            for tag in tags:
                tag_text = (tag.text or "").strip()
                match = option_match(tag_text)
                if match:
                    return match
        return None

    # 3. Check for matches inside <option> tags
    option_match_result = match_inside_tags("option")
    if option_match_result:
        return "unanswered" if option_match_result == "unanswered" else option_match_result

    # 4. Check for matches inside <answer> tags
    answer_match_result = match_inside_tags("answer")
    if answer_match_result:
        return "unanswered" if answer_match_result == "unanswered" else answer_match_result

    # 5. If no matches found
    return None

def build_answer_extraction_prompt(response_text, options_list, permutation_pattern):
    """
    Build a prompt for extracting answers based on the provided options list and response text.

    Args:
        response_text (str): The text response to analyze.
        options_list (list): A list of possible choices for the question.
        permutation_pattern (str): String of four digits representing the order options were presented.

    Returns:
        str: A formatted prompt string.
    """
    if len(permutation_pattern) != 4 or not all(char.isdigit() for char in permutation_pattern):
        raise ValueError("Order string must be a string of four digits.")

    try:
        reordered_options = [options_list[int(index)] for index in permutation_pattern]
    except (IndexError, ValueError):
        raise ValueError("Order string contains invalid indices for the options list.")

    # Format the reordered options into a string
    options_string = "\n".join([f"- {option}" for option in reordered_options])

    # Build the prompt
    prompt = (
        "Your task is to extract an answer from some response text. "
        "The response was given in answer to a question with the following four possible answers, "
        "which were given in this order:\n\n"
        f"{options_string}\n\n"
        "Here is the text of the response:\n\n"
        "<response>\n"
        f"{response_text}\n"
        "</response>\n\n"
        "Which answer was given to the question? "
        "If none of the options was given as the answer, respond with \"unanswered.\"\n\n"
        "Respond with ONLY one of the following outputs:\n\n"
        f"{options_string}\n"
        "- unanswered"
    )
    return prompt

def extract_answer_by_llm(
    content_received, 
    options_list, 
    permutation_pattern, 
    model_mapping_file="model_mapping.json", 
    answer_extraction_model_name="claude-35-haiku"
):
    """
    Extract an answer from content using LLM assistance.
    
    Args:
        content_received (str): Content to extract answer from
        options_list (list): List of valid options
        permutation_pattern (str): Pattern of option presentation
        model_mapping_file (str): Path to model mapping file
        answer_extraction_model_name (str): Name of model to use for extraction
        
    Returns:
        dict: Dictionary containing extraction results and metadata
    """
    # Initialize chat history for human-readable summary
    chat_history = []

    # Load model mapping
    model_mapping = get_model_mapping(model_mapping_file)
    if answer_extraction_model_name not in model_mapping:
        raise ValueError(f"Model name '{answer_extraction_model_name}' not found in the model mapping file.")
    model_address = model_mapping[answer_extraction_model_name]

    # Build the prompt
    prompt_to_extract_answer = build_answer_extraction_prompt(content_received, options_list, permutation_pattern)

    # Initial user message
    chat_history.append({"role": "user", "content": prompt_to_extract_answer})
    
    # Call the LLM first attempt
    completion_params = {
        "model": model_address,
        "messages": [{"role": "user", "content": prompt_to_extract_answer}],
        "stream": False,
    }

    try:
        response = completion(**completion_params)
    except Exception as e:
        return {
            "llm_extract": None,
            "llm_extract_chat_history": "An error occurred during LLM completion on the first attempt.",
            "llm_extract_error": str(e),
            "llm_extract_model": answer_extraction_model_name
        }

    output_content_first_attempt = response["choices"][0]["message"]["content"]
    chat_history.append({"role": "assistant", "content": output_content_first_attempt})

    # Try extracting answer by rule from first attempt
    first_attempt_result = extract_answer_by_rule(output_content_first_attempt, options_list)
    if first_attempt_result:
        return {
            "llm_extract": first_attempt_result,
            "llm_extract_chat_history": "\n".join([f"{m['role']}: {m['content']}" for m in chat_history]),
            "llm_extract_error": None,
            "llm_extract_model": answer_extraction_model_name
        }

    # If first attempt didn't yield a valid answer, attempt a follow-up
    follow_up_message = (
        "Please respond with ONLY one of the options provided or 'unanswered'. "
        "Do not wrap your answer in tags or provide additional commentary. "
        "Please respond ONLY with one of the options or 'unanswered'."
    )
    chat_history.append({"role": "user", "content": follow_up_message})
    
    # Add follow-up message to messages list
    completion_params["messages"].append({
        "role": "assistant", 
        "content": output_content_first_attempt
    })
    completion_params["messages"].append({
        "role": "user", 
        "content": follow_up_message
    })

    try:
        response = completion(**completion_params)
    except Exception as e:
        return {
            "llm_extract": None,
            "llm_extract_chat_history": "\n".join([f"{m['role']}: {m['content']}" for m in chat_history]),
            "llm_extract_error": str(e),
            "llm_extract_model": answer_extraction_model_name
        }

    output_content_second_attempt = response["choices"][0]["message"]["content"]
    chat_history.append({"role": "assistant", "content": output_content_second_attempt})

    # Try extracting answer by rule from second attempt
    second_attempt_result = extract_answer_by_rule(output_content_second_attempt, options_list)

    return {
        "llm_extract": second_attempt_result,
        "llm_extract_chat_history": "\n".join([f"{m['role']}: {m['content']}" for m in chat_history]),
        "llm_extract_error": None,
        "llm_extract_model": answer_extraction_model_name
    }
