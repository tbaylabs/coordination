"""
General utility functions for the LLM coordination experiments.
"""

import json
import hashlib
from itertools import permutations

def generate_permutations():
    """
    Generate 24 unique permutations of indices 0-3.
    
    Returns:
        list: List of permutation strings.
    """
    indices = [0, 1, 2, 3]
    return [''.join(map(str, p)) for p in permutations(indices)]

def get_permutation_pattern_from_result_number(permutations_list, result_number):
    """
    Selects a permutation from the list based on the modulo 24 of the given result ID.

    Args:
        permutations_list (list): List of 24 permutations.
        result_number (int): A number between 1 and 120.

    Returns:
        str: The selected permutation.
    """
    if not 1 <= result_number <= 120:
        raise ValueError("Result ID must be between 1 and 120.")
    
    # Calculate the index by taking modulo 24
    index = (result_number - 1) % 24
    return permutations_list[index]

def generate_hash(data):
    """
    Generate a deterministic hash from a dictionary or other data structure.
    
    Args:
        data: Data to hash (will be converted to JSON string)
        
    Returns:
        str: SHA-256 hash of the data
    """
    data_string = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_string.encode('utf-8')).hexdigest()

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

