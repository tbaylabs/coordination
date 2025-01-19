"""
Utility functions for loading data and environment variables.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    """
    Load environment variables and validate API keys are accessible.
    Returns a list of accessible API keys.
    """
    print("LOADING ENVIRONMENT")
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / ".env.local"
    print(f"Looking for .env.local at: {env_path}")
    load_dotenv(env_path)
    
    env_vars = [
        "AZURE_AI_API_BASE",
        "AZURE_AI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "GITHUB_API_KEY",
        "GEMINI_API_KEY"
    ]

    accessible_vars = {}
    missing_vars = []
    for env_var in env_vars:
        value = os.getenv(env_var)
        if value:
            if len(value) >= 8:
                accessible_vars[env_var] = f"{value[:8]}<redacted>"
            else:
                print(f"Warning: {env_var} value is too short (length {len(value)})")
        else:
            missing_vars.append(env_var)
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")
        print(f"Current environment file: {env_path}")
        print(f"File exists: {env_path.exists()}")
    
    return accessible_vars

def load_prompt_components(prompts_dir=None):
    """
    Load base prompts, response instructions, and options sets from files in the prompts directory.

    Args:
        prompts_dir (str, optional): Directory containing prompt files. 
                                   If None, uses PROMPTS_DIR env var or 'prompts'.

    Returns:
        tuple: Dictionaries for base prompts, response instructions, and options sets.
    """
    if prompts_dir is None:
        root_dir = Path(__file__).parent.parent.parent
        prompts_dir = root_dir / 'prompts'
    else:
        prompts_dir = Path(prompts_dir)
    
    paths = {
        'task': prompts_dir / 'task_instruction_components.json',
        'options': prompts_dir / 'options_lists.json',
        'reasoning': prompts_dir / 'reasoning_instruction_components.json'
    }

    print(f"Looking for prompt files in: {prompts_dir}")
    try:
        print(f"Attempting to open: {paths['task']}")
        with open(paths['task'], 'r', encoding='utf-8') as f:
            task_instruction_components = json.load(f)
        with open(paths['options'], 'r', encoding='utf-8') as f:
            options_lists = json.load(f)
        with open(paths['reasoning'], 'r', encoding='utf-8') as f:
            reasoning_instruction_components = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find prompt files in {prompts_dir}") from e

    return task_instruction_components, options_lists, reasoning_instruction_components

def get_model_mapping(mapping_file_path="model_mapping.json"):
    """
    Loads model mapping from the given JSON file.
    Returns a dictionary of model name to model address mappings.
    """
    mapping_file_path = Path(mapping_file_path)
    if not mapping_file_path.exists():
        raise FileNotFoundError(f"The model mapping file {mapping_file_path} does not exist.")

    with open(mapping_file_path, 'r') as file:
        return json.load(file)
