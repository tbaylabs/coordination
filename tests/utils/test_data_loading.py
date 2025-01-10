import pytest
import json
import os
from pathlib import Path
from src.utils.data_loading import (
    load_environment,
    load_prompt_components,
    get_model_mapping
)

def test_load_environment(monkeypatch):
    """Test environment loading"""
    # Mock environment variables
    test_vars = {
        "AZURE_AI_API_KEY": "test_key_12345678more",
        "OPENAI_API_KEY": "sk-test_key_12345678more",
        "ANTHROPIC_API_KEY": "test_key_12345678more"
    }
    
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    
    env_vars = load_environment()
    
    # Check redacted format
    for key in test_vars:
        if key in env_vars:
            assert env_vars[key].endswith("<redacted>")
            assert len(env_vars[key]) > 8

def test_load_prompt_components(tmp_path):
    """Test prompt components loading"""
    # Create temporary test files
    components = {
        "task": {"test": "instruction"},
        "options": ["A", "B", "C"],
        "reasoning": {"test": "reasoning"}
    }
    
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    
    (prompts_dir / "task_instruction_components.json").write_text(
        json.dumps(components["task"])
    )
    (prompts_dir / "options_lists.json").write_text(
        json.dumps(components["options"])
    )
    (prompts_dir / "reasoning_instruction_components.json").write_text(
        json.dumps(components["reasoning"])
    )
    
    # Test loading with wrong path
    with pytest.raises(FileNotFoundError):
        load_prompt_components("nonexistent/prompts")
    
    # Test loading from temp directory
    os.environ["PROMPTS_DIR"] = str(prompts_dir)
    task, options, reasoning = load_prompt_components()
    assert task == components["task"]
    assert options == components["options"]
    assert reasoning == components["reasoning"]
    del os.environ["PROMPTS_DIR"]

def test_get_model_mapping(tmp_path):
    """Test model mapping loading"""
    # Create test mapping file
    test_mapping = {
        "test-model": "test-address",
        "gpt-4": "azure/deployment"
    }
    
    mapping_file = tmp_path / "test_mapping.json"
    mapping_file.write_text(json.dumps(test_mapping))
    
    # Test loading
    mapping = get_model_mapping(mapping_file)
    assert mapping == test_mapping
    
    # Test missing file
    with pytest.raises(FileNotFoundError):
        get_model_mapping("nonexistent.json")
