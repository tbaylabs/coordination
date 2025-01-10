import pytest
from src.utils.prompt_utils import build_prompt

def test_build_prompt_basic():
    """Test basic prompt building"""
    prompt = build_prompt(
        task_instruction_component_key="control",
        options_lists_key="colours",
        reasoning_instruction_component_key="none",
        permutation_pattern="0123"
    )
    assert isinstance(prompt, str)
    assert "Choose from the following options:" in prompt

def test_build_prompt_xml():
    """Test XML format prompt building"""
    prompt = build_prompt(
        task_instruction_component_key="control",
        options_lists_key="colours",
        reasoning_instruction_component_key="none",
        permutation_pattern="0123",
        return_xml=True
    )
    assert "<task>" in prompt
    assert "<instruction>" in prompt
    assert "<options>" in prompt
    assert "</task>" in prompt

def test_build_prompt_invalid_permutation():
    """Test handling of invalid permutation patterns"""
    with pytest.raises(ValueError):
        build_prompt(
            task_instruction_component_key="control",
            options_lists_key="colours",
            reasoning_instruction_component_key="none",
            permutation_pattern="012"  # Too short
        )
    
    with pytest.raises(ValueError):
        build_prompt(
            task_instruction_component_key="control",
            options_lists_key="colours",
            reasoning_instruction_component_key="none",
            permutation_pattern="0124"  # Invalid index
        )

def test_build_prompt_components():
    """Test prompt components are correctly included"""
    prompt = build_prompt(
        task_instruction_component_key="control",
        options_lists_key="colours",
        reasoning_instruction_component_key="step-by-step",
        permutation_pattern="0123"
    )
    
    # Check that all components are present
    assert "Choose from the following options:" in prompt
    assert any(emoji in prompt for emoji in ["🟩", "🟨", "🟧", "🟫"])
    assert "Think step-by-step" in prompt
