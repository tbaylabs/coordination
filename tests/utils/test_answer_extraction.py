import pytest
from src.utils.answer_extraction import (
    extract_answer_by_rule,
    extract_answer_by_llm,
    build_answer_extraction_prompt,
    is_answer_valid
)

def test_is_answer_valid():
    """Test answer validation"""
    options = ["red", "blue", "green"]
    
    # Test valid answers
    assert is_answer_valid(options, "red") is True
    assert is_answer_valid(options, "blue") is True
    assert is_answer_valid(options, "unanswered") is True
    
    # Test invalid answers
    assert is_answer_valid(options, "yellow") is False
    assert is_answer_valid(options, "") is False
    assert is_answer_valid(options, "RED") is False  # Case sensitive

def test_extract_answer_by_rule():
    """Test rule-based answer extraction"""
    options = ["red", "blue", "green"]
    
    # Test direct matches
    assert extract_answer_by_rule("red", options) == "red"
    assert extract_answer_by_rule("I choose: red", options) == "red"
    
    # Test XML format
    assert extract_answer_by_rule("<answer>red</answer>", options) == "red"
    assert extract_answer_by_rule("<option>red</option>", options) == "red"
    
    # Test no match
    assert extract_answer_by_rule("yellow", options) is None
    assert extract_answer_by_rule("", options) is None

def test_build_answer_extraction_prompt():
    """Test answer extraction prompt building"""
    response_text = "I choose red"
    options = ["red", "blue", "green", "yellow"]
    pattern = "0123"
    
    prompt = build_answer_extraction_prompt(response_text, options, pattern)
    
    assert isinstance(prompt, str)
    assert "Your task is to extract an answer" in prompt
    assert response_text in prompt
    assert all(option in prompt for option in options)
    assert "unanswered" in prompt

def test_invalid_extraction_prompt():
    """Test invalid inputs for prompt building"""
    with pytest.raises(ValueError):
        build_answer_extraction_prompt(
            "test",
            ["red", "blue"],
            "012"  # Invalid pattern length
        )
