"""
Tests for pipeline runner functionality.
"""

import os
import json
import pytest
import shutil
from unittest.mock import patch
from pathlib import Path

from src.pipeline_orchestration import create_and_run_pipeline

@pytest.fixture
def test_base_folder(request):
    """Provides a test folder and cleans it up after tests"""
    folder = "pipeline_test_runner"
    os.makedirs(folder, exist_ok=True)
    
    def cleanup():
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"\nCleaned up test folder: {folder}")
    
    request.addfinalizer(cleanup)
    return folder

@pytest.fixture
def basic_prompt_conditions():
    """Basic prompt conditions for testing"""
    return {
        "task_instruction_component_key": "control",
        "options_lists_key": "colours",
        "reasoning_instruction_component_key": "none"
    }

@pytest.fixture
def basic_model_parameters():
    """Basic model parameters for testing"""
    return {
        "model_name": "test-model",
        "temperature": "default"
    }

@pytest.fixture
def mock_llm_response():
    """Mock successful LLM response"""
    return {
        "choices": [
            {
                "message": {
                    "content": "I choose: red"
                }
            }
        ]
    }

def test_create_and_run_pipeline(test_base_folder, basic_prompt_conditions, 
                               basic_model_parameters, mock_llm_response, tmp_path):
    """Test complete pipeline creation and execution"""
    
    # Create mock model mapping file
    mock_mapping = {
        "test-model": "test-deployment",
        "claude-35-haiku": "claude-3-5-haiku-20241022"
    }
    mock_mapping_path = tmp_path / "test_model_mapping.json"
    with open(mock_mapping_path, 'w') as f:
        json.dump(mock_mapping, f)
    
    # Mock the LLM completion call
    with patch('src.pipeline_orchestration.data_collector.completion') as mock_completion:
        # Configure mock to return our fake response
        mock_completion.return_value = mock_llm_response
        
        # Run the pipeline with test parameters
        pipeline_summary = create_and_run_pipeline(
            prompt_conditions=basic_prompt_conditions,
            model_parameters=basic_model_parameters,
            n=2,  # Small number for testing
            base_folder=test_base_folder,
            test_mode=True,
            model_mapping_file=str(mock_mapping_path)
        )
        
        # Verify the pipeline was created with correct structure
        test_folders = [
            os.path.join(test_base_folder, "test_1_data_collection"),
            os.path.join(test_base_folder, "test_2_answer_extraction"),
            os.path.join(test_base_folder, "test_3_results")
        ]
        
        for folder in test_folders:
            assert os.path.exists(folder)
            assert any(f.endswith('.json') for f in os.listdir(folder))
        
        # Verify pipeline execution results
        assert "data_collection" in pipeline_summary
        assert "answer_extraction" in pipeline_summary
        assert "results" in pipeline_summary
        
        # Check success counts
        assert pipeline_summary["data_collection"]["success_count"] > 0
        assert pipeline_summary["answer_extraction"]["success_count"] > 0
        assert pipeline_summary["results"] == "completed successfully"
        
        # Verify LLM was called the expected number of times
        assert mock_completion.call_count == 2  # Once for each n

def test_pipeline_error_handling(test_base_folder, basic_prompt_conditions, 
                               basic_model_parameters):
    """Test pipeline handles errors appropriately"""
    
    # Mock LLM to raise an exception
    with patch('src.pipeline_orchestration.data_collector.completion') as mock_completion:
        mock_completion.side_effect = Exception("API Error")
        
        # Run pipeline and check error handling
        pipeline_summary = create_and_run_pipeline(
            prompt_conditions=basic_prompt_conditions,
            model_parameters=basic_model_parameters,
            n=1,
            base_folder=test_base_folder,
            test_mode=True
        )
        
        # Verify error was captured in summary
        assert pipeline_summary["data_collection"]["error_logs"]
        assert pipeline_summary["results"] == "failed"
        assert "error" in pipeline_summary

def test_invalid_parameters(test_base_folder):
    """Test pipeline handles invalid parameters appropriately"""
    
    invalid_prompt_conditions = {
        "task_instruction_component_key": "invalid_key",
        "options_lists_key": "invalid_list",
        "reasoning_instruction_component_key": "none"
    }
    
    with pytest.raises(ValueError) as exc_info:
        create_and_run_pipeline(
            prompt_conditions=invalid_prompt_conditions,
            model_parameters={"model_name": "test-model"},
            n=1,
            base_folder=test_base_folder,
            test_mode=True
        )
    
    assert "Options list not found" in str(exc_info.value)
