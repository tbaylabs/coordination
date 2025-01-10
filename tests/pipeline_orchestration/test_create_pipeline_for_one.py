import os
import json
import pytest
import shutil
from pathlib import Path

from src.pipeline_orchestration.create_pipeline_for_one import create_pipeline_for_one_condition

# Test fixtures
@pytest.fixture
def test_base_folder(request):
    """Provides a test folder and cleans it up after tests"""
    folder = "pipeline_test"
    os.makedirs(folder, exist_ok=True)
    
    def cleanup():
        if os.path.exists(folder):
            # Get the current test name
            current_test = request.node.name
            
            # Skip validation for tests that are expected to fail
            if current_test != "test_invalid_options_list":
                # Verify expected structure before deletion
                test_folders = [
                    os.path.join(folder, "test_1_data_collection"),
                    os.path.join(folder, "test_2_answer_extraction"),
                    os.path.join(folder, "test_3_results")
                ]
                
                # Check if test folders were created as expected
                for test_folder in test_folders:
                    assert os.path.exists(test_folder), f"Expected test folder missing: {test_folder}"
                    assert any(f.endswith('.json') for f in os.listdir(test_folder)), \
                        f"No JSON files found in {test_folder}"
            
            # Clean up
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

# Tests
def test_basic_pipeline_creation(test_base_folder, basic_prompt_conditions, basic_model_parameters):
    """Test basic pipeline creation with default parameters"""
    dc_path = create_pipeline_for_one_condition(
        basic_prompt_conditions,
        basic_model_parameters,
        base_folder=test_base_folder,
        test_mode=True
    )
    
    assert os.path.exists(dc_path)
    assert dc_path.endswith(".json")
    
    # Check if all three files were created
    base_path = os.path.dirname(os.path.dirname(dc_path))
    ae_path = dc_path.replace("test_1_data_collection", "test_2_answer_extraction").replace("dc_", "ae_")
    res_path = dc_path.replace("test_1_data_collection", "test_3_results").replace("dc_", "res_")
    
    assert os.path.exists(ae_path)
    assert os.path.exists(res_path)

def test_temperature_parameter(test_base_folder, basic_prompt_conditions, basic_model_parameters):
    """Test that temperature parameter is correctly handled in filename"""
    basic_model_parameters["temperature"] = 0.7
    dc_path = create_pipeline_for_one_condition(
        basic_prompt_conditions,
        basic_model_parameters,
        base_folder=test_base_folder,
        test_mode=True
    )
    
    assert "temp07" in dc_path

def test_xml_prompt_parameter(test_base_folder, basic_prompt_conditions, basic_model_parameters):
    """Test that XML prompt flag is correctly handled in filename"""
    basic_model_parameters["xml_prompt"] = True
    dc_path = create_pipeline_for_one_condition(
        basic_prompt_conditions,
        basic_model_parameters,
        base_folder=test_base_folder,
        test_mode=True
    )
    
    assert "xml_prompt" in dc_path

def test_hash_consistency(test_base_folder, basic_prompt_conditions, basic_model_parameters):
    """Test that identical inputs produce identical hashes"""
    dc_path1 = create_pipeline_for_one_condition(
        basic_prompt_conditions,
        basic_model_parameters,
        base_folder=test_base_folder,
        test_mode=True
    )
    
    with open(dc_path1, 'r') as f:
        hash1 = json.load(f)["pipeline-hash"]
    
    # Create another pipeline with same parameters
    dc_path2 = create_pipeline_for_one_condition(
        basic_prompt_conditions,
        basic_model_parameters,
        base_folder=test_base_folder,
        test_mode=True
    )
    
    with open(dc_path2, 'r') as f:
        hash2 = json.load(f)["pipeline-hash"]
    
    assert hash1 == hash2

def test_invalid_options_list(test_base_folder, basic_prompt_conditions, basic_model_parameters):
    """Test that invalid options list key raises ValueError"""
    basic_prompt_conditions["options_lists_key"] = "invalid_key"
    
    with pytest.raises(ValueError) as exc_info:
        create_pipeline_for_one_condition(
            basic_prompt_conditions,
            basic_model_parameters,
            base_folder=test_base_folder,
            test_mode=True
        )
    
    assert "Options list not found" in str(exc_info.value)

def test_default_pipeline_location(basic_prompt_conditions, basic_model_parameters):
    """Test that default pipeline location is in project root"""
    dc_path = create_pipeline_for_one_condition(
        basic_prompt_conditions,
        basic_model_parameters
    )
    
    # Get project root and check if path is relative to it
    project_root = Path(__file__).parent.parent.parent
    expected_base = project_root / "pipeline"
    
    assert Path(dc_path).parent.parent == expected_base

def test_file_content_structure(test_base_folder, basic_prompt_conditions, basic_model_parameters):
    """Test that created files have correct content structure"""
    dc_path = create_pipeline_for_one_condition(
        basic_prompt_conditions,
        basic_model_parameters,
        base_folder=test_base_folder,
        test_mode=True
    )
    
    with open(dc_path, 'r') as f:
        content = json.load(f)
    
    # Check required keys
    assert "pipeline-hash" in content
    assert "overview" in content
    assert "pipeline-paths" in content
    assert "data_collection_log" in content
    
    # Check overview content
    assert "model_name" in content["overview"]
    assert "temperature" in content["overview"]
    assert "options_list" in content["overview"]
    assert "example_prompt" in content["overview"]
