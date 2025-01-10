import pytest
from src.utils.general import (
    generate_permutations,
    get_permutation_pattern_from_result_number,
    generate_hash
)

def test_generate_permutations():
    """Test permutation generation"""
    perms = generate_permutations()
    assert len(perms) == 24  # 4! = 24 permutations
    assert len(set(perms)) == 24  # All unique
    assert all(len(p) == 4 for p in perms)  # All length 4
    assert all(sorted(p) == list('0123') for p in perms)  # All contain 0-3

def test_get_permutation_pattern():
    """Test permutation pattern selection"""
    perms = generate_permutations()
    
    # Test valid inputs
    assert get_permutation_pattern_from_result_number(perms, 1) == perms[0]
    assert get_permutation_pattern_from_result_number(perms, 24) == perms[23]
    assert get_permutation_pattern_from_result_number(perms, 25) == perms[0]
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        get_permutation_pattern_from_result_number(perms, 0)
    with pytest.raises(ValueError):
        get_permutation_pattern_from_result_number(perms, 121)

def test_generate_hash():
    """Test hash generation"""
    # Test deterministic output
    data1 = {"test": "data"}
    data2 = {"test": "data"}
    assert generate_hash(data1) == generate_hash(data2)
    
    # Test different inputs produce different hashes
    data3 = {"test": "different"}
    assert generate_hash(data1) != generate_hash(data3)
    
    # Test different types
    assert generate_hash([1, 2, 3]) != generate_hash({"1": 2, "3": 4})
    
    # Test hash is string of correct length (SHA-256 = 64 chars)
    assert len(generate_hash(data1)) == 64
    assert isinstance(generate_hash(data1), str)

