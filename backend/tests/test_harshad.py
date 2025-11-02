"""
Unit tests for Harshad number functions.
"""

import pytest
from app.harshad import (
    is_harshad,
    digit_sum,
    factorial,
    first_nonharshad_factorial,
    find_consecutive_harshads,
    explain_max_consecutive
)


class TestDigitSum:
    """Test digit sum calculation."""
    
    def test_single_digit(self):
        assert digit_sum(5) == 5
        assert digit_sum(0) == 0
        assert digit_sum(9) == 9
    
    def test_multi_digit(self):
        assert digit_sum(18) == 9
        assert digit_sum(123) == 6
        assert digit_sum(999) == 27
    
    def test_negative(self):
        assert digit_sum(-18) == 9


class TestIsHarshad:
    """Test Harshad number detection."""
    
    def test_known_harshad(self):
        """Test known Harshad numbers."""
        harshad_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18, 20, 21, 24, 27]
        for num in harshad_numbers:
            assert is_harshad(num), f"{num} should be Harshad"
    
    def test_known_non_harshad(self):
        """Test known non-Harshad numbers."""
        non_harshad = [11, 13, 14, 15, 16, 17, 19, 22, 23, 25, 26, 28, 29]
        for num in non_harshad:
            assert not is_harshad(num), f"{num} should not be Harshad"
    
    def test_specific_cases(self):
        """Test specific important cases."""
        # 110, 111, 112 are consecutive Harshad numbers
        assert is_harshad(110)
        assert is_harshad(111)
        assert is_harshad(112)
        
        # 24 = 4! should be Harshad
        assert is_harshad(24)
    
    def test_zero_and_negative(self):
        """Test edge cases."""
        assert not is_harshad(0)
        assert not is_harshad(-5)


class TestFactorial:
    """Test factorial computation."""
    
    def test_small_factorials(self):
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(2) == 2
        assert factorial(3) == 6
        assert factorial(4) == 24
        assert factorial(5) == 120
    
    def test_larger_factorial(self):
        assert factorial(10) == 3628800


class TestFirstNonHarshadFactorial:
    """Test finding first non-Harshad factorial."""
    
    def test_first_non_harshad(self):
        """Test that we find the correct first non-Harshad factorial."""
        result = first_nonharshad_factorial()
        
        # Verify structure
        assert "k" in result
        assert "factorial_value" in result
        assert "is_harshad" in result
        assert "explanation" in result
        
        # Verify it's not Harshad
        assert result["is_harshad"] is False
        
        # Verify all previous factorials ARE Harshad
        k = result["k"]
        for i in range(1, k):
            fact_i = factorial(i)
            assert is_harshad(fact_i), f"{i}! = {fact_i} should be Harshad"
    
    def test_next_factorial(self):
        """Test that next factorial info is provided."""
        result = first_nonharshad_factorial()
        assert "next_factorial" in result
        assert "k" in result["next_factorial"]
        assert "is_harshad" in result["next_factorial"]


class TestConsecutiveHarshads:
    """Test finding consecutive Harshad numbers."""
    
    def test_three_consecutive(self):
        """Test finding 3 consecutive Harshad numbers (110, 111, 112)."""
        result = find_consecutive_harshads(3, start_hint=100)
        
        assert result["length"] == 3
        assert len(result["numbers"]) == 3
        
        # Verify all are Harshad
        for num in result["numbers"]:
            assert is_harshad(num)
        
        # Verify they are consecutive
        nums = result["numbers"]
        for i in range(len(nums) - 1):
            assert nums[i + 1] == nums[i] + 1
    
    def test_verification_structure(self):
        """Test that verification data is provided."""
        result = find_consecutive_harshads(3, start_hint=100)
        
        assert "verification" in result
        assert len(result["verification"]) == 3
        
        for item in result["verification"]:
            assert "number" in item
            assert "digit_sum" in item
            assert "is_harshad" in item
            assert item["is_harshad"] is True
    
    def test_ten_consecutive(self):
        """Test finding 10 consecutive Harshad numbers."""
        # This is computationally expensive, so we use a hint
        result = find_consecutive_harshads(10, start_hint=510500)
        
        assert result["length"] == 10
        assert len(result["numbers"]) == 10
        
        # Verify all are consecutive
        nums = result["numbers"]
        for i in range(len(nums) - 1):
            assert nums[i + 1] == nums[i] + 1
        
        # Verify all are Harshad
        for num in nums:
            assert is_harshad(num)
    
    def test_impossible_length(self):
        """Test that requesting 20+ consecutive raises error or returns explanation."""
        with pytest.raises(ValueError):
            find_consecutive_harshads(20)


class TestExplainMaxConsecutive:
    """Test explanation for maximum consecutive Harshad numbers."""
    
    def test_explanation_exists(self):
        """Test that explanation is non-empty."""
        explanation = explain_max_consecutive()
        assert len(explanation) > 100
        assert "20" in explanation or "consecutive" in explanation


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_factorial_harshad_sequence(self):
        """Test sequence of factorials for Harshad property."""
        # 1! through 3! should be Harshad
        for k in range(1, 4):
            fact = factorial(k)
            assert is_harshad(fact), f"{k}! = {fact} should be Harshad"
        
        # Find first non-Harshad
        result = first_nonharshad_factorial()
        k_non_harshad = result["k"]
        
        # Verify it's actually non-Harshad
        fact_non_harshad = int(result["factorial_value"])
        assert not is_harshad(fact_non_harshad)
