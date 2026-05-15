"""
Unit Tests

Tests for the main functions - least square error calculation and the mapping criterion.
Making sure everything works as expected before running on actual data.
"""

import unittest
import math
import pandas as pd
import numpy as np
import sys
import os
import importlib.util
from sqlalchemy import create_engine, MetaData, Table, Column, Float, Integer, insert, text

# Import the functions we want to test from files starting with numbers
# Can't do normal imports since Python doesn't like module names that start with numbers
# So using importlib as a workaround

# Load 2_find_best_functions.py
spec1 = importlib.util.spec_from_file_location("find_best_functions", "2_find_best_functions.py")
find_best_functions = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(find_best_functions)
calculate_least_square_error = find_best_functions.calculate_least_square_error

# Load 3_run_mapper.py
spec2 = importlib.util.spec_from_file_location("run_mapper", "3_run_mapper.py")
run_mapper = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(run_mapper)
check_mapping_criterion = run_mapper.check_mapping_criterion


class TestLeastSquareLogic(unittest.TestCase):
    """
    Tests for the least-square error function.
    Checking if it correctly calculates sum of squared differences.
    """
    
    def test_least_square_with_perfect_match(self):
        """Test least-square error when data is identical (should be 0)"""
        train_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ideal_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = calculate_least_square_error(train_data, ideal_data)
        
        self.assertEqual(result, 0.0, "Perfect match should have zero error")
    
    def test_least_square_with_known_values(self):
        """Test least-square error with known values"""
        # Simple test case where I know what the answer should be
        # train = [1, 2, 3], ideal = [1.5, 2.5, 3.5]
        # Differences are all 0.5, squared = 0.25 each, sum = 0.75
        train_data = pd.Series([1.0, 2.0, 3.0])
        ideal_data = pd.Series([1.5, 2.5, 3.5])
        
        result = calculate_least_square_error(train_data, ideal_data)
        expected = 0.75
        
        self.assertAlmostEqual(result, expected, places=5,
                              msg=f"Expected {expected}, got {result}")
    
    def test_least_square_with_larger_deviations(self):
        """Test least-square error with larger deviations"""
        # Another simple case: train = [0, 0, 0], ideal = [1, 2, 3]
        # Differences: [1, 2, 3], squared: [1, 4, 9], sum = 14
        train_data = pd.Series([0.0, 0.0, 0.0])
        ideal_data = pd.Series([1.0, 2.0, 3.0])
        
        result = calculate_least_square_error(train_data, ideal_data)
        expected = 14.0
        
        self.assertEqual(result, expected,
                        msg=f"Expected {expected}, got {result}")
    
    def test_least_square_with_negative_values(self):
        """Test least-square error with negative values"""
        train_data = pd.Series([-1.0, -2.0, -3.0])
        ideal_data = pd.Series([1.0, 2.0, 3.0])
        
        # Differences: [2, 4, 6], squared: [4, 16, 36], sum = 56
        result = calculate_least_square_error(train_data, ideal_data)
        expected = 56.0
        
        self.assertEqual(result, expected,
                        msg=f"Expected {expected}, got {result}")
    
    def test_least_square_returns_positive(self):
        """Test that least-square error is always non-negative"""
        # Just checking that we never get a negative error value
        train_data = pd.Series([10.0, -5.0, 3.2, -7.8])
        ideal_data = pd.Series([-3.0, 8.0, -2.1, 4.5])
        
        result = calculate_least_square_error(train_data, ideal_data)
        
        self.assertGreaterEqual(result, 0.0,
                               "Least-square error must be non-negative")


class TestMappingCriterion(unittest.TestCase):
    """
    Tests for the sqrt(2) mapping criterion.
    Making sure it correctly determines if a point should be mapped or not.
    """
    
    def test_mapping_criterion_passes(self):
        """Test that mapping passes when deviation is within threshold"""
        deviation = 1.0
        max_threshold = 1.0
        # 1.0 is less than 1.0 * 1.414, so should pass
        
        result = check_mapping_criterion(deviation, max_threshold)
        
        self.assertTrue(result, 
                       "Deviation within threshold should pass criterion")
    
    def test_mapping_criterion_fails(self):
        """Test that mapping fails when deviation exceeds threshold"""
        deviation = 2.0
        max_threshold = 1.0
        # 2.0 is bigger than 1.0 * 1.414, so should fail
        
        result = check_mapping_criterion(deviation, max_threshold)
        
        self.assertFalse(result,
                        "Deviation exceeding threshold should fail criterion")
    
    def test_mapping_criterion_exact_boundary(self):
        """Test mapping criterion at exact boundary"""
        max_threshold = 1.0
        deviation = max_threshold * math.sqrt(2)  # Right on the edge
        
        result = check_mapping_criterion(deviation, max_threshold)
        
        self.assertTrue(result,
                       "Deviation exactly at boundary should pass criterion")
    
    def test_mapping_criterion_just_below_boundary(self):
        """Test mapping criterion just below boundary"""
        max_threshold = 1.0
        deviation = max_threshold * math.sqrt(2) - 0.001  # Slightly under
        
        result = check_mapping_criterion(deviation, max_threshold)
        
        self.assertTrue(result,
                       "Deviation just below boundary should pass criterion")
    
    def test_mapping_criterion_just_above_boundary(self):
        """Test mapping criterion just above boundary"""
        max_threshold = 1.0
        deviation = max_threshold * math.sqrt(2) + 0.001  # Slightly over
        
        result = check_mapping_criterion(deviation, max_threshold)
        
        self.assertFalse(result,
                        "Deviation just above boundary should fail criterion")
    
    def test_mapping_criterion_zero_deviation(self):
        """Test mapping criterion with zero deviation (perfect match)"""
        deviation = 0.0
        max_threshold = 1.5
        
        result = check_mapping_criterion(deviation, max_threshold)
        
        self.assertTrue(result,
                       "Zero deviation should always pass criterion")
    
    def test_mapping_criterion_with_different_thresholds(self):
        """Test mapping criterion with various threshold values"""
        # Testing a bunch of different cases
        test_cases = [
            (1.5, 1.2, True),   # 1.5 is less than 1.2 * sqrt(2)
            (2.0, 1.2, False),  # 2.0 exceeds 1.2 * sqrt(2)
            (0.5, 0.5, True),   # 0.5 is less than 0.5 * sqrt(2)
            (1.0, 0.5, False),  # 1.0 exceeds 0.5 * sqrt(2)
        ]
        
        for deviation, threshold, expected in test_cases:
            with self.subTest(deviation=deviation, threshold=threshold):
                result = check_mapping_criterion(deviation, threshold)
                self.assertEqual(result, expected,
                               f"Failed for deviation={deviation}, threshold={threshold}")
    
    def test_mapping_criterion_custom_factor(self):
        """Test mapping criterion with custom multiplication factor"""
        deviation = 3.0
        max_threshold = 1.0
        custom_factor = 3.0  # Using 3.0 instead of sqrt(2)
        
        result = check_mapping_criterion(deviation, max_threshold, custom_factor)
        
        self.assertTrue(result,
                       "Should pass with custom factor of 3.0")


class TestIntegration(unittest.TestCase):
    """
    Integration tests - testing both functions together.
    Simulates a simplified version of the actual workflow.
    """
    
    def test_workflow_simulation(self):
        """
        Test a simplified workflow.
        Creates some fake data and runs through the whole process.
        """
        # Make some fake data to test with
        train = pd.Series([1.0, 2.0, 3.0, 4.0])
        ideal1 = pd.Series([1.1, 2.1, 3.1, 4.1])  # Pretty close
        ideal2 = pd.Series([5.0, 6.0, 7.0, 8.0])  # Way off
        
        # Calculate errors for both
        error1 = calculate_least_square_error(train, ideal1)
        error2 = calculate_least_square_error(train, ideal2)
        
        # The closer one should have a smaller error
        self.assertLess(error1, error2,
                       "Closer ideal function should have lower error")
        
        # Now test the mapping criterion with the better match
        max_deviation = abs(train - ideal1).max()  # Should be 0.1
        test_deviation = 0.14  # This should be within sqrt(2) * 0.1
        
        passes = check_mapping_criterion(test_deviation, max_deviation)
        self.assertTrue(passes,
                       "Test point should map to ideal function")


class TestDatabasePersistence(unittest.TestCase):
    """
    Tests the SQLite result table used by the Mapper.
    """
    def test_mapped_result_can_be_saved_and_read(self):
        """Test that mapped rows can be stored and queried again."""
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        mapped_results = Table(
            "mapped_results", metadata,
            Column("X", Float, nullable=False),
            Column("Y", Float, nullable=False),
            Column("Delta_Y", Float, nullable=False),
            Column("No_of_ideal_func", Integer, nullable=False),
        )
        metadata.create_all(engine)
        sample_row = {"X": 1.0, "Y": 2.0, "Delta_Y": 0.1, "No_of_ideal_func": 42}
        with engine.begin() as connection:
            connection.execute(insert(mapped_results), [sample_row])
            saved = connection.execute(
                text("SELECT X, Y, Delta_Y, No_of_ideal_func FROM mapped_results")
            ).fetchone()
        self.assertEqual(saved[3], 42)
        self.assertAlmostEqual(saved[2], 0.1)

# Test runner
if __name__ == "__main__":
    print("="*70)
    print("Running unit tests...")
    print("="*70)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)
