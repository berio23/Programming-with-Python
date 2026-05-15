"""
Main Mapper Script

takes test data and maps it to our chosen ideal functions.
Uses OOP with inheritance and has custom exception handling because the assignment requires it.

The sqrt(2) criterion: a test point gets mapped if its deviation is within max_threshold * sqrt(2).
If it passes, we save it to the database.
"""

import pandas as pd
import json
import math
from sqlalchemy import create_engine, MetaData, Table, insert


def check_mapping_criterion(deviation, max_threshold, sqrt_factor=math.sqrt(2)):
    """
    Check if a test point passes the sqrt(2) criterion.
    
    Basically just checks: is deviation <= threshold * sqrt(2)?
    
    Args:
        deviation: How far the test point is from the ideal function
        max_threshold: The max deviation we saw in training
        sqrt_factor: Multiplication factor (default is sqrt(2))
    
    Returns:
        bool: True if it passes, False if not
    """
    return deviation <= (max_threshold * sqrt_factor)


# Custom Exception Class
class MappingError(Exception):
    """
    Custom exception for when mapping goes wrong.
    Gets raised if we can't find an x value or something else breaks.
    """
    pass


# Base Class
class DatabaseManager:
    """
    Base class that handles database connection stuff.
    Other classes inherit from this to get database access.
    """
    
    def __init__(self, db_path='sqlite:///assignment.db'):
        """
        Set up database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.engine = create_engine(db_path)
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        print(f"Database connected: {db_path}")


# Main Class (inherits from DatabaseManager)
class DataProcessor(DatabaseManager):
    """
    Main class that does the actual data processing.
    Inherits from DatabaseManager to get database access.
    """
    
    def __init__(self, results_file='best_fit_results.json', test_file='test.csv'):
        """
        Initialize the processor.
        
        Args:
            results_file: JSON file with best fit results
            test_file: CSV file with test data
        """
        # Call parent constructor to get database connection
        super().__init__()
        
        print("\nSetting up data processor...")
        
        # Load the JSON file with our best fit results
        print(f"Loading {results_file}...")
        with open(results_file, 'r') as f:
            self.best_fit_data = json.load(f)
        
        # Pull out the ideal function names and their max deviations
        # We need these for the mapping criterion
        self.chosen_functions = {}
        for train_func, data in self.best_fit_data.items():
            ideal_func = data['ideal_function']
            max_dev = data['max_deviation']
            self.chosen_functions[ideal_func] = max_dev
        
        print(f"Loaded {len(self.chosen_functions)} best-fit functions:")
        for func, threshold in self.chosen_functions.items():
            print(f"  {func}: threshold = {threshold:.4f}")
        
        # Get the ideal functions table from database
        # Setting x as index makes lookups easier later
        print("\nLoading ideal functions...")
        self.ideal_df = pd.read_sql_table('ideal_functions', con=self.engine)
        self.ideal_df = self.ideal_df.set_index('x')
        print(f"Loaded {len(self.ideal_df)} rows")
        
        # Load test data from CSV
        print(f"\nLoading test data from {test_file}...")
        self.test_df = pd.read_csv(test_file)
        print(f"Loaded {len(self.test_df)} test points")
    
    def map_test_data(self):
        """
        Map test data to ideal functions using sqrt(2) criterion.
        
        Goes through each test point and checks if it fits any of our 4 chosen functions.
        
        Returns:
            list: List of dicts with mapped results
        """
        print("\n" + "="*70)
        print("Starting mapping process...")
        print("="*70)
        
        results_list = []
        mapped_count = 0
        unmapped_count = 0
        
        # Go through each test point
        for idx, row in self.test_df.iterrows():
            x_value = row['x']
            y_value = row['y']
            
            best_match = None
            
            try:
                # Check all 4 chosen ideal functions and keep the valid match with the smallest deviation
                for ideal_func_name, max_threshold in self.chosen_functions.items():
                    
                    # Make sure this x value exists in our ideal functions
                    if x_value not in self.ideal_df.index:
                        raise KeyError(f"x value {x_value} not found in ideal functions")
                    
                    # Look up what the ideal function predicts for this x
                    ideal_y_value = self.ideal_df.loc[x_value, ideal_func_name]
                    
                    # Calculate how far off we are
                    deviation = abs(y_value - ideal_y_value)
                    
                    # Apply the sqrt(2) criterion; if several functions pass, choose the closest one
                    if check_mapping_criterion(deviation, max_threshold):
                        func_number = int(ideal_func_name.replace("y", ""))
                        candidate = {
                            'X': float(x_value),
                            'Y': float(y_value),
                            'Delta_Y': float(deviation),
                            'No_of_ideal_func': func_number
                        }
                        
                        if best_match is None or candidate['Delta_Y'] < best_match['Delta_Y']:
                            best_match = candidate
                
                if best_match is not None:
                    results_list.append(best_match)
                    mapped_count += 1
                else:
                    unmapped_count += 1
                    
            except KeyError as e:
                raise MappingError(f"Failed to map x={x_value}: {e}")
            except Exception as e:
                raise MappingError(f"Unexpected error mapping x={x_value}, y={y_value}: {e}")
        
        print(f"\nMapping complete!")
        print(f"  Mapped: {mapped_count} points")
        print(f"  Unmapped: {unmapped_count} points")
        print(f"  Total: {len(self.test_df)} points")
        
        return results_list
    
    def save_results(self, results_list):
        """
        Save mapped results to the database.
        
        Args:
            results_list: List of result dicts to save
        """
        print("\n" + "="*70)
        print("Saving to database...")
        print("="*70)
        
        if not results_list:
            print("No results to save!")
            return
        
        # Get the mapped_results table
        mapped_results_table = Table('mapped_results', self.metadata, autoload_with=self.engine)
        
        # Clear old mapped results and insert the fresh run output
        with self.engine.begin() as connection:
            connection.execute(mapped_results_table.delete())
            connection.execute(insert(mapped_results_table), results_list)
            # Transaction commits automatically when the context exits
        
        print(f"Saved {len(results_list)} results to 'mapped_results' table")
        print("="*70)


# Main execution block
if __name__ == "__main__":
    print("="*70)
    print("Data Mapper - Test Data Processing")
    print("="*70)
    
    try:
        # Create processor instance
        processor = DataProcessor()
        
        # Map the test data
        mapped_data = processor.map_test_data()
        
        # Save results
        processor.save_results(mapped_data)
        
        print("\n" + "="*70)
        print("All done!")
        print("="*70)
        
    except MappingError as e:
        print(f"\nMapping Error: {e}")
    except FileNotFoundError as e:
        print(f"\nFile Not Found: {e}")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        raise
