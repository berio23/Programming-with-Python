"""
Finds the 4 ideal functions (out of 50) that best fit the training data
by comparing least-square errors. Results saved to JSON for the mapper.
"""

import pandas as pd
import json
from sqlalchemy import create_engine


def calculate_least_square_error(train_data, ideal_data):
    """
    Sum of squared differences between two series.
    Lower value means better fit.
    """
    return ((train_data - ideal_data) ** 2).sum()


if __name__ == "__main__":
    engine = create_engine('sqlite:///assignment.db')
    print("Connected to database")

    # Pull training and ideal data from DB
    print("\nLoading data...")
    training_df = pd.read_sql('training_data', con=engine)
    ideal_df = pd.read_sql('ideal_functions', con=engine)
    print(f"Got {len(training_df)} rows of training data")
    print(f"Got {len(ideal_df)} rows of ideal functions")
    
    # y-columns only (skip x)
    training_columns = [col for col in training_df.columns if col.lower() != 'x']
    ideal_columns = [col for col in ideal_df.columns if col.lower() != 'x']
    
    print(f"\nTraining functions: {training_columns}")
    print(f"Total ideal functions available: {len(ideal_columns)}")
    
    best_fit_results = {}
    
    print("\n" + "="*70)
    print("Finding best matches for each training function...")
    print("="*70)
    
    # For each training function, try all 50 ideal functions
    for train_col in training_columns:
        print(f"\nChecking {train_col}...")
        
        min_error = float('inf')
        best_ideal_func = None
        
        for ideal_col in ideal_columns:
            error = calculate_least_square_error(training_df[train_col], ideal_df[ideal_col])
            
            if error < min_error:
                min_error = error
                best_ideal_func = ideal_col
        
        # Also record max deviation -- needed for sqrt(2) criterion later
        max_deviation = abs(training_df[train_col] - ideal_df[best_ideal_func]).max()
        
        best_fit_results[train_col] = {
            'ideal_function': best_ideal_func,
            'least_square_error': float(min_error),
            'max_deviation': float(max_deviation)
        }
        
        print(f"  Best match: {best_ideal_func}")
        print(f"  Error: {min_error:.4f}")
        print(f"  Max deviation: {max_deviation:.4f}")
    
    # Persist to JSON so the mapper can pick it up
    output_file = 'best_fit_results.json'
    with open(output_file, 'w') as f:
        json.dump(best_fit_results, f, indent=4)
    
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    for train_func, result in best_fit_results.items():
        print(f"  {train_func} → {result['ideal_function']} (max dev: {result['max_deviation']:.4f})")
    
    print(f"\nSaved results to {output_file}")
    print("Done!")
    print("="*70)
