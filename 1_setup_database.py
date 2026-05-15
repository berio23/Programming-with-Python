"""
Sets up the SQLite database: loads train.csv and ideal.csv into their own
tables, then creates an empty mapped_results table for later use.
"""

import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Float, String, Integer

engine = create_engine('sqlite:///assignment.db')
print("Connected to database!")

# Load CSVs into their respective tables
print("\nLoading training data from CSV...")
train_df = pd.read_csv('train.csv')
train_df.to_sql('training_data', con=engine, if_exists='replace', index=False)
print(f"Done! Created 'training_data' table with {len(train_df)} rows")

print("\nLoading ideal functions from CSV...")
ideal_df = pd.read_csv('ideal.csv')
ideal_df.to_sql('ideal_functions', con=engine, if_exists='replace', index=False)
print(f"Done! Created 'ideal_functions' table with {len(ideal_df)} rows")

# Third table needs explicit column definitions for the mapper output
print("\nCreating empty table for mapped results...")
metadata = MetaData()

mapped_results = Table(
    'mapped_results',
    metadata,
    Column('X', Float, nullable=False),
    Column('Y', Float, nullable=False),
    Column('Delta_Y', Float, nullable=False),
    Column('No_of_ideal_func', Integer, nullable=False)
)

metadata.create_all(engine)
print("Done! Created 'mapped_results' table (empty for now)")

# Print summary
print("\n" + "="*50)
print("All tables created successfully!")
print("="*50)
print("Database: assignment.db")
print("Tables:")
print("  1. training_data")
print("  2. ideal_functions")
print("  3. mapped_results (empty)")
print("="*50)
