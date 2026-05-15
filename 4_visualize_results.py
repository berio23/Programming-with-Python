"""
Visualization: two Bokeh plots showing (1) training vs chosen ideal functions
and (2) mapped test points against those ideal functions.
Output goes to mapped_results.html.
"""

import pandas as pd
import json
from sqlalchemy import create_engine
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import column

engine = create_engine('sqlite:///assignment.db')
print("Connected to database")

# Load all data we need
print("\nLoading training data...")
train_df = pd.read_sql('training_data', con=engine)
print(f"Got {len(train_df)} training rows")

print("\nLoading mapped results...")
mapped_df = pd.read_sql('mapped_results', con=engine)
print(f"Got {len(mapped_df)} mapped points")

print("\nLoading ideal functions...")
ideal_df = pd.read_sql('ideal_functions', con=engine)
print(f"Got {len(ideal_df)} rows of ideal data")

print("\nLoading best fit results...")
with open('best_fit_results.json', 'r') as f:
    best_fit_data = json.load(f)

chosen_functions = [data['ideal_function'] for data in best_fit_data.values()]
training_columns = list(best_fit_data.keys())
print(f"Chosen functions: {chosen_functions}")
print(f"Training columns: {training_columns}")

color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
train_markers = ['circle', 'square', 'triangle', 'diamond']

output_file("mapped_results.html")
print("\nCreating plots...")

# Plot 1: Training scatter + ideal function lines
p1 = figure(
    title="Training Data vs. Chosen Ideal Functions (Best Fit)",
    x_axis_label="X",
    y_axis_label="Y",
    width=1000,
    height=600,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

for idx, col in enumerate(training_columns):
    train_source = ColumnDataSource({
        'x': train_df['x'],
        'y': train_df[col]
    })
    p1.scatter(
        'x',
        'y',
        source=train_source,
        size=6,
        color=color_palette[idx],
        alpha=0.4,
        legend_label=f'Training {col}',
        marker=train_markers[idx]
    )

for idx, func_name in enumerate(chosen_functions):
    ideal_source = ColumnDataSource({
        'x': ideal_df['x'],
        'y': ideal_df[func_name]
    })
    p1.line(
        'x',
        'y',
        source=ideal_source,
        line_width=2,
        color=color_palette[idx],
        alpha=0.8,
        legend_label=f'Ideal Function {func_name}'
    )

p1.legend.location = "top_left"
p1.legend.click_policy = "hide"
p1.grid.grid_line_alpha = 0.3

# Plot 2: Mapped test points + ideal function lines
p2 = figure(
    title="Mapped Test Data vs. Ideal Functions",
    x_axis_label="X",
    y_axis_label="Y",
    width=1000,
    height=600,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

mapped_source = ColumnDataSource(mapped_df)
p2.scatter(
    'X', 
    'Y', 
    source=mapped_source,
    size=8,
    color='black',
    alpha=0.6,
    legend_label='Mapped Test Data',
    marker='circle'
)

for idx, func_name in enumerate(chosen_functions):
    ideal_source = ColumnDataSource({
        'x': ideal_df['x'],
        'y': ideal_df[func_name]
    })
    p2.line(
        'x',
        'y',
        source=ideal_source,
        line_width=2,
        color=color_palette[idx],
        alpha=0.8,
        legend_label=f'Ideal Function {func_name}'
    )

p2.legend.location = "top_left"
p2.legend.click_policy = "hide"
p2.grid.grid_line_alpha = 0.3

print("Plots created!")

show(column(p1, p2))

