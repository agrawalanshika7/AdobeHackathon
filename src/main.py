import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import os
import numpy as np
import pandas as pd
from regularization import regularize_curves

from symmetry import detect_symmetry
from completion import handle_occlusion
from utils import polylines2svg

def process_file(file_path, output_path):
    # Step 1: Regularize Curves
    lines, circles, polygons, stars = regularize_curves(file_path)
    
    # Step 2: Symmetry Detection
    symmetric_shapes = detect_symmetry(file_path)
    
    # Step 3: Curve Completion
    completed_curves = handle_occlusion(file_path)
    
    # Combine all shapes for output
    all_shapes = lines + circles + polygons + stars + symmetric_shapes + completed_curves
    
    # Step 4: Save Output in CSV Format
    save_output_as_csv(all_shapes, output_path)
    
    # Optionally, you can also generate SVG for visualization
    svg_output_path = output_path.replace('.csv', '.svg')
    polylines2svg(all_shapes, svg_output_path)

def save_output_as_csv(shapes, output_path):
    with open(output_path, 'w') as f:
        for shape in shapes:
            for point in shape:
                f.write(','.join(map(str, point.flatten())) + '\n')
            f.write('\n')  # Separate shapes by a blank line

def main():
    input_dir = 'data/input/problems'
    output_dir = 'data/output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_file(input_path, output_path)
            print(f"Processed {filename} -> {output_path}")

if __name__ == '__main__':
    main()
