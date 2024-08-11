import numpy as np
from src.regularization import regularize_curves
from src.symmetry import detect_symmetry
from src.completion import handle_occlusion
from src.utils import polylines2svg

def main():
    input_path = 'data/input/sample_image.png'
    svg_output_path = 'data/output/output.svg'
    
    # Step 1: Regularize Curves
    lines, circles, polygons, stars = regularize_curves(input_path)
    
    # Step 2: Symmetry Detection
    symmetric_shapes = detect_symmetry(input_path)
    
    # Step 3: Curve Completion
    completed_curves = handle_occlusion(input_path)
    
    # Step 4: Visualization
    all_shapes = lines + circles + polygons + stars + symmetric_shapes + completed_curves
    polylines2svg(all_shapes, svg_output_path)

if __name__ == '__main__':
    main()
