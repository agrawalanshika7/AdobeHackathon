import cv2
import numpy as np

def is_symmetric(points):
    # Check for symmetry in the points
    n = len(points)
    if n % 2 == 0:
        first_half = points[:n//2]
        second_half = points[n//2:]
        return np.allclose(first_half, second_half[::-1])
    return False

def detect_symmetry(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    symmetric_shapes = []
    for contour in contours:
        if is_symmetric(contour):
            symmetric_shapes.append(contour)
    
    return symmetric_shapes
