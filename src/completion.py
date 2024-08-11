import cv2
import numpy as np
from sklearn.impute import SimpleImputer

def complete_curve(curves, method='linear'):
    imputer = SimpleImputer(strategy=method)
    completed_curves = []
    for curve in curves:
        curve = curve.reshape(-1, 1)
        imputed_curve = imputer.fit_transform(curve)
        completed_curves.append(imputed_curve)
    return completed_curves

def handle_occlusion(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    completed_curves = complete_curve(contours)
    
    return completed_curves
