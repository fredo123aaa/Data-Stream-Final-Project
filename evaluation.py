import pandas as pd
from river import base, optim
from typing import Dict, Tuple
from river import metrics

def evaluate_online(model, data_stream):
    """
    Online evaluation using test-then-train
    
    Parameters:
    -----------
    model : River model
        Model instance
    data_stream : generator
        Stream of (x, y) tuples
        
    Returns:
    --------
    results : dict
        Evaluation metrics and predictions
    """
    mae_metric = metrics.MAE()
    mse_metric = metrics.MSE()
    
    predictions = []
    actuals = []
    errors = []
    
    for x, y in data_stream:
        # TEST: Predict first
        y_pred = model.predict_one(x)
        
        # Update metrics
        mae_metric.update(y, y_pred)
        mse_metric.update(y, y_pred)
        
        # Store for DS calculation
        predictions.append(y_pred)
        actuals.append(y)
        errors.append((y - y_pred) ** 2)
        
        # TRAIN: Learn from this sample
        model.learn_one(x, y)
    
    # Calculate Directional Symmetry
    ds = calculate_directional_symmetry(actuals, predictions)
    
    return {
        'MAE': mae_metric.get(),
        'MSE': mse_metric.get(),
        'DS': ds,
        'predictions': predictions,
        'actuals': actuals,
        'errors': errors
    }


def calculate_directional_symmetry(actuals, predictions):
    """Calculate DS metric"""
    correct = 0
    total = 0
    
    for i in range(1, len(actuals)):
        actual_direction = actuals[i] - actuals[i-1]
        pred_direction = predictions[i] - predictions[i-1]
        
        if (actual_direction * pred_direction) >= 0:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0
