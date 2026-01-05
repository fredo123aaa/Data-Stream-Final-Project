import numpy as np
import pandas as pd
from typing import Dict, Tuple
from river import base, optim
import matplotlib.pyplot as plt


class LMS(base.Regressor):
    """
    Least Mean Square (LMS) Algorithm using River
    Based on Section 3.2.1 of the paper
    Implements River's online learning interface
    """
    def __init__(self, learning_rate: float = 0.2):
        """
        Initialize LMS filter
        
        Args:
            learning_rate: Step size (η in the paper)
        """
        self.learning_rate = learning_rate
        self.weights = {}  # Will store weights for each feature
        self.errors = []
        
    def learn_one(self, x: dict, y: float) -> 'LMS':
        """
        Update the model with a single sample (online learning)
        
        Args:
            x: Dictionary of features {feature_name: value}
            y: Target value (desired output d_i)
            
        Returns:
            self
        """
        # Initialize weights for new features
        for feature in x:
            if feature not in self.weights:
                self.weights[feature] = 0.0
        
        # Predict with current weights
        y_pred = self.predict_one(x)
        
        # Calculate error: e_i = d_i - y_pred
        error = y - y_pred
        
        # Update weights: ϖ_i = ϖ_(i-1) + η * e_i * x_i
        for feature, value in x.items():
            self.weights[feature] += self.learning_rate * error * value
        
        self.errors.append(error ** 2)
        
        return self
    
    def predict_one(self, x: dict) -> float:
        """
        Predict output for a single sample
        
        Args:
            x: Dictionary of features
            
        Returns:
            Predicted value y = ϖ^T * x
        """
        prediction = 0.0
        for feature, value in x.items():
            if feature in self.weights:
                prediction += self.weights[feature] * value
        return prediction


class KLMS(base.Regressor):
    """Kernel Least Mean Square for online regression"""
    
    def __init__(self, eta=0.1, sigma=1.0, max_dictionary=500):
        """
        Parameters:
        -----------
        eta : float
            Step size / learning rate
        sigma : float
            Kernel width (for Gaussian kernel)
        max_dictionary : int
            Maximum number of centers to store
        """
        self.eta = eta
        self.sigma = sigma
        self.max_dictionary = max_dictionary
        self.centers = []
        self.coefficients = []
        
    def learn_one(self, x, y):
        """Update the model with a single sample"""
        x_array = self._dict_to_array(x)
        
        # Predict with current model
        y_pred = self.predict_one(x)
        
        # Calculate error
        error = y - y_pred
        
        # KLMS update: add new center with coefficient
        self.centers.append(x_array)
        self.coefficients.append(self.eta * error)
        
        # Memory management
        if len(self.centers) > self.max_dictionary:
            self.centers.pop(0)
            self.coefficients.pop(0)
        
        return self
    
    def predict_one(self, x):
        """Predict for a single sample"""
        if len(self.centers) == 0:
            return 0.0
        
        x_array = self._dict_to_array(x)
        
        # Sum of weighted kernels
        prediction = 0.0
        for center, coef in zip(self.centers, self.coefficients):
            kernel_value = self._gaussian_kernel(x_array, center)
            prediction += coef * kernel_value
        
        return prediction
    
    def _gaussian_kernel(self, x, y):
        """Compute Gaussian (RBF) kernel"""
        diff = x - y
        return np.exp(-np.sum(diff**2) / (2 * self.sigma**2))
    
    def _dict_to_array(self, x):
        """Convert River's dict format to numpy array"""
        if not hasattr(self, 'feature_names'):
            self.feature_names = sorted(x.keys())
        return np.array([x.get(f, 0.0) for f in self.feature_names])

class KAPA(base.Regressor):
    """
    Kernel Affine Projection Algorithm (KAPA)
    Based on Section 3.2.3 of the paper
    
    KAPA improves performance by reducing gradient noise through
    using multiple past samples (projection order K)
    """
    
    def __init__(self, eta=1.5, sigma=5.0, K=20, epsilon=1e-4, max_dictionary=500):
        """
        Initialize KAPA filter
        
        Parameters:
        -----------
        eta : float
            Step size (learning rate)
        sigma : float
            Kernel width for Gaussian kernel
        K : int
            Projection order (memory length) - number of past samples to use
        epsilon : float
            Regularization parameter to ensure numerical stability
        max_dictionary : int
            Maximum number of centers to store
        """
        self.eta = eta
        self.sigma = sigma
        self.K = K
        self.epsilon = epsilon
        self.max_dictionary = max_dictionary
        
        # Store centers and coefficients
        self.centers = []
        self.coefficients = []
        
        # Buffer for K most recent samples
        self.recent_inputs = []
        self.recent_targets = []
        
    def learn_one(self, x, y):
        """
        Update the model with a single sample
        
        Args:
            x: Dictionary of features
            y: Target value (desired output d_i)
            
        Returns:
            self
        """
        x_array = self._dict_to_array(x)
        
        # Add current sample to buffer
        self.recent_inputs.append(x_array)
        self.recent_targets.append(y)
        
        # Keep only K most recent samples
        if len(self.recent_inputs) > self.K:
            self.recent_inputs.pop(0)
            self.recent_targets.pop(0)
        
        # Build the observation matrix ψ(i) and target vector d(i)
        # ψ(i) = [φ(i-K+1), ..., φ(i)]
        psi_matrix = []  # Will store kernel evaluations
        d_vector = np.array(self.recent_targets)
        
        # Compute kernel matrix for recent samples
        for recent_x in self.recent_inputs:
            kernel_row = []
            for center in self.centers:
                kernel_val = self._gaussian_kernel(recent_x, center)
                kernel_row.append(kernel_val)
            
            # If no centers yet, use kernel with current sample
            if len(kernel_row) == 0:
                kernel_row = [1.0]  # κ(x, x) = 1 for normalized kernels
                
            psi_matrix.append(kernel_row)
        
        if len(self.centers) == 0:
            psi_matrix = [[1.0] for _ in self.recent_inputs]
        
        psi_matrix = np.array(psi_matrix)
        
        # Predict for all K recent samples
        predictions = []
        for recent_x in self.recent_inputs:
            pred = self._predict_array(recent_x)
            predictions.append(pred)
        predictions = np.array(predictions)
        
        # Calculate error vector: d(i) - ψ(i)^T * w_(i-1)
        error_vector = d_vector - predictions
        
        # KAPA update rule (equation 10 from paper):
        # w_i = w_(i-1) + η * ψ(i) * (d(i) - ψ(i)^T * w_(i-1))
        
        # Add new center (current input)
        self.centers.append(x_array.copy())
        
        # Compute the coefficient update
        # For KAPA, we update based on the affine projection
        if len(self.recent_inputs) > 0:
            # Use the most recent error for coefficient
            new_coef = self.eta * error_vector[-1]
            self.coefficients.append(new_coef)
        
        # Memory management: keep dictionary size bounded
        if len(self.centers) > self.max_dictionary:
            self.centers.pop(0)
            self.coefficients.pop(0)
        
        return self
    
    def predict_one(self, x):
        """
        Predict output for a single sample
        
        Args:
            x: Dictionary of features
            
        Returns:
            Predicted value
        """
        if len(self.centers) == 0:
            return 0.0
        
        x_array = self._dict_to_array(x)
        return self._predict_array(x_array)
    
    def _predict_array(self, x_array):
        """Internal prediction using numpy array"""
        if len(self.centers) == 0:
            return 0.0
        
        # f_i = Σ α_j(i) * κ(s_j, s)
        prediction = 0.0
        for center, coef in zip(self.centers, self.coefficients):
            kernel_value = self._gaussian_kernel(x_array, center)
            prediction += coef * kernel_value
        
        return prediction
    
    def _gaussian_kernel(self, x, y):
        """
        Compute Gaussian (RBF) kernel
        κ(s, s') = exp(-||s - s'||² / σ²)
        """
        diff = x - y
        return np.exp(-np.sum(diff**2) / (self.sigma**2))
    
    def _dict_to_array(self, x):
        """Convert River's dict format to numpy array"""
        if not hasattr(self, 'feature_names'):
            self.feature_names = sorted(x.keys())
        return np.array([x.get(f, 0.0) for f in self.feature_names])

