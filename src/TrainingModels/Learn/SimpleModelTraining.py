# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:57:59 2025

@author: NDT Lab
"""

import numpy as np

# Create a simple dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)  # Feature (independent variable)
y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])  # Target (dependent variable)

from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X, y)  # Fit the model to the data

# Make predictions
y_pred = model.predict(X)

import matplotlib.pyplot as plt

# Plot the data points
plt.scatter(X, y, color='blue', label='Data points')

# Plot the regression line
plt.plot(X, y_pred, color='red', label='Regression line')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Example')
plt.show()